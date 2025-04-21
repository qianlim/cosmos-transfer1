# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Usage (run from Cosmos-Transfer1 root directory):
    torchrun --nproc_per_node=8 -m scripts.convert_checkpoints_tp_to_fsdp > output.txt
    
This script is designed to convert a Tensor Parallel (TP) checkpoint
to a Fully Sharded Data Parallel (FSDP) compatible format for a video diffusion model.

Using experiment `BASE2B001_002_128N_LR-14_VideoImage_1-1` as an example:
For a model trained with Tensor Parallel (TP), the checkpoints are saved in the following formats:
```
edify_video4/BASE2B001/BASE2B001_002_128N_LR-14_VideoImage_1-1/checkpoints/iter_000250000_model_mp_0.pt
edify_video4/BASE2B001/BASE2B001_002_128N_LR-14_VideoImage_1-1/checkpoints/iter_000250000_model_mp_1.pt
...
```

where `*_model_mp_0.pt` and `*_model_mp_1.pt` are the model checkpoints for the two TP ranks.

This script will load the TP model checkpoint and convert it to a FSDP-compatible format.
The converted checkpoints will be saved
to a new directory `fsdp_checkpoints` under the same experiment directory, e.g.,
 `edify_video4/BASE2B001/BASE2B001_002_128N_LR-14_VideoImage_1-1/fsdp_checkpoints/`.

It has the following formats:
```
iter_000250000_reg_model.pt
iter_000250000_ema_model.pt
```
"""

import os

import torch
import torch.distributed as dist
from megatron.core import parallel_state

from imaginaire.utils import log
from projects.cosmos.diffusion.v1.config.base.vae import DummyJointImageVideoConfig
from projects.cosmos.diffusion.v1.model import finalize_model_grads
from projects.cosmos.diffusion.v1.tensor_parallel_test import (
    assert_close_gradients,
    copy_params_from_tp,
    get_video_batch,
)


def convert_tp_checkpoint_to_fsdp(experiment: str, checkpoint_path: str, output_directory: str) -> None:
    """
    Convert a Tensor Parallel (TP) checkpoint to a Fully Sharded Data Parallel (FSDP) compatible format.

    This function performs the following steps:
    1. Loads a TP model checkpoint
    2. Initializes a non-TP model
    3. Converts the checkpoint from TP format to FSDP compatible format
    4. Verifies the conversion by comparing outputs, losses, and gradients

    Args:
        experiment (str): The name of the experiment for which to convert the checkpoint.
        checkpoint_path (str): The path to the TP checkpoint file.
        output_directory (str): The directory where the converted FSDP checkpoint will be saved.

    Raises:
        ValueError: If the conversion process fails or if the verification step detects significant discrepancies.

    Note:
        This function assumes that the necessary configurations and dependencies are properly set up.
        It uses bfloat16 as the default dtype for better performance and memory efficiency.

    """
    log.info(f"Converting TP checkpoint to FSDP for experiment: {experiment}")
    from omegaconf import OmegaConf

    from imaginaire.lazy_config import LazyCall as L
    from imaginaire.utils import distributed
    from imaginaire.utils.config_helper import override
    from imaginaire.utils.easy_io import easy_io
    from imaginaire.utils.misc import set_random_seed
    from projects.cosmos.diffusion.v1.config.config import make_config
    from projects.cosmos.diffusion.v1.model import DiffusionModel
    from projects.edify_image.v4.train import instantiate_model

    # Clean up any existing parallel state
    parallel_state.destroy_model_parallel()

    # Set the default dtype to bfloat16 for better performance and memory efficiency
    dtype = torch.bfloat16
    torch.set_default_dtype(dtype)

    # Initialize and load the Tensor Parallel (TP) model
    config_tp = make_config()
    override_tp = [
        "--",
        f"experiment={experiment}",
        f"checkpoint.load_path={checkpoint_path}",
        "checkpoint.load_training_state=False",
    ]
    config_tp = override(
        config_tp,
        override_tp,
    )

    # TODO: (qsh 2024-08-17) Remove these temporary configurations once the real VAE is implemented
    OmegaConf.set_struct(DummyJointImageVideoConfig, False)
    config_tp.model.vae = DummyJointImageVideoConfig
    config_tp.model.vae.pixel_chunk_duration = 121
    config_tp.model.vae.latent_chunk_duration = 16

    # Configure for DDP (Distributed Data Parallel) and disable FSDP for now
    config_tp.trainer.distributed_parallelism = "ddp"
    config_tp.model.fsdp_enabled = False
    config_tp.model_obj = L(DiffusionModel)(config=None)

    # Set up S3 backend for checkpoint loading/saving
    easy_io.set_s3_backend(
        backend_args={
            "backend": "s3",
            "path_mapping": {
                "s3://rundir/": f"s3://checkpoints/{config_tp.job.path}/",
            },
            "s3_credential_path": config_tp.checkpoint.save_to_object_store.credentials,
        }
    )

    # Initialize trainer, model, optimizer, scheduler, and grad scaler for TP
    trainer_tp = config_tp.trainer.type(config_tp)
    tp_group = parallel_state.get_tensor_model_parallel_group()
    tp_size = parallel_state.get_tensor_model_parallel_world_size()
    global_tp_src_rank = parallel_state.get_tensor_model_parallel_src_rank()
    global_rank = dist.get_rank()

    # Set random seed by global rank to ensure diversity within TP groups
    set_random_seed(global_rank)
    model_tp = instantiate_model(config_tp, trainer_tp).cuda()
    optimizer_tp, scheduler_tp = model_tp.init_optimizer_scheduler(config_tp.optimizer, config_tp.scheduler)
    grad_scaler_tp = torch.amp.GradScaler("cuda", **config_tp.trainer.grad_scaler_args)

    # Load checkpoint and prepare model for training
    trainer_tp.checkpointer.load(model_tp, optimizer_tp, scheduler_tp, grad_scaler_tp)
    model_tp.on_train_start()
    model_tp_ddp = distributed.parallel_model_wrapper(config_tp.trainer.ddp, model_tp)

    # We intentionally set random seed by global rank to ensure ranks in the same TP group
    # have different random seeds. Our model broadcast data/noises so this should not affect the results.
    # We also re-generate the batch to ensure sync works.
    set_random_seed(global_rank)
    data_batch = get_video_batch(num_frames=121)
    if global_rank != global_tp_src_rank:
        data_batch = get_video_batch(num_frames=121)  # Re-generate batch to ensure sync

    # Run forward pass with TP model
    log.info("Running model_tp_ddp")
    set_random_seed(global_rank)
    output_batch_with_tp, loss_with_tp = model_tp_ddp.training_step(data_batch, iteration=0)

    # Extract and verify x0 predictions and losses across TP ranks
    x0_pred_with_tp_B_Q_T_H_W = output_batch_with_tp["model_pred"].x0
    gathered_loss_with_tp = [torch.randn_like(loss_with_tp) for _ in range(tp_size)]
    dist.all_gather(gathered_loss_with_tp, loss_with_tp, tp_group)
    torch.testing.assert_close(gathered_loss_with_tp[0], gathered_loss_with_tp[1])
    gathered_x0_pred_with_tp = [torch.randn_like(x0_pred_with_tp_B_Q_T_H_W) for _ in range(tp_size)]
    dist.all_gather(gathered_x0_pred_with_tp, x0_pred_with_tp_B_Q_T_H_W, tp_group)
    torch.testing.assert_close(gathered_x0_pred_with_tp[0], gathered_x0_pred_with_tp[1])

    # Perform backward pass and finalize gradients for TP model
    loss_with_tp.backward()
    finalize_model_grads([model_tp])

    # Store gradients for later comparison
    names = []
    grads_with_tp = []
    for name, p in model_tp.named_parameters():
        if p.grad is not None:
            grads_with_tp.append(p.grad.clone())
            names.append(name)

    # Clear gradients to prepare for non-TP model
    model_tp.zero_grad()

    # Initialize and prepare the non-TP model
    parallel_state.destroy_model_parallel()

    config = make_config()
    config = override(
        config,
        [
            "--",
            f"experiment={experiment}",
            "ckpt_klass=multi_rank",
            "checkpoint.load_from_object_store.enabled=False",
            "checkpoint.load_path=''",
            "upload_reproducible_setup=False",
            "model_parallel.tensor_model_parallel_size=1",
            "model_parallel.sequence_parallel=False",
        ],
    )

    OmegaConf.set_struct(DummyJointImageVideoConfig, False)
    config.model.vae = DummyJointImageVideoConfig
    config.model.vae.pixel_chunk_duration = 121
    config.model.vae.latent_chunk_duration = 16

    # Configure for DDP and disable FSDP for now
    config.trainer.distributed_parallelism = "ddp"
    config.model.fsdp_enabled = False
    config.model_obj = L(DiffusionModel)(config=None)

    # Initialize non-TP model and copy parameters from TP model
    trainer = config.trainer.type(config)
    model = instantiate_model(config, trainer).cuda()
    model.on_train_start()
    copy_params_from_tp(model, model_tp, tp_size=tp_size)
    model_ddp = distributed.parallel_model_wrapper(config.trainer.ddp, model)

    # To test the correctness of TP, we will sync the input data across all ranks.
    set_random_seed(global_rank)
    data_batch = get_video_batch(num_frames=121)
    for value in data_batch.values():
        if isinstance(value, torch.Tensor):
            dist.broadcast(value, global_tp_src_rank, group=tp_group)

    # Run `training_step`. Since diffusion training involves noise sampling, set the random seed to ensure
    # we sample the same noise across all ranks.
    log.info("Running model_ddp")
    set_random_seed(global_tp_src_rank)
    output_batch_without_tp, loss_without_tp = model_ddp.training_step(data_batch, iteration=0)

    # Extract and verify x0 predictions and losses for non-TP model
    x0_pred_without_tp_B_Q_T_H_W = output_batch_without_tp["model_pred"].x0
    gathered_loss = [torch.randn_like(loss_without_tp) for _ in range(parallel_state.get_data_parallel_world_size())]
    dist.all_gather(gathered_loss, loss_without_tp, group=tp_group)
    torch.testing.assert_close(gathered_loss[0], gathered_loss[1])
    gathered_x0_pred = [torch.randn_like(x0_pred_without_tp_B_Q_T_H_W) for _ in range(dist.get_world_size())]
    dist.all_gather(gathered_x0_pred, x0_pred_without_tp_B_Q_T_H_W, group=tp_group)
    torch.testing.assert_close(gathered_x0_pred[0], gathered_x0_pred[1])

    # Perform backward pass for non-TP model
    loss_without_tp.backward()

    # Store gradients for comparison
    names = []
    grads_without_tp = []
    for name, p in model.named_parameters():
        if p.grad is not None:
            grads_without_tp.append(p.grad.clone())
            names.append(name)

    # Compare outputs between TP and non-TP models
    tols = dict(atol=4.5e-2, rtol=2e-2)
    torch.testing.assert_close(x0_pred_without_tp_B_Q_T_H_W, x0_pred_with_tp_B_Q_T_H_W, **tols)
    log.info("Outputs match", rank0_only=False)

    # Compare losses between TP and non-TP models
    torch.testing.assert_close(loss_with_tp, loss_without_tp)
    log.info("Losses match", rank0_only=False)

    # Compare gradients between TP and non-TP models
    tols = dict(atol=1.2e-1, rtol=1.6e-2)
    assert_close_gradients(
        grads_with_tp, grads_without_tp, tols=tols, names=names, verbose=True, tp_group=tp_group, tp_size=tp_size
    )
    log.info("Gradients match", rank0_only=False)

    # Clean up parallel state
    parallel_state.destroy_model_parallel()

    # Save the converted model checkpoints
    if torch.distributed.get_rank() == 0:
        # Save regular model checkpoint
        checkpoint_name = os.path.basename(checkpoint_path)
        reg_model_checkpoint_name = checkpoint_name.replace(".pt", "_reg_model.pt")
        reg_model_path = os.path.join("s3://checkpoints-us-east-1", output_directory, reg_model_checkpoint_name)
        easy_io.dump(model.state_dict()["model"], reg_model_path)

        # Save EMA model checkpoint with necessary post-processing
        ema_state_dict = {k.replace("-", "."): v for k, v in model.state_dict()["ema"].items()}
        for key in ["net.pos_embedder.seq", "logvar.0.freqs", "logvar.0.phases"]:
            ema_state_dict[key] = model.state_dict()["model"][key]
        ema_model_checkpoint_name = checkpoint_name.replace(".pt", "_ema_model.pt")
        ema_model_path = os.path.join("s3://checkpoints-us-east-1", output_directory, ema_model_checkpoint_name)
        easy_io.dump(ema_state_dict, ema_model_path)

        log.info(
            f"Conversion complete. FSDP-compatible checkpoints saved for experiment: {experiment}\n"
            f"Regular model saved at {reg_model_path}\n"
            f"EMA model saved at {ema_model_path}"
        )


if __name__ == "__main__":
    experiment = "BASE002_003_512N_LR-143_VideoImage_1-1"
    checkpoint_path = "edify_video4/BASE002/BASE002_003_512N_LR-143_VideoImage_1-1/checkpoints/iter_000058000.pt"
    output_directory = "edify_video4/BASE002/BASE002_003_512N_LR-143_VideoImage_1-1/fsdp_checkpoints/"
    convert_tp_checkpoint_to_fsdp(experiment, checkpoint_path, output_directory)
