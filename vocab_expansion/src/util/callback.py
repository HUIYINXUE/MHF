from transformers import TrainerCallback, TrainerControl, TrainerState, TrainingArguments
from typing import Dict, Any, Optional
import os
import json
import torch.distributed as dist

import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class StopOnCheckpointCallback(TrainerCallback):
    """
    A callback that stops training when a checkpoint is saved.

    Args:
        stop_after_n_checkpoints (int, optional): Stop training after this many checkpoints
            have been saved. Default is 1.
        stop_after_epoch (int, optional): Only stop training if a checkpoint is saved during or
            after this epoch. Default is None (epoch doesn't matter).
    """

    def __init__(self, stop_after_n_checkpoints: int = 1, stop_after_epoch: Optional[int] = None):
        self.stop_after_n_checkpoints = stop_after_n_checkpoints
        self.stop_after_epoch = stop_after_epoch
        self.checkpoint_count = 0
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()

    def on_save(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        """Event called after a checkpoint save."""
        self.checkpoint_count += 1

        # Check if epoch requirement is met (if any)
        epoch_condition_met = True
        if self.stop_after_epoch is not None:
            epoch_condition_met = state.epoch >= self.stop_after_epoch

        # Check if we've saved enough checkpoints
        should_stop = self.checkpoint_count >= self.stop_after_n_checkpoints and epoch_condition_met

        if should_stop and self.rank == 0:
            print(f"Checkpoint #{self.checkpoint_count} saved. Stopping training.")

        # Set stop flag for all ranks
        if should_stop:
            control.should_training_stop = True

        # Optional: Add barrier to ensure synchronization (usually unnecessary with Trainer)
        if dist.is_initialized():
            dist.barrier()

        return control


class SaveCheckpointAtStepCallback(TrainerCallback):
    """
    A callback that saves a checkpoint when a specific training step is reached.

    Args:
        save_steps (list): List of step numbers at which to save checkpoints.
        output_dir (str): Directory to save the checkpoints.
    """

    def __init__(self, save_steps: list, output_dir: str):
        self.save_steps = set(save_steps)
        self.output_dir = output_dir
        self.rank = 0 if not dist.is_initialized() else dist.get_rank()
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Will save additional checkpoints at steps: {self.save_steps}")

    def on_step_end(
            self,
            args: TrainingArguments,
            state: TrainerState,
            control: TrainerControl,
            **kwargs
    ):
        """Event called at the end of a training step."""
        if state.global_step in self.save_steps:
            # Only save on main process
            if self.rank == 0:
                logger.info(f"Saving checkpoint at step {state.global_step}")
                checkpoint_dir = os.path.join(self.output_dir, f"checkpoint-{state.global_step}")
                control.should_save = True

            # Synchronize processes
            if dist.is_initialized():
                dist.barrier()

        return control