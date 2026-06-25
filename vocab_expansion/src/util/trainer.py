## added CustomHashTrainer

import torch
from transformers import Trainer
from transformers.trainer import safe_globals
from transformers.trainer_pt_utils import set_rng_state_for_device
from transformers.training_args import ParallelMode
import logging
import os
import random
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


import torch.nn as nn
from transformers.trainer import deepspeed_sp_compute_loss
from transformers.trainer import _is_peft_model
from transformers.trainer import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from typing import TYPE_CHECKING, Any




class CustomTrainer(Trainer):
    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)


class CustomHashTrainer(Trainer):
    def _load_rng_state(self, checkpoint):
        # Load RNG states from `checkpoint`
        if checkpoint is None:
            return

        if self.args.world_size > 1:
            process_index = self.args.process_index
            rng_file = os.path.join(checkpoint, f"rng_state_{process_index}.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    f"Didn't find an RNG file for process {process_index}, if you are resuming a training that "
                    "wasn't launched in a distributed fashion, reproducibility is not guaranteed."
                )
                return
        else:
            rng_file = os.path.join(checkpoint, "rng_state.pth")
            if not os.path.isfile(rng_file):
                logger.info(
                    "Didn't find an RNG file, if you are resuming a training that was launched in a distributed "
                    "fashion, reproducibility is not guaranteed."
                )
                return

        with safe_globals():
            checkpoint_rng_state = torch.load(rng_file, weights_only=False)
        random.setstate(checkpoint_rng_state["python"])
        np.random.set_state(checkpoint_rng_state["numpy"])
        torch.random.set_rng_state(checkpoint_rng_state["cpu"])
        is_distributed = self.args.parallel_mode == ParallelMode.DISTRIBUTED
        if torch.cuda.is_available():
            set_rng_state_for_device("CUDA", torch.cuda, checkpoint_rng_state, is_distributed)

    def compute_loss(
        self,
        model: nn.Module,
        inputs: dict[str, torch.Tensor | Any],
        return_outputs: bool = False,
        num_items_in_batch: torch.Tensor | int | None = None,
    ) -> torch.Tensor | tuple[torch.Tensor, Any]:
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Args:
            model (`nn.Module`):
                The model to compute the loss for.
            inputs (`dict[str, torch.Tensor | Any]`):
                The input data for the model.
            return_outputs (`bool`, *optional*, defaults to `False`):
                Whether to return the model outputs along with the loss.
            num_items_in_batch (Optional[torch.Tensor], *optional*):
                The number of items in the batch. If not passed, the loss is computed
                using the default batch size reduction logic.

        Returns:
            The loss of the model along with its output if return_outputs was set to True

        Subclass and override for custom behavior. If you are not using `num_items_in_batch` when computing your loss,
        make sure to overwrite `self.model_accepts_loss_kwargs` to `False`. Otherwise, the loss calculation might be slightly inaccurate when performing gradient accumulation.
        """
        pc = getattr(self.accelerator, "parallelism_config", None)
        if pc is not None and pc.sp_backend == "deepspeed" and pc.sp_enabled and self.model.training:
            return deepspeed_sp_compute_loss(self.accelerator, model, inputs, return_outputs, pc)

        labels = inputs.pop("labels")
        if self.model_accepts_loss_kwargs:
            kwargs = {}
            if num_items_in_batch is not None:
                kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **kwargs}
        outputs = model(**inputs)

        # User-defined compute_loss function
        if labels is not None:
            logger.warning(
                "Trainer: `hash_loss_function` is defined but `labels=None`. "
                "Your custom loss function will still be called with labels=None. "
            )
            unwrapped_model = self.accelerator.unwrap_model(model)
            hash_labels = unwrapped_model.model.hash_map[labels].to(device=labels.device)
            hash_labels[labels == -100] = -100
            loss = self.hash_loss_func(
                outputs,
                hash_labels,
                num_items_in_batch=num_items_in_batch,
            )
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes if self.args.n_gpu <= 1 else self.args.n_gpu

        return (loss, outputs) if return_outputs else loss

    def hash_loss_func(self, outputs, labels, num_items_in_batch):
        if isinstance(outputs, dict):
            logits = outputs["logits"]
        else:
            if outputs[0] is None:
                logits = outputs[1]
            else:
                logits = outputs[0]
        logits = logits.float()
        num_hashes = logits.shape[-2]
        labels = nn.functional.pad(labels, (0, 0, 0, 1), value=-100)
        shift_labels = labels[..., 1:, :].contiguous()
        # Flatten the tokens
        shift_labels = shift_labels.view(-1)
        shift_labels = shift_labels.to(logits.device)
        reduction = "sum" if num_items_in_batch is not None else "mean"
        loss = nn.functional.cross_entropy(logits.view(-1, logits.shape[-1]), shift_labels, ignore_index=-100, reduction=reduction)
        if reduction == "sum":
            # just in case users pass an int for num_items_in_batch, which could be the case for custom trainer
            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(loss.device)
            loss = loss / num_items_in_batch
        else:
            loss = loss * num_hashes
        return loss



