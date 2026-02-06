# Adopted from https://github.com/lm-sys/FastChat. Below is the original copyright:
# Adopted from tatsu-lab@stanford_alpaca. Below is the original copyright:
#    Copyright 2023 Rohan Taori, Ishaan Gulrajani, Tianyi Zhang, Yann Dubois, Xuechen Li
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.

import os
import copy
from dataclasses import dataclass, field
import json
import logging
import pathlib
from typing import Dict, Optional, Sequence, List
import torch.distributed as dist
import torch

import transformers
import tokenizers
from llava.conversation import conv_templates
from llava.constants import IGNORE_INDEX, IMAGE_TOKEN_INDEX, DEFAULT_IMAGE_TOKEN, DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN
from torch.utils.data import Dataset
from llava.train.llava_trainer import LLaVATrainer
from llava.train.test_case_factory import TestCaseFactory
from llava import conversation as conversation_lib
from llava.model import *
from llava.mm_utils import tokenizer_image_token, process_images
from PIL import Image
import re
from transformers import TrainerCallback
import random
import numpy as np
from transformers import set_seed as hf_set_seed
import json
from torchvision.transforms import Compose, Resize, CenterCrop, Normalize, InterpolationMode
from torchvision.transforms.functional import to_pil_image
import torch.nn.functional as F

def set_seed(seed=17):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True, warn_only=True)
    hf_set_seed(seed)

set_seed(17)


local_rank = None


def rank0_print(*args):
    if local_rank == 0:
        print(*args)


from packaging import version
IS_TOKENIZER_GREATER_THAN_0_14 = version.parse(tokenizers.__version__) >= version.parse('0.14')

def extract_id_from_filename(filename):
    # Example: sample_23.png -> 23
    match = re.search(r'(\d+)', filename)
    if match:
        return int(match.group(1))
    else:
        raise ValueError(f"Cannot extract id from filename: {filename}")
        
@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="facebook/opt-125m")
    version: Optional[str] = field(default="v0")
    freeze_backbone: bool = field(default=False)
    tune_mm_mlp_adapter: bool = field(default=False)
    vision_tower: Optional[str] = field(default=None)
    mm_vision_select_layer: Optional[int] = field(default=-1)   # default to the last layer
    pretrain_mm_mlp_adapter: Optional[str] = field(default=None)
    mm_projector_type: Optional[str] = field(default='linear')
    mm_use_im_start_end: bool = field(default=False)
    mm_use_im_patch_token: bool = field(default=True)
    mm_patch_merge_type: Optional[str] = field(default='flat')
    mm_vision_select_feature: Optional[str] = field(default="patch")


@dataclass
class DataArguments:
    train_data_path: str = field(default=None,
                           metadata={"help": "Path to the training data."})
    inference_data_path: str = field(default=None,
                           metadata={"help": "Path to the inference data."})
    image_path_1: str = field(default=None,
                           metadata={"help": "Path to the image data."})
    image_path_2: str = field(default=None,
                           metadata={"help": "Path to the image data."})
    image_path_3: str = field(default=None,
                           metadata={"help": "Path to the image data."})
    lazy_preprocess: bool = False
    is_multimodal: bool = False
    image_folder: Optional[str] = field(default=None)
    image_aspect_ratio: str = 'square'

@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    remove_unused_columns: bool = field(default=False)
    freeze_mm_mlp_adapter: bool = field(default=False)
    mpt_attn_impl: Optional[str] = field(default="triton")
    model_max_length: int = field(
        default=512,
        metadata={
            "help":
            "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    double_quant: bool = field(
        default=True,
        metadata={"help": "Compress the quantization statistics through double quantization."}
    )
    quant_type: str = field(
        default="nf4",
        metadata={"help": "Quantization data type to use. Should be one of fp4 or nf4."}
    )
    bits: int = field(
        default=16,
        metadata={"help": "How many bits to use."}
    )
    lora_enable: bool = False
    lora_r: int = 64
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    lora_weight_path: str = ""
    lora_bias: str = "none"
    mm_projector_lr: Optional[float] = None
    group_by_modality_length: bool = field(default=False)
    save_adversarial_images: bool = field(default=False)
    adversarial_images_dir: str = field(default="./adversarial_images")
    element_weight: float = field(default=10.0, metadata={"help": "Weight for ELEMENT tokens"})

class DifferentiableCLIPPreprocess(torch.nn.Module):
    def __init__(
        self,
        size=336,
        crop_size=336,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
        do_normalize=True,
    ):
        super().__init__()
        self.size = size
        self.crop_size = crop_size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
        self.do_normalize = do_normalize

    def forward(self, img):
        # img: torch.Tensor, shape [C, H, W] or [B, C, H, W]
        img = img.float()
        if img.dim() == 3:
            # Single image
            imgs = img.unsqueeze(0)  # [1, C, H, W]
            is_batch = False
        elif img.dim() == 4:
            # Batch of images
            imgs = img
            is_batch = True
        else:
            raise ValueError(f"Unexpected img shape: {img.shape}")

        processed_imgs = []
        for single_img in imgs:
            c, h, w = single_img.shape
            scale = self.size / min(h, w)
            new_h, new_w = round(h * scale), round(w * scale)
            temp_img = single_img.unsqueeze(0)  # add batch
            temp_img = F.interpolate(temp_img, size=(new_h, new_w), mode='bicubic', align_corners=False, antialias=True)
            temp_img = temp_img.squeeze(0)
            # Center crop to self.crop_size
            top = (new_h - self.crop_size) // 2
            left = (new_w - self.crop_size) // 2
            temp_img = temp_img[:, top:top+self.crop_size, left:left+self.crop_size]
            if self.do_normalize:
                mean = self.mean.to(temp_img.device)
                std = self.std.to(temp_img.device)
                temp_img = (temp_img - mean) / std
            processed_imgs.append(temp_img)
        processed_imgs = torch.stack(processed_imgs)
        if not is_batch:
            return processed_imgs.squeeze(0)
        return processed_imgs

class AdversarialPerturbation(torch.nn.Module):
    def __init__(self, shape, device, processor):
        """
        shape: (C, H, W) of the original image
        device: torch device
        processor: image processor (e.g., CLIPImageProcessor)
        """
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(1, 3, 295, 323, device=device))

        self.device = device

        # CLIP's normalization values
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        # Target size from processor
        self.target_size = processor.size['shortest_edge']
        print(f"Target size: {self.target_size}")

        # Compose pipeline for tensor input (skip ToTensor)
        self.transform = DifferentiableCLIPPreprocess(
            size=self.target_size,
            crop_size=self.target_size,
            mean=self.mean,
            std=self.std,
            do_normalize=True,
        )

        self.current_epoch = None

    def resize(self, image_tensor):
        batched = image_tensor.dim() == 4
        x = image_tensor if batched else image_tensor.unsqueeze(0)

        if isinstance(self.target_size, int):
            size = (self.target_size, self.target_size)
        else:
            size = self.target_size

        y = F.interpolate(
            x,
            size=size,
            mode='bicubic',
            align_corners=False,
            antialias=True,
        )
        return y if batched else y.squeeze(0)

    def forward(self, image_tensor, target_boxes, epoch=None):    
        """
        image_tensor: [B, 3, H, W] clean images (values in [0,1])
        target_boxes: tensor/list of length B with (x, y, h, w) in image coordinates
                      expect shape [B,4] with ints
        Returns: preprocessed tensor ready for model consumption
        """
        if image_tensor.dim() == 3:
            image_tensor = image_tensor.unsqueeze(0)  # [1, 3, H, W]
        delta = self.delta.to(image_tensor.device, dtype=image_tensor.dtype)
        perturbed = image_tensor.clone()
        B, C, H, W = image_tensor.shape

        # For each image in batch, insert resized shared patch at its box
        for i in range(B):
            x, y, h_box, w_box = target_boxes[i]  # assuming (x, y, h, w)
            # clamp coords to image bounds to avoid indexing errors
            x1 = int(x)
            y1 = int(y)
            h_box = 295
            w_box = 323
            x2 = min(x1 + w_box, W)
            y2 = min(y1 + h_box, H)

            patch_resized = self.delta.squeeze(0)
            perturbed[i, :, y1:y2, x1:x2] += patch_resized

        perturbed = self.resize(perturbed)
        processed = self.transform(perturbed)
        processed = processed.to(dtype=torch.bfloat16)
        return processed

class AdversarialTrainer(LLaVATrainer):
    def training_step(self, model, inputs):
        if 'images' in inputs:
            # Save the clean image before perturbing
            clean = inputs["images"].clone().detach()

            # Apply the adversarial perturbation (this is the only call!)
            perturbed = model.adversarial(clean)
            inputs["images"] = perturbed

            # Print the norm of the perturbation
            diff_norm = (perturbed - clean).abs().sum().item()
            print("delta diff norm:", diff_norm)

        return super().training_step(model, inputs)
from deepspeed.runtime.zero.partition_parameters import GatheredParameters

class DebugTrainer(AdversarialTrainer):

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def training_step(self, model, inputs):
        # PGD parameters (Hardcoded for now, consider moving to TrainingArguments)
        pgd_alpha = 1/255  # PGD step size for delta update
        pgd_epsilon = 16/255 # PGD L-infinity constraint for delta
        pgd_steps = 3  # Number of PGD steps per training_step

        clean_images_for_pgd = None # To store clean images for the PGD update step

        # 1) Store clean images for PGD
        if "images" in inputs:
            clean_images_for_pgd = inputs["images"].clone().detach()
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0
        # Print initial delta stats
        if hasattr(model, 'adversarial') and hasattr(model.adversarial, 'delta'):
            delta_param = model.adversarial.delta
            if self.accelerator.is_main_process:
                if delta_param.data.numel() > 0:
                    print(f"[DEBUG] Initial delta L1 norm: {delta_param.data.abs().sum().item():.6f}, L-inf norm: {delta_param.data.abs().max().item():.6f}")
                else:
                    print("[DEBUG] Initial delta is empty!")

        # 2) Multi-step PGD loop
        for step in range(pgd_steps):
            # Apply current delta to clean images
            if clean_images_for_pgd is not None:
                # Determine patch positions using image_id if available
                if "image_id" in inputs:
                    image_ids = inputs["image_id"]
                    patch_positions = []
                    for id in image_ids:
                        idx = int(id) % 2
                        if idx == 0:
                            patch_positions.append([730, 50])
                        elif idx == 1:
                            patch_positions.append([1085, 50])
                    patch_boxes = torch.tensor([[x, y, 323, 295] for x, y in patch_positions], device=clean_images_for_pgd.device)
                perturbed_images_for_loss = model.adversarial(clean_images_for_pgd, patch_boxes, epoch=current_epoch)
                inputs["images"] = perturbed_images_for_loss
                inputs["images"].requires_grad_(True)

            # Prepare inputs and compute loss
            model.train()
            batch_inputs = self._prepare_inputs(inputs)
            if "image_id" in batch_inputs:
                del batch_inputs["image_id"]
            loss = self.compute_loss(model, batch_inputs)

            # Backward pass for delta only
            try:
                loss.backward()
                if hasattr(model, 'adversarial') and hasattr(model.adversarial, 'delta'):
                    delta_grad = model.adversarial.delta.grad
                    if delta_grad is not None and self.accelerator.is_main_process:
                        print(f"[DEBUG] delta.grad (min={delta_grad.min().item():.6g}, max={delta_grad.max().item():.6g}, mean={delta_grad.mean().item():.6g}, l1={delta_grad.abs().sum().item():.6g})")
            except Exception as e:
                if self.accelerator.is_main_process:
                    print(f"ERROR during backward pass: {e}")
                    import traceback
                    traceback.print_exc()
                raise

            # PGD update for delta
            if hasattr(model, 'adversarial') and \
               hasattr(model.adversarial, 'delta') and \
               clean_images_for_pgd is not None and \
               model.adversarial.delta.grad is not None:
                delta_param = model.adversarial.delta
                # Print delta and grad stats before update
                if self.accelerator.is_main_process:
                    print(f"[DEBUG] PGD step {step+1}: delta L1 norm: {delta_param.data.abs().sum().item():.6f}, L-inf norm: {delta_param.data.abs().max().item():.6f}")
                    print(f"[DEBUG] PGD step {step+1}: grad L1 norm: {delta_param.grad.abs().sum().item():.6f}, L-inf norm: {delta_param.grad.abs().max().item():.6f}")
                with torch.no_grad():
                    current_delta_val = delta_param.data
                    delta_grad_val = delta_param.grad.data
                    # PGD descent step for targeted attack
                    updated_delta = current_delta_val - pgd_alpha * delta_grad_val.sign()
                    # Clamp delta to the allowed L-inf ball
                    updated_delta = torch.clamp(updated_delta, -pgd_epsilon, pgd_epsilon)
                    delta_param.data.copy_(updated_delta)
                # Zero out delta's gradient after the manual update
                if delta_param.grad is not None:
                    delta_param.grad.zero_()
                if self.accelerator.is_main_process:
                    print(f"  PGD: Step {step+1}/{pgd_steps} - delta L1 norm: {delta_param.data.abs().sum().item():.6f}, L-inf norm: {delta_param.data.abs().max().item():.6f}")

        # After PGD steps, do a final forward for the loss to return (using the final delta)。
        if clean_images_for_pgd is not None:
            if "image_id" in inputs:
                image_ids = inputs["image_id"]
                patch_positions = []
                for id in image_ids:
                    idx = int(id) % 2
                    if idx == 0:
                        patch_positions.append([730, 50])
                    elif idx == 1:
                        patch_positions.append([1085, 50])
                patch_boxes = torch.tensor([[x, y, 323, 295] for x, y in patch_positions], device=clean_images_for_pgd.device)
            perturbed_images_for_loss = model.adversarial(clean_images_for_pgd, patch_boxes, epoch=current_epoch)
            inputs["images"] = perturbed_images_for_loss
        model.train()
        batch_inputs = self._prepare_inputs(inputs)
        if "image_id" in batch_inputs:
            del batch_inputs["image_id"]
        loss = self.compute_loss(model, batch_inputs)

        # Inspect delta.grad under ZeRO-3 (as in original code)
        if hasattr(model, 'adversarial') and hasattr(model.adversarial, 'delta'):
            delta_to_inspect = model.adversarial.delta
            try:
                with GatheredParameters([delta_to_inspect], modifier_rank=0):
                    if self.accelerator.is_main_process:
                        if delta_to_inspect.grad is None:
                            print("  ▶ delta.grad is None (gathered, after PGD update & zeroing)")
                        else:
                            grad_sum = delta_to_inspect.grad.abs().sum().item()
                            print(f"  ▶ delta.grad sum (gathered, after PGD update & zeroing): {grad_sum}")
            except AssertionError as e:
                if self.accelerator.is_main_process:
                    print(f"⚠️  AssertionError during GatheredParameters for delta.grad (after PGD): {e}")
                    if delta_to_inspect.grad is not None:
                        print(f"⚠️  Direct delta.grad sum on rank 0 (fallback, after PGD): {delta_to_inspect.grad.abs().sum().item()}")
                    else:
                        print(f"⚠️  Direct delta.grad on rank 0 (fallback, after PGD) is None.")
            except Exception as e_other:
                if self.accelerator.is_main_process:
                    print(f"⚠️  Other error during GatheredParameters for delta.grad (after PGD): {e_other}")
        else:
            if self.accelerator.is_main_process:
                print("  ▶ Info: model.adversarial.delta not available for grad inspection.")

        return loss.detach()
            
    # def compute_loss(self, model, inputs, return_outputs=False):
    #     # 1. Pop labels so the model doesn't compute standard loss
    #     labels = inputs.pop("labels")
        
    #     input_ids = inputs.get("input_ids")
    #     attention_mask = inputs.get("attention_mask")
    #     position_ids = inputs.get("position_ids")
    #     images = inputs.get("images")

    #     # --- FIX: Helper to recursively find the base model with the method ---
    #     def get_model_with_method(m):
    #         if hasattr(m, "prepare_inputs_labels_for_multimodal"):
    #             return m
    #         # Unwrap DDP
    #         if hasattr(m, "module"):
    #             return get_model_with_method(m.module)
    #         # Unwrap PEFT if needed (though usually PEFT forwards attributes)
    #         if hasattr(m, "base_model"):
    #             return get_model_with_method(m.base_model)
    #         return m

    #     # Find the model that actually has the method
    #     model_with_method = get_model_with_method(model)
    #     # ---------------------------------------------------------------------

    #     # 3. CRITICAL: Expand labels and convert input_ids to embeddings
    #     if images is not None and hasattr(model_with_method, "prepare_inputs_labels_for_multimodal"):
    #         # Call the method on the unwrapped model
    #         _position_ids, _attention_mask, _past_key_values, _inputs_embeds, _labels = \
    #             model_with_method.prepare_inputs_labels_for_multimodal(
    #                 input_ids=input_ids,
    #                 position_ids=position_ids,
    #                 attention_mask=attention_mask,
    #                 past_key_values=None,
    #                 labels=labels,
    #                 images=images
    #             )
            
    #         labels = _labels
            
    #         # We must use inputs_embeds for the forward pass now
    #         forward_kwargs = {
    #             "inputs_embeds": _inputs_embeds,
    #             "attention_mask": _attention_mask,
    #             "position_ids": _position_ids,
    #             "images": images,
    #             "use_cache": False
    #         }
    #     else:
    #         # Fallback (should rarely happen if setup is correct)
    #         forward_kwargs = inputs

    #     # 4. Forward pass (use the original wrapped 'model' for DDP sync)
    #     outputs = model(**forward_kwargs)
        
    #     if self.args.past_index >= 0:
    #         self._past = outputs[self.args.past_index]

    #     if labels is not None:
    #         logits = outputs.logits
            
    #         shift_logits = logits[:, :-1, :].contiguous()
    #         shift_labels = labels[:, 1:].contiguous().to(shift_logits.device)

    #         # --- Calculate Weighted Loss ---
    #         loss_fct = torch.nn.CrossEntropyLoss(ignore_index=IGNORE_INDEX, reduction='none')
    #         vocab_size = shift_logits.size(-1)
            
    #         raw_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
    #         raw_loss = raw_loss.view(shift_labels.shape)

    #         # Apply Weighting Logic
    #         ELEMENT_WEIGHT = getattr(self.args, 'element_weight', 10.0) 
    #         LOOKAHEAD_STEPS = 6

    #         loss_weights = torch.ones_like(raw_loss)

    #         if not hasattr(self, "_element_token_ids") or self._element_token_ids is None:
    #             self._element_token_ids = set()
    #             variations = ["ELEMENT", " ELEMENT", "\nELEMENT", "ELEMENT:"]
    #             for v in variations:
    #                 ids = self.tokenizer(v, add_special_tokens=False).input_ids
    #                 if ids:
    #                     self._element_token_ids.update(ids)

    #         batch_size, seq_len = shift_labels.shape
    #         cpu_labels = shift_labels.detach().cpu().tolist()

    #         for b in range(batch_size):
    #             for t_idx, token_id in enumerate(cpu_labels[b]):
    #                 if token_id == IGNORE_INDEX: continue
    #                 if token_id in self._element_token_ids:
    #                     start_idx = t_idx
    #                     end_idx = min(seq_len, t_idx + LOOKAHEAD_STEPS)
    #                     loss_weights[b, start_idx:end_idx] = ELEMENT_WEIGHT

    #         weighted_loss = raw_loss * loss_weights
    #         valid_token_count = (shift_labels != IGNORE_INDEX).sum()
            
    #         if valid_token_count > 0:
    #             loss = weighted_loss.sum() / valid_token_count
    #         else:
    #             loss = weighted_loss.sum()

    #     else:
    #         if isinstance(outputs, dict) and "loss" not in outputs:
    #             raise ValueError("The model did not return a loss from the inputs.")
    #         loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

    #     return (loss, outputs) if return_outputs else loss

def maybe_zero_3(param, ignore_status=False, name=None):
    from deepspeed import zero
    from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                logging.warning(f"{name}: param.ds_status != ZeroParamStatus.NOT_AVAILABLE: {param.ds_status}")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


# Borrowed from peft.utils.get_peft_model_state_dict
def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias.items():
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mm_adapter_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model):
    # Only patch the first linear layer found
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear):
            return [name]
    return []


def safe_save_model_for_hf_trainer(trainer: transformers.Trainer,
                                   output_dir: str):
    """Collects the state dict and dump to disk."""

    if getattr(trainer.args, "tune_mm_mlp_adapter", False):
        # Only save Adapter
        keys_to_match = ['mm_projector']
        if getattr(trainer.args, "use_im_start_end", False):
            keys_to_match.extend(['embed_tokens', 'embed_in'])

        weight_to_save = get_mm_adapter_state_maybe_zero_3(trainer.model.named_parameters(), keys_to_match)
        trainer.model.config.save_pretrained(output_dir)

        current_folder = output_dir.split('/')[-1]
        parent_folder = os.path.dirname(output_dir)
        if trainer.args.local_rank == 0 or trainer.args.local_rank == -1:
            if current_folder.startswith('checkpoint-'):
                mm_projector_folder = os.path.join(parent_folder, "mm_projector")
                os.makedirs(mm_projector_folder, exist_ok=True)
                torch.save(weight_to_save, os.path.join(mm_projector_folder, f'{current_folder}.bin'))
            else:
                torch.save(weight_to_save, os.path.join(output_dir, f'mm_projector.bin'))
        return

    if trainer.deepspeed:
        torch.cuda.synchronize()
        trainer.save_model(output_dir)
        return

    state_dict = trainer.model.state_dict()
    if trainer.args.should_save:
        cpu_state_dict = {
            key: value.cpu()
            for key, value in state_dict.items()
        }
        del state_dict
        trainer._save(output_dir, state_dict=cpu_state_dict)  # noqa


def smart_tokenizer_and_embedding_resize(
    special_tokens_dict: Dict,
    tokenizer: transformers.PreTrainedTokenizer,
    model: transformers.PreTrainedModel,
):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(
            dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg


def _tokenize_fn(strings: Sequence[str],
                 tokenizer: transformers.PreTrainedTokenizer) -> Dict:
    """Tokenize a list of strings."""
    tokenized_list = [
        tokenizer(
            text,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ) for text in strings
    ]
    input_ids = labels = [
        tokenized.input_ids[0] for tokenized in tokenized_list
    ]
    input_ids_lens = labels_lens = [
        tokenized.input_ids.ne(tokenizer.pad_token_id).sum().item()
        for tokenized in tokenized_list
    ]
    return dict(
        input_ids=input_ids,
        labels=labels,
        input_ids_lens=input_ids_lens,
        labels_lens=labels_lens,
    )


def _mask_targets(target, tokenized_lens, speakers):
    # cur_idx = 0
    cur_idx = tokenized_lens[0]
    tokenized_lens = tokenized_lens[1:]
    target[:cur_idx] = IGNORE_INDEX
    for tokenized_len, speaker in zip(tokenized_lens, speakers):
        if speaker == "human":
            target[cur_idx+2:cur_idx + tokenized_len] = IGNORE_INDEX
        cur_idx += tokenized_len


def _add_speaker_and_signal(header, source, get_conversation=True):
    """Add speaker and start/end signal on each round."""
    BEGIN_SIGNAL = "### "
    END_SIGNAL = "\n"
    conversation = header
    for sentence in source:
        from_str = sentence["from"]
        if from_str.lower() == "human":
            from_str = conversation_lib.default_conversation.roles[0]
        elif from_str.lower() == "gpt":
            from_str = conversation_lib.default_conversation.roles[1]
        else:
            from_str = 'unknown'
        sentence["value"] = (BEGIN_SIGNAL + from_str + ": " +
                             sentence["value"] + END_SIGNAL)
        if get_conversation:
            conversation += sentence["value"]
    conversation += BEGIN_SIGNAL
    return conversation


def preprocess_multimodal(
    sources: Sequence[str],
    data_args: DataArguments
) -> Dict:
    is_multimodal = data_args.is_multimodal
    if not is_multimodal:
        return sources

    for source in sources:
        for sentence in source:
            if DEFAULT_IMAGE_TOKEN in sentence['value']:
                sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '').strip()
                sentence['value'] = DEFAULT_IMAGE_TOKEN + '\n' + sentence['value']
                sentence['value'] = sentence['value'].strip()
                if "mmtag" in conversation_lib.default_conversation.version:
                    sentence['value'] = sentence['value'].replace(DEFAULT_IMAGE_TOKEN, '<Image>' + DEFAULT_IMAGE_TOKEN + '</Image>')
            replace_token = DEFAULT_IMAGE_TOKEN
            if data_args.mm_use_im_start_end:
                replace_token = DEFAULT_IM_START_TOKEN + replace_token + DEFAULT_IM_END_TOKEN
            sentence["value"] = sentence["value"].replace(DEFAULT_IMAGE_TOKEN, replace_token)

    return sources

def preprocess_chatml_direct(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    has_image: bool = False
) -> Dict:
    """
    Preprocess for conv_chatml_direct (ChatML direct template).
    """
    conv = conversation_lib.conv_chatml_direct.copy()
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "user": conv.roles[0], "assistant": conv.roles[1]}

    # Apply prompt templates
    conversations = []
    for i, source in enumerate(sources):
        # Ensure the first message is from user
        if roles.get(source[0]["from"].lower(), None) != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"].lower(), sentence["from"])
            assert role == conv.roles[j % 2], f"{i}"
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())

    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids
    # print(f"[TRAIN] input_ids: {  input_ids}\n")

    targets = input_ids.clone()
    assert conv.sep_style == conversation_lib.SeparatorStyle.MPT

    # Mask targets: mask user turns (instructions)
    sep = conv.sep + conv.roles[1]
    for conversation, target in zip(conversations, targets):
        total_len = int(target.ne(tokenizer.pad_token_id).sum())

        rounds = conversation.split(conv.sep)
        re_rounds = [conv.sep.join(rounds[:3])]
        for conv_idx in range(3, len(rounds), 2):
            re_rounds.append(conv.sep.join(rounds[conv_idx:conv_idx+2]))    # user + assistant
        cur_len = 1
        target[:cur_len] = IGNORE_INDEX
        for i, rou in enumerate(re_rounds):
            if rou == "":
                break

            parts = rou.split(sep)
            if len(parts) != 2:
                break
            parts[0] += sep

            if has_image:
                round_len = len(tokenizer_image_token(rou, tokenizer))
                instruction_len = len(tokenizer_image_token(parts[0], tokenizer)) - 1
            else:
                round_len = len(tokenizer(rou).input_ids)
                instruction_len = len(tokenizer(parts[0]).input_ids) - 1

            if i != 0 and getattr(tokenizer, 'legacy', False) and IS_TOKENIZER_GREATER_THAN_0_14:
                round_len += 1
                instruction_len += 1

            target[cur_len : cur_len + instruction_len] = IGNORE_INDEX

            cur_len += round_len
        target[cur_len:] = IGNORE_INDEX

        if cur_len < tokenizer.model_max_length:
            if cur_len != total_len:
                target[:] = IGNORE_INDEX
                print(
                    f"WARNING: tokenization mismatch: {cur_len} vs. {total_len}."
                    f" (ignored)"
                )

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess_chatml_direct_multi(
    sources,
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False
) -> dict:
    """
    Preprocess for conv_chatml_direct (ChatML direct template) for multi-turn:
    - Input: the whole conversation
    - Label: only the value of the last assistant turn is unmasked, all others (including the template) are masked
    """
    conv = conversation_lib.conv_chatml_direct.copy()
    # Load system prompt from conv_sample_3.json
    with open(data_args.inference_data_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    system_prompt = conv_data["system"]
    conv.system = f"<|im_start|>system\n{system_prompt}"
    roles = {"human": conv.roles[0], "gpt": conv.roles[1], "user": conv.roles[0], "assistant": conv.roles[1]}

    conversations = []
    for i, source in enumerate(sources):
        # Ensure the first message is from user
        if roles.get(source[0]["from"].lower(), None) != conv.roles[0]:
            source = source[1:]

        conv.messages = []
        for j, sentence in enumerate(source):
            role = roles.get(sentence["from"].lower(), sentence["from"])
            conv.append_message(role, sentence["value"])
        conversations.append(conv.get_prompt())
    # Tokenize conversations
    if has_image:
        input_ids = torch.stack([tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations], dim=0)
    else:
        input_ids = tokenizer(
            conversations,
            return_tensors="pt",
            padding="longest",
            max_length=tokenizer.model_max_length,
            truncation=True,
        ).input_ids

    targets = input_ids.clone()

    # Mask everything except the value of the last assistant turn
    for idx, (source, input_id, target) in enumerate(zip(sources, input_ids, targets)):
        # Find the last assistant turn
        last_assistant_idx = None
        for i in range(len(source) - 1, -1, -1):
            if source[i]["from"].lower() in ["assistant", "gpt"]:
                last_assistant_idx = i
                break
        assert last_assistant_idx is not None, "No assistant turn found in conversation"

        # Build the prompt up to (but not including) the last assistant turn
        conv.messages = []
        for s in source[:last_assistant_idx]:
            role = roles.get(s["from"].lower(), s["from"])
            conv.append_message(role, s["value"])
        prompt_up_to_last = conv.get_prompt()

        # Add the template for the last assistant, but with empty value
        role = roles.get(source[last_assistant_idx]["from"].lower(), source[last_assistant_idx]["from"])
        conv.append_message(role, "")
        prompt_with_template = conv.get_prompt()

        # Tokenize to get indices
        if has_image:
            start = len(tokenizer_image_token(prompt_with_template, tokenizer))
            # end = start + len(tokenizer_image_token(source[last_assistant_idx]["value"], tokenizer))
        else:
            start = len(tokenizer(prompt_with_template).input_ids)
            # end = start + len(tokenizer(source[last_assistant_idx]["value"]).input_ids)

        # Mask everything except [start:end]
        target[:start] = IGNORE_INDEX

    return dict(
        input_ids=input_ids,
        labels=targets,
    )

def preprocess(
    sources: Sequence[str],
    tokenizer: transformers.PreTrainedTokenizer,
    data_args: DataArguments,
    has_image: bool = False
) -> Dict:
    """
    Given a list of sources, each is a conversation list. This transform:
    1. Add signal '### ' at the beginning each sentence, with end signal '\n';
    2. Concatenate conversations together;
    3. Tokenize the concatenated conversation;
    4. Make a deepcopy as the target. Mask human words with IGNORE_INDEX.
    """
    if conversation_lib.default_conversation.version == "chatml_direct":
        return preprocess_chatml_direct_multi(sources, tokenizer, data_args=data_args,has_image=has_image)
    # add end signal and concatenate together
    conversations = []
    for source in sources:
        header = f"{conversation_lib.default_conversation.system}\n\n"
        conversation = _add_speaker_and_signal(header, source)
        conversations.append(conversation)
    # tokenize conversations
    def get_tokenize_len(prompts):
        return [len(tokenizer_image_token(prompt, tokenizer)) for prompt in prompts]

    if has_image:
        input_ids = [tokenizer_image_token(prompt, tokenizer, return_tensors='pt') for prompt in conversations]
    else:
        conversations_tokenized = _tokenize_fn(conversations, tokenizer)
        input_ids = conversations_tokenized["input_ids"]

    targets = copy.deepcopy(input_ids)
    for target, source in zip(targets, sources):
        if has_image:
            tokenized_lens = get_tokenize_len([header] + [s["value"] for s in source])
        else:
            tokenized_lens = _tokenize_fn([header] + [s["value"] for s in source], tokenizer)["input_ids_lens"]
        speakers = [sentence["from"] for sentence in source]
        _mask_targets(target, tokenized_lens, speakers)

    return dict(input_ids=input_ids, labels=targets)


class LazySupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(self, train_data_path: str,
                 tokenizer: transformers.PreTrainedTokenizer,
                 data_args: DataArguments):
        super(LazySupervisedDataset, self).__init__()
        list_data_dict = json.load(open(train_data_path, "r"))

        rank0_print("Formatting inputs...Skip in lazy mode")
        self.tokenizer = tokenizer
        self.list_data_dict = list_data_dict
        self.data_args = data_args

    def __len__(self):
        return len(self.list_data_dict)

    @property
    def lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            img_tokens = 128 if 'image' in sample else 0
            length_list.append(sum(len(conv['value'].split()) for conv in sample['conversations']) + img_tokens)
        return length_list

    @property
    def modality_lengths(self):
        length_list = []
        for sample in self.list_data_dict:
            cur_len = sum(len(conv['value'].split()) for conv in sample['conversations'])
            cur_len = cur_len if 'image' in sample else -cur_len
            length_list.append(cur_len)
        return length_list

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        sources = self.list_data_dict[i]
        if isinstance(i, int):
            sources = [sources]
        assert len(sources) == 1, "Don't know why it is wrapped to a list"  # FIXME
        image = None
        if 'image' in sources[0]:
            image_file = self.list_data_dict[i]['image']
            # image_folder = self.data_args.image_folder
            image = Image.open(image_file).convert('RGB')
            # Convert PIL image to tensor (C, H, W), float32, [0, 1]
            image = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
            sources = preprocess_multimodal(
                copy.deepcopy([e["conversations"] for e in sources]),
                self.data_args)
        else:
            sources = copy.deepcopy([e["conversations"] for e in sources])
        data_dict = preprocess(
            sources,
            self.tokenizer,
            self.data_args,
            has_image=('image' in self.list_data_dict[i])
        )
        if isinstance(i, int):
            data_dict = dict(input_ids=data_dict["input_ids"][0],
                             labels=data_dict["labels"][0])

        # image exist in the data
        if 'image' in self.list_data_dict[i]:
            data_dict['image'] = image
        elif self.data_args.is_multimodal:
            # image does not exist in the data, but the model is multimodal
            crop_size = self.data_args.image_processor.crop_size
            data_dict['image'] = torch.zeros(3, crop_size['height'], crop_size['width'])
        # Add image_id if present
        if 'id' in self.list_data_dict[i]:
            data_dict['image_id'] = int(self.list_data_dict[i]['id'])
        return data_dict


@dataclass
class DataCollatorForSupervisedDataset(object):
    """Collate examples for supervised fine-tuning."""

    tokenizer: transformers.PreTrainedTokenizer

    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        input_ids, labels = tuple([instance[key] for instance in instances]
                                  for key in ("input_ids", "labels"))
        input_ids = torch.nn.utils.rnn.pad_sequence(
            input_ids,
            batch_first=True,
            padding_value=self.tokenizer.pad_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels,
                                                 batch_first=True,
                                                 padding_value=IGNORE_INDEX)
        input_ids = input_ids[:, :self.tokenizer.model_max_length]
        labels = labels[:, :self.tokenizer.model_max_length]
        batch = dict(
            input_ids=input_ids,
            labels=labels,
            attention_mask=input_ids.ne(self.tokenizer.pad_token_id),
        )

        if 'image' in instances[0]:
            images = [instance['image'] for instance in instances]
            if all(x is not None and x.shape == images[0].shape for x in images):
                batch['images'] = torch.stack(images)
            else:
                batch['images'] = images
        # Batch image_id if present
        if 'image_id' in instances[0]:
            batch['image_id'] = torch.tensor([instance['image_id'] for instance in instances], dtype=torch.long)
        return batch


def make_supervised_data_module(tokenizer: transformers.PreTrainedTokenizer,
                                data_args) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    train_dataset = LazySupervisedDataset(tokenizer=tokenizer,
                                train_data_path=data_args.train_data_path,
                                data_args=data_args)
    data_collator = DataCollatorForSupervisedDataset(tokenizer=tokenizer)
    return dict(train_dataset=train_dataset,
                eval_dataset=None,
                data_collator=data_collator)


def print_inference_result(model, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, image_tensor, device, max_new_tokens=450):
    model.eval()

    model_name = getattr(model, 'model_name', 'llava-v1.6-34b')
    if "v1.6-34b" in model_name.lower():
        print(f"[DEBUG] v1.6-34b")
        conv_mode = "chatml_direct"
    else:
        conv_mode = "llava_v0"
    conv = conv_templates[conv_mode].copy()
    # Add image token to prompt
    if model.config.mm_use_im_start_end:
        user_request = DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_TOKEN + DEFAULT_IM_END_TOKEN + '\n' + user_request
    else:
        user_request = DEFAULT_IMAGE_TOKEN + '\n' + user_request

    conv.system = f"<|im_start|>system\n{system_prompt}"
    conv.append_message(conv.roles[0], user_request)
    conv.append_message(conv.roles[1], assistant_answer)
    conv.append_message(conv.roles[0], grounding_prompt)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()
    input_ids = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors='pt').unsqueeze(0).to(device)
    # print(f"[DEBUG] input_ids: {input_ids}\n")
    # print(f"[DEBUG] input_ids: {input_ids}")
    # assert input_ids is not None, "input_ids is None!"
    # Generate output
    with torch.no_grad():
        print("Dtype before vision tower:", image_tensor.dtype)
        output_ids = model.generate(
            inputs=input_ids,
            images=image_tensor,
            image_sizes=[(image_tensor.shape[-1], image_tensor.shape[-2])],
            do_sample=False,
            max_new_tokens=max_new_tokens,
            use_cache=False
        )
    decoded_output = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0].strip()
    print(f"[INFERENCE] OUTPUT:\n{decoded_output}\n")
    return decoded_output

class AdversarialInferenceCallback(TrainerCallback):
    def __init__(self, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, device, base_img_tensor, processor, base_img_pil):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_request = user_request
        self.grounding_prompt = grounding_prompt
        self.assistant_answer = assistant_answer
        self.device = device
        self.base_img = base_img_tensor.to(device)
        # CLIP's normalization values
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]

        # Target size from processor
        self.target_size = processor.size['shortest_edge']
        self.image_processor = processor
        self.base_img_pil = base_img_pil
        # # Compose pipeline for tensor input (skip ToTensor)
        # self.transform = Compose([
        #     Resize(self.target_size, interpolation=InterpolationMode.BICUBIC),  # BICUBIC for CLIP
        #     CenterCrop(self.target_size),
        #     Normalize(self.mean, self.std),
        # ])
        self.transform = DifferentiableCLIPPreprocess(
            size=self.target_size,
            crop_size=self.target_size,
            mean=self.mean,
            std=self.std,
            do_normalize=True,
        )

    def resize(self, image_tensor):
        batched = image_tensor.dim() == 4
        x = image_tensor if batched else image_tensor.unsqueeze(0)

        if isinstance(self.target_size, int):
            size = (self.target_size, self.target_size)
        else:
            size = self.target_size

        y = F.interpolate(x, size=size, mode='bicubic', align_corners=False, antialias=True)

        return y if batched else y.squeeze(0)

    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if state.epoch is None or state.epoch <= 800:
            return
        if hasattr(args, 'local_rank') and args.local_rank not in [0, -1]:
            return
        print("\n[CALLBACK] Running adversarial inference at epoch end...")
    
        model.eval()
        with torch.no_grad():
            if hasattr(model, 'adversarial') and hasattr(model.adversarial, 'delta'):
                perturbed = self.base_img.clone()
                perturbed[:, 50:345, 1085:1408] += model.adversarial.delta.detach().to(self.base_img.device, dtype=self.base_img.dtype).squeeze(0)
                adv_img_tensor = perturbed.clone().cpu().squeeze(0).clamp(0, 1)
                adv_img_pil = to_pil_image(adv_img_tensor)
                adv_img_tensor = process_images([adv_img_pil], self.image_processor, model.config)
                if adv_img_tensor.ndim == 5 and adv_img_tensor.shape[1] == 5:
                    adv_img_tensor = adv_img_tensor[:, 0, :, :, :]
                adv_img_tensor = adv_img_tensor.squeeze(0)  # [3, 336, 336]
                adv_img_tensor = adv_img_tensor.to(self.device, dtype=torch.bfloat16)
                adv_img_tensor = adv_img_tensor.unsqueeze(0)

            else:
                print("[CALLBACK] No adversarial delta found on model.")
                return
            print_inference_result(model, self.tokenizer, self.system_prompt, self.user_request, self.grounding_prompt, self.assistant_answer, adv_img_tensor, self.device)

    # def on_epoch_end_2(self, args, state, control, model=None, **kwargs):
    #     should_stop = torch.tensor(0, device=self.device)

    #     if not hasattr(args, 'local_rank') or args.local_rank in [0, -1]:
    #         print("\n[CALLBACK] Running adversarial inference at epoch end...")
    #         model.eval()
    #         with torch.no_grad():
    #             if hasattr(model, 'adversarial') and hasattr(model.adversarial, 'delta'):

    #                 adv_img_tensor = (self.base_img + model.adversarial.delta.detach().to(self.base_img.device, dtype=self.base_img.dtype) * self.mask_tensor.to(self.base_img.device, dtype=self.base_img.dtype))
    #                 adv_img_tensor = adv_img_tensor.clone().cpu().squeeze(0).clamp(0, 1)
    #                 adv_img_pil = to_pil_image(adv_img_tensor)
    #                 adv_img_tensor = process_images([adv_img_pil], self.image_processor, model.config)
    #                 if adv_img_tensor.ndim == 5 and adv_img_tensor.shape[1] == 5:
    #                     adv_img_tensor = adv_img_tensor[:, 0, :, :, :]
    #                 adv_img_tensor = adv_img_tensor.squeeze(0)  # [3, 336, 336]
    #                 adv_img_tensor = adv_img_tensor.to(self.device, dtype=torch.bfloat16)
    #                 adv_img_tensor = adv_img_tensor.unsqueeze(0)

    #                 decoded_output = print_inference_result(model, self.tokenizer, self.system_prompt, self.user_request, self.grounding_prompt, self.assistant_answer, adv_img_tensor, self.device)
    #                 if "ELEMENT: D" in decoded_output:
    #                     print("Expected result achieved, stopping training.")
    #                     should_stop = torch.tensor(1, device=self.device)

    #             else:
    #                 print("[CALLBACK] No adversarial delta found on model.")

    #     # If distributed, broadcast should_stop from rank 0 to all ranks
    #     if dist.is_initialized():
    #         dist.broadcast(should_stop, src=0)

    #     # All ranks set the flag if should_stop is 1
    #     if should_stop.item() == 1:
    #         control.should_training_stop = True

def inference_before_training(model, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, device, processor, base_img_pil):
    vision_dtype = next(model.get_model().mm_projector.parameters()).dtype
    print("[DEBUG] Inference before add adversarial module DTYPE: ", vision_dtype)
    adv_img_tensor = process_images([base_img_pil], processor, model.config)
    if adv_img_tensor.ndim == 5 and adv_img_tensor.shape[1] == 5:
        adv_img_tensor = adv_img_tensor[:, 0, :, :, :]
    adv_img_tensor = adv_img_tensor.squeeze(0)  # [3, 336, 336]
    adv_img_tensor = adv_img_tensor.to(device, dtype=torch.bfloat16)
    adv_img_tensor = adv_img_tensor.unsqueeze(0)
    print_inference_result(model, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, adv_img_tensor, device)


def train(attn_implementation=None):
    global local_rank
    print("start preparing training...")
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    local_rank = training_args.local_rank
    compute_dtype = (torch.float16 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))

    image_path = data_args.image_path_1
    base_img_pil = Image.open(image_path).convert("RGB")
    with open(data_args.inference_data_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    system_prompt = conv_data["system"]
    user_request = conv_data["user"][0]
    grounding_prompt = conv_data["user"][1]
    assistant_answer = conv_data["assistant"]

    # Add adversarial image saving arguments
    if not hasattr(training_args, 'save_adversarial_images'):
        training_args.save_adversarial_images = False
    if not hasattr(training_args, 'adversarial_images_dir'):
        training_args.adversarial_images_dir = './adversarial_images'

    bnb_model_from_pretrained_args = {}
    if training_args.bits in [4, 8]:
        from transformers import BitsAndBytesConfig
        bnb_model_from_pretrained_args.update(dict(
            device_map={"": training_args.device},
            load_in_4bit=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=compute_dtype,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type='nf4' # {'fp4', 'nf4'}
            )
        ))
    print("start initializing vision tower...")
    if model_args.vision_tower is not None:
        if 'mpt' in model_args.model_name_or_path:
            config = transformers.AutoConfig.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)
            config.attn_config['attn_impl'] = training_args.mpt_attn_impl
            model = LlavaMptForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                config=config,
                cache_dir=training_args.cache_dir,
                **bnb_model_from_pretrained_args
            )
        else:
            print("start initializing llama for causal lm...")
            model = LlavaLlamaForCausalLM.from_pretrained(
                model_args.model_name_or_path,
                cache_dir=training_args.cache_dir,
                attn_implementation=attn_implementation,
                torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
                **bnb_model_from_pretrained_args
            )
            print("done initializing llama for causal lm...")
    else:
        model = transformers.LlamaForCausalLM.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            attn_implementation=attn_implementation,
            torch_dtype=(torch.bfloat16 if training_args.bf16 else None),
            **bnb_model_from_pretrained_args
        )
    model.config.use_cache = False
    # Ensure the token+vision embeddings require grad, so loss stays connected
    # if hasattr(model, "enable_input_require_grads"):
    #     model.enable_input_require_grads()

    if training_args.bits in [4, 8]:
        from peft import prepare_model_for_kbit_training
        model.config.torch_dtype=(torch.float32 if training_args.fp16 else (torch.bfloat16 if training_args.bf16 else torch.float32))
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=training_args.gradient_checkpointing)

    if training_args.gradient_checkpointing:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
        else:
            def make_inputs_require_grad(module, input, output):
                output.requires_grad_(True)
            model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    if training_args.lora_enable:
        from peft import LoraConfig, get_peft_model
        lora_config = LoraConfig(
            r=training_args.lora_r,
            lora_alpha=training_args.lora_alpha,
            target_modules=find_all_linear_names(model),
            lora_dropout=training_args.lora_dropout,
            bias=training_args.lora_bias,
            task_type="CAUSAL_LM",
        )
        rank0_print("Adding LoRA adapters...")
        model = get_peft_model(model, lora_config)
        for name, param in model.named_parameters():
            if "lora_" in name:
                param.requires_grad = False
        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            print("Trainable parameters after freezing LoRA:", trainable_params)

    if 'mpt' in model_args.model_name_or_path:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right"
        )
    else:
        tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_args.model_name_or_path,
            cache_dir=training_args.cache_dir,
            model_max_length=training_args.model_max_length,
            padding_side="right",
            use_fast=False,
        )

    if model_args.version == "v0":
        if tokenizer.pad_token is None:
            smart_tokenizer_and_embedding_resize(
                special_tokens_dict=dict(pad_token="[PAD]"),
                tokenizer=tokenizer,
                model=model,
            )
    elif model_args.version == "v0.5":
        tokenizer.pad_token = tokenizer.unk_token
    else:
        tokenizer.pad_token = tokenizer.unk_token
        if model_args.version in conversation_lib.conv_templates:
            conversation_lib.default_conversation = conversation_lib.conv_templates[model_args.version]
        else:
            conversation_lib.default_conversation = conversation_lib.conv_templates["vicuna_v1"]

    if model_args.vision_tower is not None:
        model.get_model().initialize_vision_modules(
            model_args=model_args,
            fsdp=training_args.fsdp
        )
        
        vision_tower = model.get_vision_tower()
        vision_tower.to(dtype=torch.bfloat16 if training_args.bf16 else torch.float16, device=training_args.device)

        data_args.image_processor = vision_tower.image_processor
        data_args.is_multimodal = True

        model.config.image_aspect_ratio = data_args.image_aspect_ratio
        model.config.tokenizer_padding_side = tokenizer.padding_side
        model.config.tokenizer_model_max_length = tokenizer.model_max_length

        model.config.tune_mm_mlp_adapter = training_args.tune_mm_mlp_adapter = model_args.tune_mm_mlp_adapter
        if model_args.tune_mm_mlp_adapter:
            model.requires_grad_(False)
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = True

        model.config.freeze_mm_mlp_adapter = training_args.freeze_mm_mlp_adapter
        if training_args.freeze_mm_mlp_adapter:
            for p in model.get_model().mm_projector.parameters():
                p.requires_grad = False

        if training_args.bits in [4, 8]:
            model.get_model().mm_projector.to(dtype=compute_dtype, device=training_args.device)

        model.config.mm_use_im_start_end = data_args.mm_use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_projector_lr = training_args.mm_projector_lr
        training_args.use_im_start_end = model_args.mm_use_im_start_end
        model.config.mm_use_im_patch_token = model_args.mm_use_im_patch_token
        # model.initialize_vision_tokenizer(model_args, tokenizer=tokenizer)

        # Add freeze_backbone to model.config
        model.config.freeze_backbone = model_args.freeze_backbone
        # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
        # print("[DEBUG] Inference before add adversarial module")
        # inference_before_training(model, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, training_args.device, data_args.image_processor, base_img_pil)
        # Diagnostic print from all ranks
        print(f"[Rank {training_args.local_rank}] Check before freeze_backbone block: model_args.freeze_backbone = {model_args.freeze_backbone}")

        if model_args.freeze_backbone:
            # Diagnostic print from all ranks
            print(f"[Rank {training_args.local_rank}] Entering freeze_backbone block. Setting requires_grad for parameters.")
            for name, p in model.named_parameters():
                if name.startswith("adversarial."):
                    p.requires_grad_(True)
                else:
                    p.requires_grad_(False)

        # Get a sample image to determine the original shape
        sample_image_path = data_args.image_path_1
        if sample_image_path is not None:
            sample_img = Image.open(sample_image_path).convert("RGB")
            orig_img_shape = (3, sample_img.height, sample_img.width)
        else:
            # Fallback to cropped shape if no image found
            crop = data_args.image_processor.crop_size
            orig_img_shape = (3, crop['height'], crop['width'])

        # Create and register the adversarial module with the original image shape
        adversarial_module = AdversarialPerturbation(orig_img_shape, device=training_args.device, processor=data_args.image_processor)
        model.add_module('adversarial', adversarial_module)
        # Ensure delta requires grad before Trainer initialization, especially for DeepSpeed
        model.adversarial.delta.requires_grad_(True)
        # Now check trainable parameters after setting up adversarial module
        trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
        if training_args.local_rank == 0:
            print(f"INFO (Rank 0): After setup, the following parameters are trainable: {trainable_params}")
            if not trainable_params:
                print("WARNING (Rank 0): NO parameters ended up being trainable after setup.")
            else:
                all_trainable_are_adversarial = all(name.startswith("adversarial.") for name in trainable_params)
                any_adversarial_is_trainable = any(name.startswith("adversarial.") for name in trainable_params)

                if not any_adversarial_is_trainable:
                    print(f"WARNING (Rank 0): NO adversarial parameters are trainable. Actual trainable: {trainable_params}")
                elif not all_trainable_are_adversarial:
                    non_adv_trainable = [name for name in trainable_params if not name.startswith("adversarial.")]
                    print(f"WARNING (Rank 0): Some NON-ADVERSARIAL parameters are also trainable: {non_adv_trainable}")

    if training_args.bits in [4, 8]:
        from peft.tuners.lora import LoraLayer
        for name, module in model.named_modules():
            if isinstance(module, LoraLayer):
                if training_args.bf16:
                    module = module.to(torch.bfloat16)
            if 'norm' in name:
                module = module.to(torch.float32)
            if 'lm_head' in name or 'embed_tokens' in name:
                if hasattr(module, 'weight'):
                    if training_args.bf16 and module.weight.dtype == torch.float32:
                        module = module.to(torch.bfloat16)
    # vision_tower.to(dtype=torch.bfloat16, device=training_args.device)
    # print("[DEBUG] Inference after do model type change")
    # inference_before_training(model, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, training_args.device, data_args.image_processor, base_img_pil)

    data_module = make_supervised_data_module(tokenizer=tokenizer,
                                              data_args=data_args)
    trainer = DebugTrainer(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module
    )

    image_path_1 = data_args.image_path_1
    base_img_pil_1 = Image.open(image_path_1).convert("RGB")
    base_img_tensor_1 = torch.from_numpy(np.array(base_img_pil_1)).permute(2, 0, 1).float() / 255.0
    base_img_tensor_1 = base_img_tensor_1.to(training_args.device)

    image_path_2 = data_args.image_path_2
    base_img_pil_2 = Image.open(image_path_2).convert("RGB")
    base_img_tensor_2 = torch.from_numpy(np.array(base_img_pil_2)).permute(2, 0, 1).float() / 255.0
    base_img_tensor_2 = base_img_tensor_2.to(training_args.device)

    with open(data_args.inference_data_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    system_prompt = conv_data["system"]
    user_request = conv_data["user"][0]
    grounding_prompt = conv_data["user"][1]
    assistant_answer = conv_data["assistant"]
    inference_callback = AdversarialInferenceCallback(tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, training_args.device, base_img_tensor_1, data_args.image_processor, base_img_pil_1)
    trainer.add_callback(inference_callback)


    if list(pathlib.Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print("start training...")
        trainer.train()
        print("training done")
    # trainer.save_state()

    model.config.use_cache = False

    if training_args.lora_enable:
        state_dict = get_peft_state_maybe_zero_3(
            model.named_parameters(), training_args.lora_bias
        )
        non_lora_state_dict = get_peft_state_non_lora_maybe_zero_3(
            model.named_parameters()
        )
        if training_args.local_rank == 0 or training_args.local_rank == -1:
            model.config.save_pretrained(training_args.output_dir)
            model.save_pretrained(training_args.output_dir, state_dict=state_dict)
            torch.save(non_lora_state_dict, os.path.join(training_args.output_dir, 'non_lora_trainables.bin'))
    else:
        safe_save_model_for_hf_trainer(trainer=trainer,
                                       output_dir=training_args.output_dir)

    if training_args.save_adversarial_images:
        perturbed_1 = base_img_tensor_1.clone()
        perturbed_1[:, 50:345, 730:1053] += model.adversarial.delta.detach().to(base_img_tensor_1.device, dtype=base_img_tensor_1.dtype).squeeze(0)
        img_to_save = perturbed_1.clone().cpu().squeeze(0).clamp(0, 1)
        adv_img_pil_1 = to_pil_image(img_to_save)
        img_path_1 = os.path.join(training_args.adversarial_images_dir, 'adversarial_image_with_patch_1.png')
        adv_img_pil_1.save(img_path_1)
        print(f"[INFO] Adversarial image 1 saved to {img_path_1}")

        perturbed_2 = base_img_tensor_2.clone()
        perturbed_2[:, 50:345, 1085:1408] += model.adversarial.delta.detach().to(base_img_tensor_1.device, dtype=base_img_tensor_1.dtype).squeeze(0)
        img_to_save = perturbed_2.clone().cpu().squeeze(0).clamp(0, 1)
        adv_img_pil_2 = to_pil_image(img_to_save)
        img_path_2 = os.path.join(training_args.adversarial_images_dir, 'adversarial_image_with_patch_2.png')
        adv_img_pil_2.save(img_path_2)
        print(f"[INFO] Adversarial image 2 saved to {img_path_2}")

        # --- Crop and save the adversarial patch/card ---
        # Only run file I/O on main process to avoid race conditions
        if not hasattr(training_args, 'local_rank') or training_args.local_rank == 0:
            patch_x, patch_y = 730, 50
            patch_w, patch_h = 323, 295
            adversarial_card = adv_img_pil_1.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
            pertubate_path_1 = os.path.join(training_args.adversarial_images_dir, 'adversarial_patch_1.png')
            adversarial_card.save(pertubate_path_1)
            print(f"[INFO] Adversarial patch 1 saved to {pertubate_path_1}")

            patch_x, patch_y = 1085, 50
            patch_w, patch_h = 323, 295
            adversarial_card = adv_img_pil_2.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
            pertubate_path_2 = os.path.join(training_args.adversarial_images_dir, 'adversarial_patch_2.png')
            adversarial_card.save(pertubate_path_2)
            print(f"[INFO] Adversarial patch 2 saved to {pertubate_path_2}")

            patch_1 = Image.open(pertubate_path_1).convert('RGB')
            patch_2 = Image.open(pertubate_path_2).convert('RGB')

            test_dirs = [
                ('test_samples_1_unseen', 40),
                ('test_samples_2_unseen', 40), 
                ('test_samples_3_unseen', 40),
                ('train_samples', 30)
            ]

            for dir_name, num_samples in test_dirs:
                test_samples_dir = os.path.join(training_args.adversarial_images_dir, dir_name)
                os.makedirs(test_samples_dir, exist_ok=True)
                
                applied_count = 0
                for i in range(num_samples):
                    sample_name = f'sample_{i}.png'
                    sample_path = os.path.join(test_samples_dir, sample_name)
                    if not os.path.exists(sample_path):
                        rank0_print(f"[WARNING] {sample_path} does not exist, skipping.")
                        continue
                    
                    try:
                        sample_img = Image.open(sample_path).convert('RGB')
                        idx = i % 2
                        if idx == 0:
                            x, y = 730, 50
                            sample_img.paste(patch_1, (x, y))
                        elif idx == 1:
                            x, y = 1085, 50
                            sample_img.paste(patch_2, (x, y))
                        sample_img.save(sample_path)
                        applied_count += 1
                        
                    except Exception as e:
                        print(f"[WARNING] Failed to process {sample_path}: {e}")
                
                print(f"[INFO] Overwrote adversarial patch to {applied_count}/{num_samples} samples in {dir_name}")


if __name__ == "__main__":
    train()
