import glob
import json
import logging
import os
from dataclasses import dataclass, field
from functools import partial
from typing import Dict, List, Optional, Union, Literal, Tuple, Any
from types import MethodType
from torchvision import transforms
import torch.nn as nn
from torchvision.transforms.functional import to_pil_image
import random
import numpy as np
from transformers import set_seed as hf_set_seed
import torch
import deepspeed
from transformers import Trainer
from transformers.trainer_pt_utils import nested_detach
from transformers.utils import is_sagemaker_mp_enabled
from transformers.trainer import (
    PreTrainedModel,
    is_peft_available, 
    PeftModel,
    TRAINING_ARGS_NAME,
    WEIGHTS_NAME,
    SAFE_WEIGHTS_NAME
)
from transformers.integrations import is_deepspeed_zero3_enabled
import math
import copy
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset
import sys

# --- For on-epoch inference printing
from PIL import Image
from transformers import TrainerCallback

import torch.nn.functional as F
import transformers
from accelerate.utils import DistributedType
from deepspeed import zero
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus
from transformers import AutoModel, AutoTokenizer

from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

IGNORE_INDEX = -100
logger = logging.getLogger(__name__)
logger.setLevel(logging.ERROR)

# Set random seed for reproducibility
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

# --- Atomic save helpers to avoid truncated files if process ends mid-write ---

def atomic_save_image(pil_img, final_path: str) -> None:
    import os
    import tempfile
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    base = os.path.basename(final_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=os.path.dirname(final_path))
    try:
        with os.fdopen(fd, "wb") as f:
            pil_img.save(f, format="PNG")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


def atomic_save_tensor(tensor, final_path: str) -> None:
    import os
    import tempfile
    import torch as _torch
    os.makedirs(os.path.dirname(final_path), exist_ok=True)
    base = os.path.basename(final_path)
    fd, tmp_path = tempfile.mkstemp(prefix=f".{base}.", suffix=".tmp", dir=os.path.dirname(final_path))
    try:
        with os.fdopen(fd, "wb") as f:
            _torch.save(tensor, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, final_path)
    finally:
        try:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="openbmb/MiniCPM-V-2")


@dataclass
class DataArguments:
    data_path: str = field(
        default=None, metadata={"help": "Path to the training data."}
    )
    eval_data_path: str = field(
        default=None, metadata={"help": "Path to the evaluation data."}
    )


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=10000,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    tune_vision: Optional[bool] = field(default=True)
    tune_llm: Optional[bool] = field(default=True)
    llm_type: str = field(default="minicpm")
    use_lora: Optional[bool] = field(default=False)
    max_slice_nums: Optional[int] = field(default=9)
    inference_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to a file containing the prompt."}
    )
    # device: str = field(default="cuda" if torch.cuda.is_available() else "cpu")

    # ----- Adversarial optimisation flags -----
    adv_training: Optional[bool] = field(
        default=False,
        metadata={"help": "Enable delta-only adversarial optimisation."},
    )
    pgd_steps: int = field(default=3, metadata={"help": "Number of PGD inner steps."})
    pgd_alpha: float = field(default=1 / 255, metadata={"help": "PGD step size."})
    pgd_epsilon: float = field(default=16 / 255, metadata={"help": "L-inf bound for delta."})

    adversarial_images_dir: Optional[str] = field(
        default="adversarial_output",
        metadata={"help": "Directory where the final adversarial image tensor & PNG are stored."},
    )

    # ----- quick inference demo after each epoch -----
    inference_image_path_1: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a test image used for on-epoch inference printing."},
    )
    inference_image_path_2: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a test image used for on-epoch inference printing."},
    )
    inference_folder_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path to a folder containing the test images used for on-epoch inference."},
    )
    inference_conv_json: Optional[str] = field(
        default=None,
        metadata={"help": "Conversation JSON path (with 'system','user','assistant' fields) for on-epoch inference."},
    )

    system_prompt: Optional[str] = field(
        default=None, metadata={"help": "Literal system prompt text."}
    )
    system_prompt_path: Optional[str] = field(
        default=None, metadata={"help": "Path to a file with system prompt."}
    )

    # ----- Weighted loss parameters -----
    element_weight: float = field(
        default=10.0,
        metadata={"help": "Weight multiplier for the tokens in 'ELEMENT: [Choice]' predictions."},
    )
    # Keeping these for compatibility but main logic uses element_weight
    element_reward_factor: float = field(default=0.1)
    use_focal_loss: bool = field(default=False)
    focal_alpha: float = field(default=0.25)
    focal_gamma: float = field(default=2.0)
    use_sequence_aware_loss: bool = field(default=True)
    ultra_targeted_loss: bool = field(default=False)
    confidence_penalty: bool = field(default=False)
    debug_ultra_targeted: bool = field(default=True)
    debug_pipeline_comparison: bool = field(default=True)


@dataclass
class LoraArguments:
    lora_r: int = 64
    lora_alpha: int = 64
    lora_dropout: float = 0.05
    lora_target_modules: str = r"llm\..*layers\.\d+\.self_attn\.(q_proj|k_proj|v_proj)"
    lora_weight_path: str = ""
    lora_bias: str = "none"
    q_lora: bool = False
    lora_modules_to_save: str = ""
    lora_layer_replication: Optional[List[Tuple[int, int]]] = None
    lora_layers_to_transform: Optional[List[int]] = None
    lora_layers_pattern: Optional[str] = None

local_rank = None
def rank0_print(*args):
    if local_rank == 0:
        print(*args)


class AdversarialPerturbation(nn.Module):
    """A learnable image-wide perturbation (Δ) that is optimised via PGD."""

    def __init__(
        self,
        shape: Tuple[int, int, int],  # (C, H, W)
        device: torch.device,
        epsilon: float = 8 / 255,
        patch_size: int = 14,
        scale_resolution: int = 448,
        max_slice_nums: int = 9,
        simulate_pil_quant: bool = False,
    ) -> None:
        super().__init__()
        self.epsilon = float(epsilon)
        self.register_parameter(
            "delta",
            torch.nn.Parameter(
            (torch.rand(1, 3, 295, 323, device=device) - 0.5) * 2 * 0.01
        )
        )

        self.patch_size = patch_size
        self.scale_resolution = scale_resolution
        self.max_slice_nums = max_slice_nums
        self.simulate_pil_quant = simulate_pil_quant

    # ---------------- MiniCPM slicing helpers (torch) ----------------

    @staticmethod
    def _ensure_divide(length: int, patch_size: int) -> int:
        return max(round(length / patch_size) * patch_size, patch_size)

    @staticmethod
    def _find_best_resize(original_size, scale_resolution: int, patch_size: int, allow_upscale=False):
        width, height = original_size
        if (width * height > scale_resolution * scale_resolution) or allow_upscale:
            r = width / height
            height = int(scale_resolution / math.sqrt(r))
            width = int(height * r)
        best_width = AdversarialPerturbation._ensure_divide(width, patch_size)
        best_height = AdversarialPerturbation._ensure_divide(height, patch_size)
        return (best_width, best_height)

    @staticmethod
    def _get_sliced_grid(image_size, max_slice_nums, scale_resolution):
        w, h = image_size
        log_ratio = math.log(w / h)
        ratio = w * h / (scale_resolution * scale_resolution)
        multiple = min(math.ceil(ratio), max_slice_nums)
        if multiple <= 1:
            return None
        candidate_split_nums = [x for x in [multiple - 1, multiple, multiple + 1] if x != 1 and x <= max_slice_nums]
        candidate_grids = []
        for split in candidate_split_nums:
            m = 1
            while m <= split:
                if split % m == 0:
                    candidate_grids.append([m, split // m])
                m += 1
        best_grid = [1, 1]
        min_error = float("inf")
        for g in candidate_grids:
            error = abs(log_ratio - math.log(g[0] / g[1]))
            if error < min_error:
                best_grid, min_error = g, error
        return best_grid

    # ------------------------------------------------------------
    # Processor-equivalent helpers for refine size
    # ------------------------------------------------------------

    @staticmethod
    def _get_refine_size(original_size, grid, scale_resolution, patch_size, allow_upscale=False):
        width, height = original_size
        grid_x, grid_y = grid

        refine_width = AdversarialPerturbation._ensure_divide(width, grid_x)
        refine_height = AdversarialPerturbation._ensure_divide(height, grid_y)

        grid_width = refine_width / grid_x
        grid_height = refine_height / grid_y

        best_grid_size = AdversarialPerturbation._find_best_resize(
            (grid_width, grid_height), scale_resolution, patch_size, allow_upscale=allow_upscale
        )
        refine_size = (int(best_grid_size[0] * grid_x), int(best_grid_size[1] * grid_y))
        return refine_size

    def _split_to_patches(self, img: torch.Tensor, grid):
        patches = []
        _, H, W = img.shape
        grid_x = int(W / grid[0])
        grid_y = int(H / grid[1])
        for i in range(0, H, grid_y):
            for j in range(0, W, grid_x):
                patch = img[:, i : i + grid_y, j : j + grid_x].clone()
                patches.append(patch)
        return patches

    def _reshape_by_patch(self, img: torch.Tensor, patch_size: int = 14):
        # img shape [3,H,W] divisible by patch_size
        patches = F.unfold(img.unsqueeze(0), (patch_size, patch_size), stride=(patch_size, patch_size))
        patches = patches.reshape(img.size(0), patch_size, patch_size, -1)
        patches = patches.permute(0, 1, 3, 2).reshape(img.size(0), patch_size, -1)
        return patches

    def _slice_image(self, img: torch.Tensor):
        C, H, W = img.shape
        original_size = (W, H)
        grid = self._get_sliced_grid(original_size, self.max_slice_nums, self.scale_resolution)
        patches = []
        if grid is None:
            best_size = self._find_best_resize(original_size, self.scale_resolution, self.patch_size, allow_upscale=True)
            resized = F.interpolate(img.unsqueeze(0), size=(best_size[1], best_size[0]), mode="bicubic", align_corners=False, antialias=True).squeeze(0)
            source = resized
        else:
            best_resize = self._find_best_resize(original_size, self.scale_resolution, self.patch_size)
            source = F.interpolate(
                img.unsqueeze(0), size=(best_resize[1], best_resize[0]), mode="bicubic", align_corners=False, antialias=True
            ).squeeze(0)

            refine_w, refine_h = self._get_refine_size(original_size, grid, self.scale_resolution, self.patch_size)
            refine_img = F.interpolate(
                img.unsqueeze(0), size=(refine_h, refine_w), mode="bicubic", align_corners=False, antialias=True
            ).squeeze(0)

            patches = self._split_to_patches(refine_img, grid)
        return source, patches

    # ------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------
    def forward(self, img: torch.Tensor, target_box: torch.Tensor, epoch: Optional[int] = None) -> torch.Tensor:
        if img.dim() != 3:
            raise ValueError(f"AdversarialPerturbation.forward expects a single image of shape [C, H, W], got {img.shape}")
        if target_box.dim() != 1 or target_box.numel() != 4:
            raise ValueError(f"target_box must be a 1D tensor of 4 elements, got {target_box}")

        C, H, W = img.shape
        # Ensure inputs are on the same device & dtype as delta so gradient flows back correctly
        if img.device != self.delta.device:
            img = img.to(self.delta.device)
        if img.dtype != self.delta.dtype:
            img = img.to(self.delta.dtype)
        perturbed_img = img.clone()
        delta = self.delta  # keep as Parameter (leaf) so grads accumulate

        x, y, h_box, w_box = target_box
        x1 = int(x)
        y1 = int(y)
        h_box = 295
        w_box = 323
        x2 = min(x1 + w_box, W)
        y2 = min(y1 + h_box, H)

        patch_resized = self.delta.squeeze(0)
        perturbed_img[:, y1:y2, x1:x2] += patch_resized

        # slice & patchify
        src, patch_list = self._slice_image(perturbed_img)
        slices = [src] + patch_list

        # normalise + reshape_by_patch per slice
        mean = torch.tensor((0.5, 0.5, 0.5), device=src.device, dtype=src.dtype).view(-1, 1, 1)
        std = mean
        out = []
        for sl in slices:
            norm = (sl - mean) / std
            out.append(self._reshape_by_patch(norm, self.patch_size))
        
        # DEBUG: Log custom preprocessing statistics occasionally
        if hasattr(self, '_debug_counter'):
            self._debug_counter += 1
        else:
            self._debug_counter = 0
        
        if self._debug_counter % 100 == 0:  # Every 100 forward passes
            rank0_print(f"\n[CUSTOM-PIPELINE-DEBUG] Forward pass #{self._debug_counter}:")
            rank0_print(f"  Input image: shape={img.shape}, mean={img.mean():.4f}, std={img.std():.4f}")
            rank0_print(f"  After slicing: {len(slices)} slices, main slice shape={src.shape}")
            rank0_print(f"  After normalization: mean={norm.mean():.4f}, std={norm.std():.4f}, min={norm.min():.4f}, max={norm.max():.4f}")
            rank0_print(f"  Normalization params: mean={mean.flatten()}, std={std.flatten()}")
            rank0_print(f"  Output: {len(out)} tensors, first shape={out[0].shape if out else 'None'}")
        
        return out

class CPMTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False):
        if "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        
        # DEBUG: check tgt_sizes before forward
        if labels is not None and self.state.global_step==0 and self.args.local_rank in (-1,0):
            ts = inputs.get("tgt_sizes", None)
            print("[DEBUG compute_loss] tgt_sizes len", len(ts) if ts else None, "types", [type(t) for t in ts][:5] if ts else None)

        if not self.args.use_lora:
            outputs = self.model(data = inputs, use_cache=False)
        else:
            with self.model._enable_peft_forward_hooks(**inputs):
                outputs = self.model.base_model(data = inputs, use_cache=False)
                
        if labels is not None:
            logits = outputs.logits  # (B, L, V)
            vocab_size = logits.size(-1)

            # Shift logits and labels so that tokens <t> predict label <t+1>.
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous().to(logits.device)

            # --- MODIFICATION START: Weighted Loss Calculation ---
            
            # 1. Calculate loss per token (reduction='none')
            # This returns a tensor of losses with the same shape as labels
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100, reduction='none')
            raw_loss = loss_fct(shift_logits.view(-1, vocab_size), shift_labels.view(-1))
            raw_loss = raw_loss.view(shift_labels.shape)
            
            # 2. Create a weight mask (default 1.0)
            loss_weights = torch.ones_like(raw_loss)
            
            # 3. Define Weighting Hyperparameters
            # Use argument if available, else default to 10.0
            ELEMENT_WEIGHT = getattr(self.args, 'element_weight', 10.0) 
            LOOKAHEAD_STEPS = 6     # How many tokens after "ELEMENT" to boost?

            # 4. Identify tokens corresponding to "ELEMENT"
            # We scan the tokenizer once to find the ID(s) for "ELEMENT"
            if not hasattr(self, "_element_token_ids"):
                self._element_token_ids = set()
                # Encode variations to catch however the tokenizer splits it
                variations = ["ELEMENT", " ELEMENT", "\nELEMENT"]
                for v in variations:
                    ids = self.tokenizer(v, add_special_tokens=False).input_ids
                    if ids: self._element_token_ids.update(ids)

            # 5. Apply weights
            # We iterate over the batch on CPU for simpler logic
            batch_size, seq_len = shift_labels.shape
            cpu_labels = shift_labels.detach().cpu().tolist()
            
            for b in range(batch_size):
                for t_idx, token_id in enumerate(cpu_labels[b]):
                    if token_id == -100: continue
                    
                    if token_id in self._element_token_ids:
                        # Found "ELEMENT". Boost this token and the next few (which contain the Choice)
                        start_idx = t_idx
                        end_idx = min(seq_len, t_idx + LOOKAHEAD_STEPS)
                        loss_weights[b, start_idx:end_idx] = ELEMENT_WEIGHT
            
            # 6. Compute final weighted mean
            # We multiply raw loss by weights, then normalize by the number of valid tokens
            weighted_loss = raw_loss * loss_weights
            valid_tokens_count = (shift_labels != -100).sum()
            
            if valid_tokens_count > 0:
                loss = weighted_loss.sum() / valid_tokens_count
            else:
                loss = weighted_loss.sum()
            
            # --- MODIFICATION END ---

            # -------- Debug block: print labelled text and model predictions once --------
            if getattr(self.state, "global_step", 0) < 2 and (self.args.local_rank in (-1, 0)):
                with torch.no_grad():
                    valid_mask = shift_labels.ne(-100)
                    full_mask = labels.ne(-100)
                    for b in range(min(2, shift_labels.size(0))):
                        # ... (Your existing debug print logic remains here) ...
                        full_ids = labels[b][full_mask[b]].tolist()
                        try:
                            full_text = self.tokenizer.decode(full_ids, skip_special_tokens=True)
                        except Exception:
                            full_text = str(full_ids)
                        
                        valid_ids = shift_labels[b][valid_mask[b]].tolist()
                        try:
                            lbl_text = self.tokenizer.decode(valid_ids, skip_special_tokens=True)
                        except Exception:
                            lbl_text = str(valid_ids)

                        pred_ids = shift_logits[b][valid_mask[b]].argmax(dim=-1).tolist()
                        try:
                            pred_text = self.tokenizer.decode(pred_ids, skip_special_tokens=True)
                        except Exception:
                            pred_text = str(pred_ids)
            # ---------------------------------------------------------------------------
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        return (loss, outputs) if return_outputs else loss

    def prediction_step(
        self,
        model: nn.Module,
        inputs: Dict[str, Union[torch.Tensor, Any]],
        prediction_loss_only: bool,
        ignore_keys: Optional[List[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        
        has_labels = (
            False
            if len(self.label_names) == 0
            else all(inputs.get(k) is not None for k in self.label_names)
        )
        return_loss = inputs.get("return_loss", None)
        if return_loss is None:
            return_loss = self.can_return_loss
        loss_without_labels = (
            True if len(self.label_names) == 0 and return_loss else False
        )

        inputs = self._prepare_inputs(inputs)
        if ignore_keys is None:
            if hasattr(self.model, "config"):
                ignore_keys = getattr(
                    self.model.config, "keys_to_ignore_at_inference", []
                )
            else:
                ignore_keys = []

        if has_labels or loss_without_labels:
            labels = nested_detach(tuple(inputs.get(name)
                                   for name in self.label_names))
            if len(labels) == 1:
                labels = labels[0]
        else:
            labels = None

        with torch.no_grad():
            if is_sagemaker_mp_enabled():
                raw_outputs = smp_forward_only(model, inputs)
                if has_labels or loss_without_labels:
                    if isinstance(raw_outputs, dict):
                        loss_mb = raw_outputs["loss"]
                        logits_mb = tuple(
                            v
                            for k, v in raw_outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        loss_mb = raw_outputs[0]
                        logits_mb = raw_outputs[1:]

                    loss = loss_mb.reduce_mean().detach().cpu()
                    logits = smp_nested_concat(logits_mb)
                else:
                    loss = None
                    if isinstance(raw_outputs, dict):
                        logits_mb = tuple(
                            v for k, v in raw_outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits_mb = raw_outputs
                    logits = smp_nested_concat(logits_mb)
            else:
                if has_labels or loss_without_labels:
                    with self.compute_loss_context_manager():
                        loss, outputs = self.compute_loss(
                            model, inputs, return_outputs=True
                        )
                    loss = loss.mean().detach()

                    if isinstance(outputs, dict):
                        logits = tuple(
                            v
                            for k, v in outputs.items()
                            if k not in ignore_keys + ["loss"]
                        )
                    else:
                        logits = outputs[1:]
                else:
                    loss = None
                    with self.compute_loss_context_manager():
                        outputs = model(**inputs)
                    if isinstance(outputs, dict):
                        logits = tuple(
                            v for k, v in outputs.items() if k not in ignore_keys
                        )
                    else:
                        logits = outputs
                    if self.args.past_index >= 0:
                        self._past = outputs[self.args.past_index - 1]

        if prediction_loss_only:
            return (loss, None, None)

        logits = nested_detach(logits)
        if len(logits) == 1:
            logits = logits[0]

        return (loss, logits, labels)
        
    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:
        model.train()
        # Normalise tgt_sizes: ensure flat list of IntTensor
        if "tgt_sizes" in inputs and len(inputs["tgt_sizes"]) > 0:
            flat_elems = []
            for elem in inputs["tgt_sizes"]:
                if isinstance(elem, (list, tuple)):
                    flat_elems.extend(elem)
                else:
                    flat_elems.append(elem)

            tensor_list = []
            import numpy as _np
            for elem in flat_elems:
                if isinstance(elem, torch.Tensor):
                    tensor_list.append(elem.to(dtype=torch.int32))
                elif isinstance(elem, _np.ndarray):
                    tensor_list.append(torch.tensor(elem, dtype=torch.int32))
                else:
                    tensor_list.append(torch.tensor(elem, dtype=torch.int32))

            inputs["tgt_sizes"] = tensor_list

            if self.args.local_rank in (-1,0) and self.state.global_step==0:
                print("[DEBUG training_step] tgt_sizes normalised len", len(tensor_list), "types", [type(t) for t in tensor_list][:3])

        inputs = self._prepare_inputs(inputs)

        if self.state.global_step < 2 and self.args.local_rank in (-1,0):
            ts = inputs.get("tgt_sizes", None)
            if ts is not None:
                print("[TGT-SIZES-DEBUG] len:", len(ts), "types:", [type(x) for x in ts][:5])
                if len(ts)==0 or not all(isinstance(x, torch.Tensor) for x in ts):
                    print("[TGT-SIZES-DEBUG] Problematic tgt_sizes, dumping input slice counts")
                    pv = inputs.get("pixel_values", [])
                    if pv:
                        print("   pixel_values set lengths:", [len(s) for s in pv][:5])

        if is_sagemaker_mp_enabled():
            loss_mb = smp_forward_backward(model, inputs, self.args.gradient_accumulation_steps)
            return loss_mb.reduce_mean().detach().to(self.args.device)

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs
        torch.cuda.empty_cache()

        if self.args.n_gpu > 1:
            loss = loss.mean()  # mean() to average on multi-gpu parallel training

        if self.use_apex:
            with amp.scale_loss(loss, self.optimizer) as scaled_loss:
                scaled_loss.backward()
        else:
            self.accelerator.backward(loss)

        return loss.detach() / self.args.gradient_accumulation_steps
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        # If we are executing this function, we are the process zero, so we don't check for that.
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Saving model checkpoint to {output_dir}")

        supported_classes = (PreTrainedModel,) if not is_peft_available() else (PreTrainedModel, PeftModel)
        # Save a trained model and configuration using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        if not isinstance(self.model, supported_classes):
            if state_dict is None:
                state_dict = self.model.state_dict()

            if isinstance(unwrap_model(self.model), supported_classes):
                unwrap_model(self.model).save_pretrained(
                    output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
                )
            else:
                logger.info("Trainer.model is not a `PreTrainedModel`, only saving its state dict.")
                if self.args.save_safetensors:
                    safetensors.torch.save_file(
                        state_dict, os.path.join(output_dir, SAFE_WEIGHTS_NAME), metadata={"format": "pt"}
                    )
                else:
                    torch.save(state_dict, os.path.join(output_dir, WEIGHTS_NAME))
        else:
            
            self.model.save_pretrained(
                output_dir, state_dict=state_dict, safe_serialization=self.args.save_safetensors
            )

        if self.tokenizer is not None:
            self.tokenizer.save_pretrained(output_dir)

        # Good practice: save your training arguments together with the trained model
        torch.save(self.args, os.path.join(output_dir, TRAINING_ARGS_NAME))

# -----------------------------
# AdversarialTrainer (delta-only optimisation)
# -----------------------------

from deepspeed.runtime.zero.partition_parameters import GatheredParameters


class AdversarialTrainer(CPMTrainer):
    """Trainer that freezes all model weights and optimises only a single
    perturbation tensor (model.adversarial.delta) via a Projected Gradient
    Descent (PGD) inner loop at every training step.
    """

    def training_step(self, model: nn.Module, inputs: Dict[str, Union[torch.Tensor, Any]]) -> torch.Tensor:  # type: ignore
        # Unwrap DDP/FSDP wrappers to access custom attributes
        model_unwrapped = model.module if hasattr(model, "module") else model

        if "pixel_values" not in inputs:
            print("[DEBUG] No pixel_values found, using super().training_step")
            return super().training_step(model, inputs)

        # Extract raw images and clean up inputs (clean pattern like callback)
        raw_imgs = inputs.get("raw_image")
        image_ids = inputs.get("image_id")
        
        if raw_imgs is None:
            return super().training_step(model, inputs)
            
        clean_pixels = [r.clone().detach() for r in raw_imgs]
        current_epoch = int(self.state.epoch) if self.state.epoch is not None else 0

        # Hyper-parameters from TrainingArguments
        pgd_steps = getattr(self.args, "pgd_steps", 4)
        pgd_alpha = getattr(self.args, "pgd_alpha", 0.5 / 255)
        pgd_epsilon = getattr(self.args, "pgd_epsilon", 8 / 255)

        # Convenience ref to delta parameter
        delta_param = model_unwrapped.adversarial.delta if hasattr(model_unwrapped, "adversarial") else None

        # ---------------- PGD inner loop ----------------
        for step in range(pgd_steps):
            # Apply adversarial perturbation (clean pattern like callback)
            perturbed_pixel_values = []
            for i, raw_img in enumerate(clean_pixels):
                # Determine patch position based on image_id (same as callback)
                if image_ids is not None and len(image_ids) > i:
                    img_id = image_ids[i].item()
                    if img_id % 2 == 1:
                        patch_box = torch.tensor([1085, 50, 295, 323])
                    else:
                        patch_box = torch.tensor([730, 50, 295, 323])
                else:
                    patch_box = torch.tensor([730, 50, 295, 323])
                
                # Apply adversarial processing (same as callback)
                perturbed_slices = model_unwrapped.adversarial(raw_img, patch_box, epoch=current_epoch)
                perturbed_pixel_values.append(perturbed_slices)
            
            # Replace pixel_values with adversarial ones (clean pattern like callback)
            inputs["pixel_values"] = perturbed_pixel_values

            # Compute loss
            model.train()
            batch_inputs = self._prepare_inputs(inputs)
            if "image_id" in batch_inputs:
                del batch_inputs["image_id"]
            loss = self.compute_loss(model, batch_inputs)

            # PGD update
            loss.backward()
            if delta_param is not None and delta_param.grad is not None:
                if self.accelerator.is_main_process:
                    print(f"[DEBUG] PGD step {step+1}: delta L1 norm: {delta_param.data.abs().sum().item():.6f}, L-inf norm: {delta_param.data.abs().max().item():.6f}")
                    print(f"[DEBUG] PGD step {step+1}: grad L1 norm: {delta_param.grad.abs().sum().item():.6f}, L-inf norm: {delta_param.grad.abs().max().item():.6f}")
                with torch.no_grad():
                    # PGD step
                    updated_delta = delta_param.data - pgd_alpha * delta_param.grad.data.sign()
                    updated_delta = torch.clamp(updated_delta, -pgd_epsilon, pgd_epsilon)
                    delta_param.data.copy_(updated_delta)
                delta_param.grad.zero_()

        # Final forward pass for logging
        perturbed_pixel_values = []
        for i, raw_img in enumerate(clean_pixels):
            if image_ids is not None and len(image_ids) > i:
                img_id = image_ids[i].item()
                if img_id % 2 == 1:
                    patch_box = torch.tensor([1085, 50, 295, 323])
                else:
                    patch_box = torch.tensor([730, 50, 295, 323])
            else:
                patch_box = torch.tensor([730, 50, 295, 323])
            
            perturbed_slices = model_unwrapped.adversarial(raw_img, patch_box, epoch=current_epoch)
            perturbed_pixel_values.append(perturbed_slices)
        
        inputs["pixel_values"] = perturbed_pixel_values

        model.train()
        batch_inputs = self._prepare_inputs(inputs)
        if "image_id" in batch_inputs:
            del batch_inputs["image_id"]
        loss_final = self.compute_loss(model, batch_inputs)

        return loss_final.detach()

class SupervisedDataset(Dataset):
    """Dataset for supervised fine-tuning."""

    def __init__(
        self,
        raw_data,
        tokenizer,
        processor=None,
        system_prompt: str | None = None,
    ):
        super(SupervisedDataset, self).__init__()
        self.raw_data = raw_data
        self.tokenizer = tokenizer
        self.processor = processor  # AutoProcessor providing image_processor & tokenizer
        self.system_prompt = system_prompt

    def __len__(self):
        return len(self.raw_data)

    def __getitem__(self, i) -> Dict[str, torch.Tensor]:
        try:
            img_field = self.raw_data[i]["image"]
            if isinstance(img_field, str):
                img_path = img_field
            elif isinstance(img_field, Dict):
                # take the first image in dict for delta optimisation
                img_path = list(img_field.values())[0]
            else:
                raise ValueError("Unsupported image specification in data entry")

            pil_img = Image.open(img_path).convert("RGB")
            raw_tensor = torch.from_numpy(np.array(pil_img)).permute(2, 0, 1).float() / 255.0
            conv_raw = copy.deepcopy(self.raw_data[i]["conversations"])

            role_map = {"human": "user", "gpt": "assistant"}
            conv = []
            for m in conv_raw:
                if "role" in m and "content" in m:
                    conv.append(m)
                elif "from" in m and "value" in m:
                    new_role = role_map.get(m["from"].lower(), m["from"].lower())
                    conv.append({"role": new_role, "content": m["value"]})
                else:
                    raise ValueError("Conversation item must have (role,content) or (from,value) keys")

            if self.system_prompt:
                if conv and conv[0]["role"] == "system":
                    conv[0]["content"] = self.system_prompt  # replace
                else:
                    conv.insert(0, {"role": "system", "content": self.system_prompt})

            # Build prompt string using chat template
            prompt_full_str = self.tokenizer.apply_chat_template(conv, tokenize=False, add_generation_prompt=False)

            if self.processor is not None:
                # Use clean processor pattern like callback - get all inputs at once
                inputs = self.processor(text=prompt_full_str, images=[pil_img], return_tensors="pt", max_slice_nums=9)
                
                # Extract individual components and update inputs dict
                input_ids = inputs["input_ids"][0]
                pixel_values = inputs["pixel_values"][0]  # This will be replaced during training
                tgt_sizes = inputs["tgt_sizes"][0]  # Extract tgt_sizes as tensor
                attention_mask = inputs["attention_mask"][0]
                image_bound = inputs["image_bound"][0]
                
                # Update inputs dict to remove batch dimension
                inputs.update({
                    "input_ids": input_ids,
                    "attention_mask": attention_mask,
                    "pixel_values": pixel_values,
                    "tgt_sizes": tgt_sizes,
                    "image_bound": image_bound,
                })

            # Labels (mask all but last assistant turn)
            if self.processor is not None:
                try:
                    last_assistant_idx = len(conv) - 1 - next(i for i, m in enumerate(reversed(conv)) if m["role"] == "assistant")
                    conv_trunc = copy.deepcopy(conv)
                    conv_trunc[last_assistant_idx]["content"] = ""

                    proc_trunc = self.processor(
                        text=self.tokenizer.apply_chat_template(conv_trunc, tokenize=False, add_generation_prompt=False),
                        images=[pil_img],
                        return_tensors="pt",
                        max_slice_nums=9,
                    )
                    base_len = len(proc_trunc["input_ids"][0])

                    # Build labels: mask everything before the start of the last assistant content
                    assistant_text = conv[last_assistant_idx]["content"]
                    content_ids = self.tokenizer(assistant_text, add_special_tokens=False).input_ids
                    anchor = content_ids[: min(5, len(content_ids))] if len(content_ids) > 0 else []

                    def _find_subseq(haystack, needle, start, end):
                        if not needle:
                            return None
                        end = min(end, len(haystack))
                        start = max(0, start)
                        for pos in range(start, end - len(needle) + 1):
                            if haystack[pos:pos + len(needle)] == needle:
                                return pos
                        return None

                    candidate_window = 10
                    pos = _find_subseq(
                        input_ids.tolist(),
                        anchor,
                        base_len - candidate_window,
                        base_len + candidate_window,
                    ) if anchor else None

                    if pos is not None:
                        start_idx = pos
                    else:
                        start_idx = max(0, base_len - 2)

                    labels = input_ids.clone().to(torch.long)
                    labels[:start_idx] = IGNORE_INDEX

                    # ---------- DEBUG: show masking details on first few samples ----------
                    if i < 2:  # only print for the first two samples to avoid spam
                        lbl_cnt = labels.ne(IGNORE_INDEX).sum().item()
                        img_tok_cnt = (input_ids == self.tokenizer.convert_tokens_to_ids("<image_start>")).sum().item() if hasattr(self.tokenizer, "convert_tokens_to_ids") else -1
                        # print("[MASK-DEBUG] sample", i,
                        #       "| total tokens", input_ids.size(0),
                        #       "| unmasked tokens", lbl_cnt,
                        #       "| start_idx", start_idx,
                        #       "| image_tokens", img_tok_cnt)
                        # Show first 40 decoded tokens for manual inspection
                        try:
                            dec_input = self.tokenizer.decode(input_ids[:start_idx], skip_special_tokens=False)
                            dec_labels = self.tokenizer.decode(input_ids[start_idx:start_idx+40], skip_special_tokens=False)
                            # print("[MASK-DEBUG]", dec_input)
                            # print("[MASK-DEBUG] first 40 label span tokens:", dec_labels)
                        except Exception as _e_dec:
                            print("[MASK-DEBUG] decode failed:", _e_dec)
                        try:
                            pre_start_dec = self.tokenizer.decode(input_ids[max(0, start_idx-40):start_idx], skip_special_tokens=False)
                            # print("[MASK-DEBUG] 40 tokens before span:", pre_start_dec)
                        except Exception as _e_dec2:
                            print("[MASK-DEBUG] decode pre-span failed:", _e_dec2)
                
                except Exception as e_lbl:
                    logger.error(f"label build via processor failed: {e_lbl}")
                    _, labels = tokenise_mask_last(conv, self.tokenizer)
            else:
                _, labels = tokenise_mask_last(conv, self.tokenizer)

            # Ensure label tensor length matches input_ids
            if labels.size(0) != input_ids.size(0):
                seq_len = input_ids.size(0)
                if labels.size(0) > seq_len:
                    labels = labels[:seq_len]
                else:
                    pad_len = seq_len - labels.size(0)
                    labels = torch.cat([labels, torch.full((pad_len,), IGNORE_INDEX, dtype=torch.long)])

            if 'id' in self.raw_data[i]:
                image_id = int(self.raw_data[i]['id'])
                
            # Return clean inputs dict with additional training info
            ret = dict(
                **inputs,
                position_ids=torch.arange(len(input_ids)),
                labels=labels,
                raw_image=raw_tensor,
                image_id=image_id,
            )
        except Exception as e:
            logger.error(f"data fetch error idx={i}: {e}")
            # Fallback to raw text only path
            try:
                input_ids, labels = tokenise_mask_last(conv, self.tokenizer)
                ret = dict(
                    input_ids=input_ids,
                    position_ids=torch.arange(len(input_ids)),
                    labels=labels,
                    attention_mask=torch.ones_like(input_ids, dtype=torch.bool),
                    pixel_values=[],
                    tgt_sizes=[],
                    image_bound=[],
                    raw_image=raw_tensor if 'raw_tensor' in locals() else None,
                )
                return ret
            except Exception as e2:
                logger.error(f"fallback failed: {e2}")
                raise
        return ret

        
def data_collator(examples, padding_value=0, max_length=4096):
    # For each example, crop from the left so that the last non-ignored label
    # token is inside the max_length window. This avoids losing the answer span
    # when the sequence is long (e.g., due to many vision tokens).

    cropped_input_ids = []
    cropped_position_ids = []
    cropped_labels = []
    cropped_attention = []

    for idx, ex in enumerate(examples):
        ids = ex["input_ids"]
        lbl = ex["labels"]
        attn = ex["attention_mask"]

        seq_len = ids.size(0)
        # Find last non-ignored label token position
        non_ignored = (lbl != -100).nonzero(as_tuple=False).view(-1)
        if non_ignored.numel() > 0:
            last_pos = int(non_ignored[-1].item())
            start_idx = max(0, last_pos - (max_length - 1))
        else:
            # No labels: keep the tail of the sequence by default
            start_idx = max(0, seq_len - max_length)
        end_idx = min(seq_len, start_idx + max_length)

        ids_s = ids[start_idx:end_idx]
        lbl_s = lbl[start_idx:end_idx]
        attn_s = attn[start_idx:end_idx]
        pos_s = torch.arange(ids_s.size(0))

        cropped_input_ids.append(ids_s)
        cropped_labels.append(lbl_s)
        cropped_attention.append(attn_s)
        cropped_position_ids.append(pos_s)

    def pad_list(seq_list, batch_first, padding_value):
        return pad_sequence(seq_list, batch_first=batch_first, padding_value=padding_value)

    input_ids = pad_list(cropped_input_ids, batch_first=True, padding_value=padding_value)
    position_ids = pad_list(cropped_position_ids, batch_first=True, padding_value=padding_value)
    targets = pad_list(cropped_labels, batch_first=True, padding_value=-100)
    attention_mask = pad_list(cropped_attention, batch_first=True, padding_value=padding_value)

    # Keep processor outputs as-is (clean pattern like callback)
    pixel_values = [example["pixel_values"] for example in examples]
    raw_images = [example.get("raw_image") for example in examples]
    image_bound = [example["image_bound"] for example in examples]
    tgt_sizes = [example["tgt_sizes"] for example in examples]
    
    batch = {
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": targets,
        "attention_mask": attention_mask,
        "image_bound": image_bound,
        "tgt_sizes": tgt_sizes,
        "pixel_values": pixel_values,
        "raw_image": raw_images,
    }
    if 'image_id' in examples[0]:
        batch['image_id'] = torch.tensor([example['image_id'] for example in examples], dtype=torch.long)
    return batch


# ---------------------------------------------------------------------
# Helper : tokenise conversation using tokenizer's chat template & mask only last assistant turn
# ---------------------------------------------------------------------

def tokenise_mask_last(conversation: List[Dict[str, str]], tokenizer):
    """Tokenise conversation via tokenizer.apply_chat_template.  The label
    tensor keeps only the last assistant turn; all previous tokens are set
    to IGNORE_INDEX so the loss is computed only on that span.
    """
    assert any(msg["role"] == "assistant" for msg in conversation), "Need at least one assistant message"

    # 1) Identify index of last assistant turn
    last_assistant_idx = len(conversation) - 1 - next(i for i,m in enumerate(reversed(conversation)) if m["role"]=="assistant")

    # 2) Build two prompt variants using chat template
    #    a) full conversation
    prompt_full_str = tokenizer.apply_chat_template(conversation, tokenize=False, add_generation_prompt=False)
    input_ids = tokenizer(prompt_full_str, add_special_tokens=False).input_ids

    #    b) conversation truncated right before last assistant *content* (keep role header but empty content)
    conv_trunc = conversation[:]
    truncated = copy.deepcopy(conversation[last_assistant_idx])
    truncated["content"] = ""
    conv_trunc[last_assistant_idx] = truncated
    prompt_trunc_str = tokenizer.apply_chat_template(conv_trunc, tokenize=False, add_generation_prompt=False)
    start = len(tokenizer(prompt_trunc_str, add_special_tokens=False).input_ids)

    input_ids = torch.tensor(input_ids, dtype=torch.long)

    # 3) Build labels: mask everything < start
    labels = input_ids.clone()
    labels[:start] = IGNORE_INDEX

    # 4) Optionally ensure EOS prediction for final token (already in labels)
    return input_ids, labels

class EpochInferenceCallback(TrainerCallback):
    """Callback that prints the model's answer on a fixed test prompt & image
    at the end of every training epoch (rank-0 only).
    Now supports a folder of images: will run inference for each image in the folder.
    Also tracks per-sample loss for debugging.
    """

    def __init__(self, device, tokenizer, image_path: str | None, conv_json: str | None, raw_image_path: str | None):
        self.device = device
        self.tokenizer = tokenizer
        self.images = []  # List of (PIL.Image, filename)
        if image_path:
            if os.path.isdir(image_path):
                # Load all images in the directory
                exts = ('.png', '.jpg', '.jpeg', '.bmp', '.tiff')
                for fname in sorted(os.listdir(image_path)):
                    if fname.lower().endswith(exts):
                        fpath = os.path.join(image_path, fname)
                        try:
                            img = Image.open(fpath).convert("RGB")
                            self.images.append((img, fname))
                        except Exception as e:
                            rank0_print(f"[WARN] Failed to load image {fpath}: {e}")
            elif os.path.isfile(image_path):
                try:
                    img = Image.open(image_path).convert("RGB")
                    self.images.append((img, os.path.basename(image_path)))
                except Exception as e:
                    rank0_print(f"[WARN] Failed to load image {image_path}: {e}")
        if conv_json and os.path.isfile(conv_json):
            with open(conv_json, "r", encoding="utf-8") as f:
                self.conv_data = json.load(f)
        else:
            self.conv_data = None
        if raw_image_path:
            self.raw_image = Image.open(raw_image_path).convert("RGB")

    def on_epoch_end(self, args, state, control, **kwargs):
        if args.local_rank not in (-1, 0):
            return  # only log from rank-0

        model = kwargs.get("model")
        if model is None or not self.images or self.conv_data is None:
            return

        model_eval = model.module if hasattr(model, "module") else model
        model_eval.eval()
        

        self.on_epoch_end1(model_eval, state)
        # # Compare results
        # self.compare_results(results_official, results_custom, state)
        
    def on_epoch_end1(self, model_eval, state):
        """Test using OFFICIAL preprocessing pipeline (current inference method)"""
        rank0_print(f"\n[OFFICIAL PIPELINE TEST]")
        
        results = {}
        import re
        
        for test_image, fname in self.images[:30]:  # Test all 30 samples for complete comparison
            try:
                # Extract sample number from filename
                match = re.match(r"sample_(\d+)\.png", fname)
                if not match:
                    continue
                sample_num = int(match.group(1))
                
                # Build perturbed image if adversarial delta is present
                perturbed_pil = test_image
                base_tensor = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0
                if hasattr(model_eval, "adversarial") and hasattr(model_eval.adversarial, "delta"):
                    try:
                        delta = model_eval.adversarial.delta.detach()
                        perturbed = base_tensor.clone()
                        # Same logic as training: even sample_num → [730, 50], odd sample_num → [1085, 50]
                        if sample_num % 2 == 1:  # Odd sample number
                            patch_x, patch_y = 1085, 50
                            perturbed[:, 50:345, 1085:1408] += delta.to(base_tensor.device, dtype=base_tensor.dtype).squeeze(0)
                        else:  # Even sample number
                            patch_x, patch_y = 730, 50
                            perturbed[:, 50:345, 730:1053] += delta.to(base_tensor.device, dtype=base_tensor.dtype).squeeze(0)
                        
                        adv_tensor = perturbed.clone().cpu().squeeze(0).clamp(0, 1)
                        perturbed_pil = to_pil_image(adv_tensor.cpu()).convert("RGB")
                    except Exception as e_img:
                        rank0_print(f"[WARN] Failed to compose adversarial image for callback: {e_img}")

                # Build conversation msgs list expected by MiniCPM chat
                msgs = [
                    {"role": "system", "content": self.conv_data["system"]},
                    {"role": "user", "content": [perturbed_pil, self.conv_data["user"][0]]},
                    {"role": "assistant", "content": self.conv_data["assistant"]},
                    {"role": "user", "content": self.conv_data["user"][1]}
                ]

                # OFFICIAL INFERENCE
                with torch.no_grad():
                    answer = model_eval.chat(image=perturbed_pil, msgs=msgs, tokenizer=self.tokenizer, temperature=0, max_new_tokens=200, do_sample=False, use_cache=False)

                
                rank0_print(f"  Sample {sample_num}: {answer} (Official)")
                
            except Exception as e:
                rank0_print(f"[WARN] Official pipeline failed for {fname}: {e}")
                
        return results
    
    def on_epoch_end2(self, model_eval, state):
        """Test using CUSTOM preprocessing pipeline (exact training replica)"""
        rank0_print(f"\n[CUSTOM PIPELINE TEST]")
        
        results = {}
        import re
        
        for test_image, fname in self.images:  # Test all 30 samples to match official pipeline
            try:
                # Extract sample number from filename
                match = re.match(r"sample_(\d+)\.png", fname)
                if not match:
                    continue
                sample_num = int(match.group(1))
                raw_tensor = torch.from_numpy(np.array(test_image)).permute(2, 0, 1).float() / 255.0
                conv = [
                    {"role": "system", "content": self.conv_data["system"]},
                    {"role": "user", "content": [test_image, self.conv_data["user"][0]]},
                    {"role": "assistant", "content": self.conv_data["assistant"]},
                    {"role": "user", "content": self.conv_data["user"][1]}
                ]

                copy_msgs = copy.deepcopy(conv)
                processed_msgs = []
                for m in copy_msgs:
                    content = m["content"]
                    if isinstance(content, str):
                        content_list = [content]
                    else:
                        content_list = content
                    parts = []
                    for c in content_list:
                        if isinstance(c, Image.Image):
                            parts.append("(<image>./</image>)")
                        elif isinstance(c, np.ndarray):
                            parts.append("(<audio>./</audio>)")
                        else:
                            parts.append(c)
                    processed_msgs.append({"role": m["role"], "content": "\n".join(parts)})

                prompt_full_str = self.tokenizer.apply_chat_template(
                    processed_msgs,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                from transformers import AutoProcessor
                processor = AutoProcessor.from_pretrained(model_eval.config._name_or_path, trust_remote_code=True)
                model_device = next(model_eval.parameters()).device
                proc_out = processor(text=prompt_full_str, images=[test_image], return_tensors="pt", max_slice_nums=9).to(model_device)
    
                if sample_num % 2 == 1:
                    patch_box = torch.tensor([1085, 50, 295, 323])
                else:
                    patch_box = torch.tensor([730, 50, 295, 323])
                custom_pixel_values = model_eval.adversarial(raw_tensor, patch_box)
                proc_out["pixel_values"][0] = custom_pixel_values
                
                proc_out.pop("image_sizes")
                with torch.no_grad():
                    res, outputs = model_eval.generate(
                        **proc_out,
                        tokenizer=self.tokenizer,
                        max_new_tokens=200,
                        num_beams=3,
                        repetition_penalty=1.2,
                        do_sample=False,
                        use_cache=False
                    )
                                             # Use res directly since it contains the correct answer
                    answer = res[0] if isinstance(res, list) else res
                    print(f"[CUSTOM] Outputs: {res}")
                    print(f"[CUSTOM] Using answer from res: {answer}")
                
                print(f"[CUSTOM] Sample {sample_num} answer: {answer}")
                
                # Extract element prediction - use the answer we got
                text_to_parse = answer if answer.strip() else (res[0] if isinstance(res, list) else res)
                element_match = re.search(r'ELEMENT:\s*([A-E])', text_to_parse)
                if element_match:
                    element = element_match.group(1)
                else:
                    element = "?"
                
                results[sample_num] = {
                    'prediction': element,
                    'full_answer': text_to_parse,
                    'pipeline': 'custom'
                }
                print(f"  Sample {sample_num}: {element} (Custom)")
                
            except Exception as e:
                print(f"[ERROR] Sample {sample_num}: {e}")
                import traceback
                print(f"[ERROR] Traceback: {traceback.format_exc()}")
                results[sample_num] = {
                    'prediction': "ERROR",
                    'full_answer': str(e),
                    'pipeline': 'custom'
                }
        
        return results
    
    # def compare_results(self, results_official, results_custom, state):
    #     """Compare results from both pipelines"""
    #     rank0_print(f"\n[PIPELINE COMPARISON RESULTS]")
        
    #     matches = 0
    #     total = 0
        
    #     for sample_num in sorted(set(results_official.keys()) | set(results_custom.keys())):
    #         official = results_official.get(sample_num, {})
    #         custom = results_custom.get(sample_num, {})
            
    #         official_pred = official.get('prediction', 'MISSING')
    #         custom_pred = custom.get('prediction', 'MISSING')
            
    #         match = "✅" if official_pred == custom_pred else "❌"
    #         if official_pred == custom_pred and official_pred != 'MISSING':
    #             matches += 1
    #         if official_pred != 'MISSING' and custom_pred != 'MISSING':
    #             total += 1
            
    #         rank0_print(f"  Sample {sample_num:2d}: Official={official_pred} | Custom={custom_pred} {match}")
        
    #     if total > 0:
    #         match_rate = (matches / total) * 100
    #         rank0_print(f"\nPIPELINE MATCH RATE: {matches}/{total} = {match_rate:.1f}%")
            
    #         if match_rate < 80:
    #             rank0_print(f"⚠️  PIPELINE MISMATCH DETECTED! Only {match_rate:.1f}% agreement")
    #             rank0_print("This explains the training/inference gap!")
    #         else:
    #             rank0_print(f"✅ Pipelines mostly agree ({match_rate:.1f}%)")
        
    #     # Additional analysis
    #     official_e_count = sum(1 for r in results_official.values() if r.get('prediction') == 'E')
    #     custom_e_count = sum(1 for r in results_custom.values() if r.get('prediction') == 'E')
        
    #     rank0_print(f"\nE PREDICTION RATES:")
    #     rank0_print(f"  Official pipeline: {official_e_count}/{len(results_official)} = {100*official_e_count/max(1,len(results_official)):.1f}%")
    #     rank0_print(f"  Custom pipeline:   {custom_e_count}/{len(results_custom)} = {100*custom_e_count/max(1,len(results_custom)):.1f}%")
        
    #     rank0_print(f"{'='*80}\n")

def make_supervised_data_module(
    tokenizer: transformers.PreTrainedTokenizer,
    data_args,
    system_prompt: str | None,
    data_collator=None,
    processor=None,
) -> Dict:
    """Make dataset and collator for supervised fine-tuning."""
    dataset_cls = SupervisedDataset

    rank0_print("Loading data...")

    train_json = json.load(open(data_args.data_path, "r"))
    train_dataset = dataset_cls(train_json, tokenizer, processor=processor, system_prompt=system_prompt)

    if data_args.eval_data_path:
        eval_json = json.load(open(data_args.eval_data_path, "r"))
        eval_dataset = dataset_cls(eval_json, tokenizer, processor=processor, system_prompt=system_prompt)
    else:
        eval_dataset = None

    return dict(
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )


def build_transform():
    IMAGENET_INCEPTION_MEAN = (0.5, 0.5, 0.5) # timm.data.IMAGENET_INCEPTION_MEAN
    IMAGENET_INCEPTION_STD = (0.5, 0.5, 0.5)  # timm.data.IMAGENET_INCEPTION_STD
    return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD
                ),
            ]
        )

def get_parameter_number(model):
    trainable_params, all_param = 0, 0
    for param in model.parameters():
        num_params = param.numel()
        # if using DS Zero 3 and the weights are initialized empty
        if num_params == 0 and hasattr(param, "ds_numel"):
            num_params = param.ds_numel

        all_param += num_params
        if param.requires_grad:
            trainable_params += num_params
        
    return {'Total': all_param, 'Trainable': trainable_params}


local_rank = 0


def train():
    global local_rank
    parser = transformers.HfArgumentParser(
        (ModelArguments, DataArguments, TrainingArguments, LoraArguments)
    )

    (
        model_args,
        data_args,
        training_args,
        lora_args,
    ) = parser.parse_args_into_dataclasses()
    # Resolve system prompt
    if training_args.system_prompt_path:
        import pathlib, json
        with open(training_args.system_prompt_path, "r", encoding="utf-8") as f:
            conv_data = json.load(f)
        training_args.system_prompt = conv_data["system"]

    if getattr(training_args, "deepspeed", None) : 
        training_args.distributed_state.distributed_type = DistributedType.DEEPSPEED

    compute_dtype = (
        torch.float16
        if training_args.fp16
        else (torch.bfloat16 if training_args.bf16 else torch.float32)
    )

    local_rank = training_args.local_rank
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    ddp = world_size != 1
    device_map = None
    if lora_args.q_lora:
        device_map = {"": int(os.environ.get("LOCAL_RANK") or 0)} if ddp else None
        if len(training_args.fsdp) > 0 or is_deepspeed_zero3_enabled():
            logging.warning(
                "FSDP or ZeRO3 are not incompatible with QLoRA."
            )
    
    model = AutoModel.from_pretrained(
        model_args.model_name_or_path,
        trust_remote_code=True,
        torch_dtype=torch.bfloat16,
        attn_implementation='sdpa',
        device_map=device_map,
        init_vision=True,
        init_audio=False,
        init_tts=False,
    )

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path, trust_remote_code=True
    )

    from transformers import AutoProcessor
    processor = AutoProcessor.from_pretrained(model_args.model_name_or_path, trust_remote_code=True)

    if not training_args.tune_vision:
        model.vpm.requires_grad_(False)
    if not training_args.tune_llm:
        model.llm.requires_grad_(False)
        
    if training_args.use_lora:
        if training_args.use_lora and training_args.tune_llm:
            raise ValueError("The model cannot simultaneously adjust LLM parameters and apply LoRA.")
            
        rank0_print("Currently using LoRA for fine-tuning the MiniCPM-V model.")
        for name, param in model.llm.named_parameters():
            param.requires_grad = False
        modules_to_save = ['embed_tokens','resampler']
        if training_args.tune_vision:
            modules_to_save.append('vpm')

        # --- LLaVA-style selection: only patch the first Linear layer found ---
        def find_all_linear_names(m):
            for n, mod in m.named_modules():
                if isinstance(mod, nn.Linear):
                    return [n]
            return []
        target_modules = find_all_linear_names(model)

        lora_config = LoraConfig(
            r=lora_args.lora_r,
            lora_alpha=lora_args.lora_alpha,
            target_modules=target_modules,
            lora_dropout=lora_args.lora_dropout,
            bias=lora_args.lora_bias,
            modules_to_save=modules_to_save,
        )
        if not hasattr(model, 'get_input_embeddings'):
            def get_input_embeddings(self):
                return self.llm.get_input_embeddings()
            model.get_input_embeddings = MethodType(get_input_embeddings, model)
        if lora_args.q_lora:
            model = prepare_model_for_kbit_training(
                model, use_gradient_checkpointing=training_args.gradient_checkpointing
            )
        model = get_peft_model(model, lora_config)
        if training_args.gradient_checkpointing:
            model.enable_input_require_grads()

    rank0_print(get_parameter_number(model))

    llm_type = training_args.llm_type    
    
    rank0_print(f'llm_type={llm_type}')

    
    # Load data
    if hasattr(model.config, "slice_config"):
        model.config.slice_config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.slice_config.to_dict()
    else:
        model.config.max_slice_nums = training_args.max_slice_nums
        slice_config = model.config.to_dict()

    if hasattr(model.config, "batch_vision_input"):
        batch_vision = model.config.batch_vision_input
    else:
        batch_vision = False

    transform_func = build_transform()
    data_module = make_supervised_data_module(
        tokenizer=tokenizer,
        data_args=data_args,
        system_prompt=training_args.system_prompt,
        data_collator=data_collator,
        processor=processor,
    )

    # -------------------------------------------------------------
    # Adversarial setup (freeze model + add delta parameter)
    # -------------------------------------------------------------
    if training_args.adv_training:
        # 1) Freeze every existing parameter
        for p in model.parameters():
            p.requires_grad = False

        # 2) Infer image shape from first training sample
        try:
            first_sample = data_module["train_dataset"][0]  # type: ignore
            sample_pixel = first_sample.get("raw_image", None)
            if sample_pixel is None:
                sample_pixel = first_sample.get("pixel_values", None)
            if sample_pixel is None:
                raise KeyError("No image tensor found in dataset sample")
            img_shape = sample_pixel.shape  # (C,H,W)
            print("img_shape", img_shape)
        except Exception:
            img_shape = (3, getattr(model.config, "image_size", 336), getattr(model.config, "image_size", 336))

        # 4) Attach adversarial module to model
        model.adversarial = AdversarialPerturbation(
            shape=img_shape,
            device=training_args.device,
            epsilon=training_args.pgd_epsilon,
            patch_size=model.config.patch_size if hasattr(model.config, "patch_size") else 14,
            scale_resolution=model.config.scale_resolution if hasattr(model.config, "scale_resolution") else 448,
            max_slice_nums=training_args.max_slice_nums,
        )

    # -------------------------------------------------------------
 
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}

    TrainerClass = AdversarialTrainer if training_args.adv_training else CPMTrainer

    trainer = TrainerClass(
        model=model,
        tokenizer=tokenizer,
        args=training_args,
        **data_module,
    )

    # Add inference callback if paths provided
    if training_args.inference_folder_path and training_args.inference_conv_json:
        trainer.add_callback(EpochInferenceCallback(training_args.device, tokenizer, training_args.inference_folder_path, training_args.inference_conv_json, training_args.inference_image_path_1))

    trainer.train()

    if training_args.adv_training and hasattr(model, "adversarial"):
        try:
            test_image_1 = Image.open(training_args.inference_image_path_1).convert("RGB")
            base_img_tensor_1 = torch.from_numpy(np.array(test_image_1)).permute(2, 0, 1).float() / 255.0
            test_image_2 = Image.open(training_args.inference_image_path_2).convert("RGB")
            base_img_tensor_2 = torch.from_numpy(np.array(test_image_2)).permute(2, 0, 1).float() / 255.0
            # Gather full delta if running with ZeRO-3
            if is_deepspeed_zero3_enabled():
                with zero.GatheredParameters([model.adversarial.delta]):
                    delta_full = model.adversarial.delta.detach().clone()
                delta = delta_full.to(base_img_tensor_1.dtype)
            else:
                delta = model.adversarial.delta.detach().to(base_img_tensor_1.dtype)
            
            # Ensure both tensors are on the same device
            device = delta.device
            base_img_tensor_1 = base_img_tensor_1.to(device)
            base_img_tensor_2 = base_img_tensor_2.to(device)
            
            rank0_print(f"[INFO] Base image shape: {base_img_tensor_1.shape}, dtype: {base_img_tensor_1.dtype}")
            perturbed_1 = base_img_tensor_1.clone()
            perturbed_1[:, 50:345, 1085:1408] += delta.squeeze(0)
            adv_img_tensor_1 = perturbed_1.clone().cpu().squeeze(0).clamp(0, 1)
            # Create output dir
            os.makedirs(training_args.adversarial_images_dir, exist_ok=True)
            img_path_1 = os.path.join(training_args.adversarial_images_dir, "adversarial_image_1.png")
            pil_img_1 = to_pil_image(adv_img_tensor_1.cpu())
            atomic_save_image(pil_img_1, img_path_1)

            patch_x, patch_y = 1085, 50
            patch_w, patch_h = 323, 295
            adversarial_card_1 = pil_img_1.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
            card_path_1 = os.path.join(training_args.adversarial_images_dir, 'adversarial_product_1.png')
            atomic_save_image(adversarial_card_1, card_path_1)
            rank0_print(f"[INFO] Adversarial product (patch) saved to {card_path_1}")

            perturbed_2 = base_img_tensor_2.clone()
            perturbed_2[:, 50:345, 730:1053] += delta.squeeze(0)
            adv_img_tensor_2 = perturbed_2.clone().cpu().squeeze(0).clamp(0, 1)
            # Create output dir
            os.makedirs(training_args.adversarial_images_dir, exist_ok=True)
            img_path_2 = os.path.join(training_args.adversarial_images_dir, "adversarial_image_2.png")
            pil_img_2 = to_pil_image(adv_img_tensor_2.cpu())
            atomic_save_image(pil_img_2, img_path_2)

            patch_x, patch_y = 730, 50
            patch_w, patch_h = 323, 295
            adversarial_card_2 = pil_img_2.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
            card_path_2 = os.path.join(training_args.adversarial_images_dir, 'adversarial_product_2.png')
            atomic_save_image(adversarial_card_2, card_path_2)
            rank0_print(f"[INFO] Adversarial product (patch) saved to {card_path_2}")

            patch_1 = Image.open(card_path_1).convert('RGB')
            patch_2 = Image.open(card_path_2).convert('RGB')
            
            # Apply adversarial patch to test samples
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
                            sample_img.paste(patch_2, (730, 50))
                        elif idx == 1:
                            sample_img.paste(patch_1, (1085, 50))
                        atomic_save_image(sample_img, sample_path)
                        applied_count += 1
                        
                    except Exception as e:
                        rank0_print(f"[WARNING] Failed to process {sample_path}: {e}")
                
                rank0_print(f"[INFO] Overwrote adversarial patch to {applied_count}/{num_samples} samples in {dir_name}")
            
        except Exception as e:
            rank0_print(f"[WARN] Failed to save adversarial image: {e}")
            import traceback
            rank0_print(f"[WARN] Full traceback: {traceback.format_exc()}")



if __name__ == "__main__":
    train()