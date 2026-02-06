import argparse
import json
import os
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.nn.parallel import DistributedDataParallel
from torch.distributed import init_process_group, destroy_process_group
from accelerate import Accelerator
from accelerate.utils import gather_object
from PIL import Image
from tqdm import tqdm
from transformers import (
    AutoModelForCausalLM,
    AutoProcessor,
    AutoTokenizer,
    get_scheduler,
    Trainer,
    TrainerCallback,
    TrainingArguments,
    BatchFeature,
)
from torchvision.transforms.functional import to_pil_image
import random
from typing import Dict, List, Optional, Tuple, Union
from torch.optim import AdamW
import logging
import time
from pathlib import Path
import torchvision.transforms.functional as TF
import math
from dataclasses import field
from transformers import HfArgumentParser

# Constants
_IGNORE_INDEX = -100

def set_seed(seed=17):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

class DifferentiablePhi3VPreprocess(torch.nn.Module):
    """
    Differentiable version of Phi3VImageProcessor for tensor inputs.
    Ensures all operations preserve gradient flow.
    """
    def __init__(
        self,
        num_crops=16,  # Default from Phi-3-vision config
        num_img_tokens=144,
        size=336,
        mean=[0.48145466, 0.4578275, 0.40821073],
        std=[0.26862954, 0.26130258, 0.27577711],
    ):
        super().__init__()
        self.num_crops = num_crops
        self.num_img_tokens = num_img_tokens
        self.size = size
        self.mean = torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1)
        self.std = torch.tensor(std, dtype=torch.float32).view(-1, 1, 1)
    
    def img_processor(self, img):
        """Normalize image tensor with gradient preservation."""
        return (img - self.mean.to(img.device)) / self.std.to(img.device)
    
    def pad_to_max_num_crops_tensor(self, images, max_crops=5):
        """Pad image tensor to max_crops with gradient preservation."""
        B, C, H, W = images.shape
        if B < max_crops:
            pad = torch.zeros(max_crops - B, C, H, W, dtype=images.dtype, device=images.device, requires_grad=True)
            images = torch.cat([images, pad], dim=0)
        return images
    
    def find_closest_aspect_ratio(self, aspect_ratio, width, height, image_size):
        """Find closest crop grid (w_crops, h_crops) under num_crops budget, mirroring official logic."""
        min_num, max_num = 1, self.num_crops
        target_ratios = []
        for n in range(min_num, max_num + 1):
            for i in range(1, n + 1):
                for j in range(1, n + 1):
                    if i * j <= max_num and i * j >= min_num:
                        target_ratios.append((i, j))
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio
    
    def hd_transform(self, img_tensor):
        """
        Dynamic HD transform to replicate official tiling (global first + HD grid).
        """
        # Input is [C, H, W]
        c, height, width = img_tensor.shape
        transposed = False
        if width < height:
            img_tensor = img_tensor.permute(0, 2, 1)
            width, height = height, width
            transposed = True
        
        ratio = (width / height)
        scale = 1
        while scale * math.ceil(scale / ratio) <= self.num_crops:
            scale += 1
        scale -= 1
        
        new_w = int(scale * self.size)
        new_h = int(new_w / ratio)
        
        # Resize
        img_b = img_tensor.unsqueeze(0)
        resized = F.interpolate(
            img_b,
            size=(new_h, new_w),
            mode='bicubic',
            align_corners=False,
            antialias=True
        )
        
        # Pad height to multiple of self.size (336), width unchanged
        target_h = int(math.ceil(new_h / self.size) * self.size)
        top_padding = int((target_h - new_h) / 2)
        bottom_padding = target_h - new_h - top_padding
        if top_padding > 0 or bottom_padding > 0:
            resized = F.pad(
                resized,
                (0, 0, top_padding, bottom_padding),  # (left, right, top, bottom)
                mode='constant',
                value=1.0
            )
        
        if transposed:
            resized = resized.permute(0, 1, 3, 2)
        
        final = resized.squeeze(0)
        return final, torch.tensor([final.shape[1], final.shape[2]], dtype=torch.long)
    
    def differentiable_reshape(self, im, h, w):
        """
        Extract non-overlapping patches using unfold to preserve gradients.
        Input im: [3, H, W]; returns [num_patches, 3, size, size]
        """
        im_batch = im.unsqueeze(0)  # [1, 3, H, W]
        patches = F.unfold(im_batch, kernel_size=self.size, stride=self.size)  # [1, 3*size*size, L]
        num_patches = patches.shape[-1]
        patches = patches.transpose(1, 2).contiguous().view(num_patches, 3, self.size, self.size)
        return patches
        
    def preprocess(self, images, return_tensors=None):
        """
        Main preprocessing method with preserved gradient flow.
        """
        # Convert to list of images
        if not isinstance(images, list):
            images = [images]
            
        # Process each image
        hd_images = []
        shapes = []
        
        for im in images:
            if im.dim() == 4:
                for i in range(im.size(0)):
                    transformed, shape = self.hd_transform(im[i])
                    normalized = self.img_processor(transformed)
                    hd_images.append(normalized)
                    shapes.append(shape)
            else:
                transformed, shape = self.hd_transform(im)
                normalized = self.img_processor(transformed)
                hd_images.append(normalized)
                shapes.append(shape)
        
        global_images = []
        for im in hd_images:
            im_batch = im.unsqueeze(0)  # Add batch dim
            global_im = F.interpolate(
                im_batch.float(),
                size=(self.size, self.size),
                mode='bicubic',
                align_corners=False,
                antialias=True
            ).to(im.dtype)
            global_images.append(global_im.squeeze(0))
            
        # Reshape HD images using differentiable reshape
        hd_images_reshape = []
        for im, shape in zip(hd_images, shapes):
            h, w = shape
            final = self.differentiable_reshape(im, h, w)
            hd_images_reshape.append(final)
            
        hd_images_with_global = []
        for global_im, crops in zip(global_images, hd_images_reshape):
            global_im_expanded = global_im.unsqueeze(0)
            combined = torch.cat([global_im_expanded, crops], dim=0)
            hd_images_with_global.append(combined)
            
        max_crops = self.num_crops + 1  # +1 for global image
        image_transformed = [self.pad_to_max_num_crops_tensor(im, max_crops) for im in hd_images_with_global]
        image_transformed = torch.stack(image_transformed, dim=0)
        
        data = {
            "pixel_values": image_transformed,
            "image_sizes": torch.stack([torch.tensor(s, dtype=torch.long) for s in shapes])
        }
        
        return BatchFeature(data=data, tensor_type=return_tensors)

    def forward(self, img):
        """Process image tensor with gradient preservation."""
        return self.preprocess(img)

class AdversarialPerturbation(torch.nn.Module):
    """
    Adversarial perturbation module that applies a learnable delta to input images.
    Uses the differentiable preprocessing pipeline to maintain gradient flow.
    """
    def __init__(self, shape, device, processor):
        super().__init__()
        self.delta = torch.nn.Parameter(torch.zeros(1, 3, 295, 323, device=device))
        self.device = device
        
        # Use Phi-3's normalization values from config
        self.mean = [0.48145466, 0.4578275, 0.40821073]
        self.std = [0.26862954, 0.26130258, 0.27577711]
        # Create differentiable preprocessing pipeline
        self.transform = DifferentiablePhi3VPreprocess(
            size=336,
            mean=self.mean,
            std=self.std,
            num_crops=16,
            num_img_tokens=processor.num_img_tokens
        )

    def forward(self, image_tensor, target_boxes):
        """
        Apply perturbation to image tensor with gradient preservation.
        """
        try:
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

            perturbed = torch.clamp(perturbed, 0, 1)
            processed = self.transform(perturbed)
            processed = processed.to(dtype=torch.bfloat16)
            return processed
                
        except Exception as e:
            print(f"\nError in forward pass: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

class AdversarialTrainer(Trainer):
    """Custom trainer for adversarial training with PGD."""
    
    def __init__(self, pgd_steps=3, pgd_alpha=1/255, pgd_epsilon=16/255, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.pgd_steps = pgd_steps
        self.pgd_alpha = pgd_alpha
        self.pgd_epsilon = pgd_epsilon
        
        # Store reference to adversarial module before DDP wrapping
        if hasattr(self.model, 'adversarial'):
            self.adversarial_module = self.model.adversarial
        else:
            base_model = get_unwrapped_model(self.model)
            self.adversarial_module = base_model.adversarial
        
        self._compat_checked_once = False

    def training_step(self, model, inputs, num_items_in_batch=None):
        """Perform a PGD training step."""
        try:
            # Clone text inputs so Phi-3-V in-place ops don't break later steps
            base_text_inputs = {
                "input_ids": inputs["input_ids"].detach().clone(),
                "labels": inputs["labels"].detach().clone(),
                "attention_mask": inputs.get("attention_mask", None),
            }
            if base_text_inputs["attention_mask"] is not None:
                base_text_inputs["attention_mask"] = base_text_inputs["attention_mask"].detach().clone()
            
            # Get raw images
            raw_images = inputs.pop("raw_image")
            raw_images = raw_images.to(self.adversarial_module.delta.device)
            
            # PGD inner loop
            for step in range(self.pgd_steps):
                model.zero_grad(set_to_none=True)
                if self.adversarial_module.delta.grad is not None:
                    self.adversarial_module.delta.grad = None
                
                with torch.no_grad():
                    delta_before = self.adversarial_module.delta.data.clone()
                # Determine patch positions using image_id if available
                if "image_id" in inputs:
                    image_ids = inputs["image_id"]
                    patch_positions = []
                    for id in image_ids:
                        idx = int(id) % 3
                        if idx == 0:
                            patch_positions.append([1085, 50])
                        elif idx == 1:
                            patch_positions.append([730, 50])
                        elif idx == 2:
                            patch_positions.append([375, 50])
                    patch_boxes = torch.tensor([[x, y, 295, 323] for x, y in patch_positions], device=raw_images.device)
                else:
                    # fallback: single patch at (1074, 50)
                    print("No image_id found, using single patch at (1074, 50)")
                    B = raw_images.shape[0]
                    patch_boxes = torch.tensor([[1085, 50, 295, 323]] * B, device=raw_images.device)
                processed = self.adversarial_module(raw_images, patch_boxes)
                step_inputs = {
                    "input_ids": base_text_inputs["input_ids"].detach().clone(),
                    "labels": base_text_inputs["labels"].detach().clone(),
                    "attention_mask": base_text_inputs["attention_mask"].detach().clone() if base_text_inputs["attention_mask"] is not None else None,
                    "pixel_values": processed.pixel_values,
                    "image_sizes": processed.image_sizes,
                }

                model.train()
                batch_inputs = self._prepare_inputs(step_inputs)
                loss = self.compute_loss(model, batch_inputs)

                loss.backward()
                with torch.no_grad():
                    delta_grad = self.adversarial_module.delta.grad
                    if delta_grad is not None:
                        delta = self.adversarial_module.delta.data
                        step_tensor = self.pgd_alpha * delta_grad.sign()
                        delta = delta - step_tensor
                        
                        delta = torch.clamp(delta, -self.pgd_epsilon, self.pgd_epsilon)
                        self.adversarial_module.delta.data.copy_(delta)
                    self.adversarial_module.delta.grad = None

            # Final forward pass
            # Determine patch positions using image_id if available
            if "image_id" in inputs:
                image_ids = inputs["image_id"]
                patch_positions = []
                for id in image_ids:
                    idx = int(id) % 3
                    if idx == 0:
                        patch_positions.append([1085, 50])
                    elif idx == 1:
                        patch_positions.append([730, 50])
                    elif idx == 2:
                        patch_positions.append([375, 50])
                patch_boxes = torch.tensor([[x, y, 295, 323] for x, y in patch_positions], device=raw_images.device)
            else:
                # fallback: single patch at (1085, 50)
                print("No image_id found, using single patch at (1085, 50)")
                B = raw_images.shape[0]
                patch_boxes = torch.tensor([[1085, 50, 295, 323]] * B, device=raw_images.device)
            processed = self.adversarial_module(raw_images, patch_boxes)
            final_inputs = {
                "input_ids": base_text_inputs["input_ids"].detach().clone(),
                "labels": base_text_inputs["labels"].detach().clone(),
                "attention_mask": base_text_inputs["attention_mask"].detach().clone() if base_text_inputs["attention_mask"] is not None else None,
                "pixel_values": processed.pixel_values,
                "image_sizes": processed.image_sizes,
            }
            model.train()
            batch_inputs = self._prepare_inputs(final_inputs)
            loss = self.compute_loss(model, batch_inputs)
            
            return loss.detach()
            
        except Exception as e:
            print(f"\nError in training step: {str(e)}")
            import traceback
            traceback.print_exc()
            raise e

    def compute_loss(self, model, inputs, return_outputs: bool = False):
        """
        Compute loss with proper causal LM objective (logit shifting).
        Following the same pattern as Qwen for consistency.
        """
        try:
            # Extract labels if present
            if "labels" in inputs:
                labels = inputs.pop("labels")
            else:
                labels = None
            
            # Forward pass through model
            outputs = model(**inputs)
            
            if labels is not None:
                # Causal LM objective: each token predicts the NEXT token.
                # Shift logits and labels so that tokens <t> predict label <t+1>.
                logits = outputs.logits  # (B, L, V)
                vocab_size = logits.size(-1)

                # Remove last logit and first label
                shift_logits = logits[:, :-1, :].contiguous()
                shift_labels = labels[:, 1:].contiguous().to(logits.device)

                loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
                loss = loss_fct(shift_logits.view(-1, vocab_size),
                                shift_labels.view(-1))
                                
            else:
                # Fallback to model's internal loss if no labels
                if isinstance(outputs, dict) and "loss" in outputs:
                    loss = outputs["loss"]
                elif hasattr(outputs, 'loss'):
                    loss = outputs.loss
                else:
                    raise ValueError(
                        "The model did not return a loss from the inputs, and no labels were provided. "
                        f"Model outputs: {list(outputs.keys()) if isinstance(outputs, dict) else type(outputs)}"
                    )
            
            return (loss, outputs) if return_outputs else loss
            
        except Exception as e:
            print(f"[Trainer] Forward error: {e}")
            raise


class ConversationDataset(Dataset):
    """Dataset for adversarial training with conversations from a JSON file."""
    def __init__(self, data_path, processor, device, system_prompt):
        """
        Args:
            data_path: Path to conversation JSON file containing multiple samples
            processor: Phi-3-vision processor
            device: torch device
        """
        self.processor = processor
        self.device = device
        self.system_prompt = system_prompt
        
        # Load dataset
        with open(data_path, "r", encoding="utf-8") as f:
            self.samples = json.load(f)

    def __len__(self):
        return len(self.samples)
        
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image if present
        image = None
        if 'image' in sample:
            image_path = sample['image']
            if not os.path.isabs(image_path):
                base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
                image_path = os.path.join(base_dir, image_path)
                
            # Load and convert image to tensor for adversarial perturbation
            image = Image.open(image_path).convert('RGB')
            image_tensor = torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0
        
        # Extract conversation components
        conversations = sample["conversations"]
        
        # Map conversation roles if needed
        role_map = {"human": "user", "gpt": "assistant"}
        chat = []
        for msg in conversations:
            if "role" in msg and "content" in msg:
                chat.append(msg)
            elif "from" in msg and "value" in msg:
                role = role_map.get(msg["from"].lower(), msg["from"].lower())
                chat.append({"role": role, "content": msg["value"]})
            else:
                raise ValueError("Conversation item must have (role,content) or (from,value) keys")
        
        # Add system prompt like Qwen does
        if self.system_prompt:
            if chat and chat[0]["role"] == "system":
                chat[0]["content"] = self.system_prompt  # replace
            else:
                chat.insert(0, {"role": "system", "content": self.system_prompt})
        
        # Find the last assistant message
        last_assistant_idx = None
        for i in range(len(chat) - 1, -1, -1):
            if chat[i]["role"] == "assistant":
                last_assistant_idx = i
                break
                
        if last_assistant_idx is None:
            raise ValueError("No assistant message found in conversation")
            
        # Extract the target text (what we want the model to generate)
        target_text = chat[last_assistant_idx]["content"]
        
        # Build prompt using processor's chat template
        chat_input = chat.copy()
        # Keep the full conversation - model sees everything, we just mask labels
        
        # Apply chat template to get the prompt
        prompt = self.processor.tokenizer.apply_chat_template(chat_input, tokenize=False, add_generation_prompt=False)
        if prompt.endswith('<|endoftext|>'):
            prompt = prompt.rstrip('<|endoftext|>')
        
        # Process prompt and image together to get input_ids
        inputs = self.processor(prompt, images=[image] if image is not None else None, return_tensors='pt')
        
        # Get target ids - IMPORTANT: don't add special tokens since they're already in conversation
        target_ids = self.processor.tokenizer(target_text, return_tensors="pt", add_special_tokens=False).input_ids[0]
        
        # Create labels: mask everything except target
        input_ids = inputs.input_ids[0]
        labels = torch.full_like(input_ids, _IGNORE_INDEX)
        
        # Find where target sequence appears in input for proper label alignment
        input_list = input_ids.tolist()
        target_list = target_ids.tolist()
        
        # Find where target sequence appears in input
        target_start_pos = None
        for i in range(len(input_list) - len(target_list) + 1):
            if input_list[i:i+len(target_list)] == target_list:
                target_start_pos = i
                break
        
        # CRITICAL: Place labels exactly where target appears in input for proper alignment
        if target_start_pos is not None:
            labels[target_start_pos:target_start_pos+len(target_list)] = target_ids
        else:
            # Fallback: place at end (but this may cause alignment issues)
            if target_ids.numel() <= labels.numel():
                labels[-target_ids.numel():] = target_ids
            else:
                available_space = labels.numel()
                labels[-available_space:] = target_ids[-available_space:]

        # Basic logging for verification
        if idx < 1:  # Only first sample
            print(f"[Dataset] Target found at position: {target_start_pos}")
            print(f"[Dataset] Input length: {len(input_ids)}, Target length: {len(target_ids)}, Labeled tokens: {(labels != _IGNORE_INDEX).sum()}")
            
            if target_start_pos is None:
                print("⚠️  [Dataset] WARNING: Target sequence not found in input!")
            else:
                print("✅ [Dataset] Target sequence successfully aligned")
        
        return {
            "input_ids": inputs.input_ids[0],  # Remove batch dim
            "labels": labels,
            "raw_image": image_tensor,  # Keep raw image for perturbation
            "attention_mask": inputs.attention_mask[0],
            "pixel_values": inputs.pixel_values[0] if hasattr(inputs, "pixel_values") else None,
            "image_sizes": inputs.image_sizes[0] if hasattr(inputs, "image_sizes") else None,
            "image_id": int(sample['id']) if 'id' in sample else None,
        }


def pad_sequence(sequences, padding_side='right', padding_value=0):
    """
    Pad a list of sequences to the same length.
    sequences: list of tensors in [seq_len, *] shape
    """
    assert padding_side in ['right', 'left']
    max_size = sequences[0].size()
    trailing_dims = max_size[1:]
    max_len = max(len(seq) for seq in sequences)
    batch_size = len(sequences)
    output = sequences[0].new_full((batch_size, max_len) + trailing_dims, padding_value)
    for i, seq in enumerate(sequences):
        length = seq.size(0)
        if padding_side == 'right':
            output.data[i, :length] = seq
        else:
            output.data[i, -length:] = seq
    return output

class DataCollatorForAdversarialDataset:
    """Collate examples for adversarial training with Phi-3-vision."""
    
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.pad_token_id = tokenizer.pad_token_id
        
    def __call__(self, instances):
        # Extract and pad input_ids, labels, and attention masks
        batch_input_ids = []
        batch_label_ids = []
        batch_pixel_values = []
        batch_image_sizes = []
        batch_raw_images = []
        batch_image_ids = []
        
        for instance in instances:
            batch_input_ids.append(instance["input_ids"])
            batch_label_ids.append(instance["labels"])
            batch_pixel_values.append(instance["pixel_values"])
            batch_image_sizes.append(instance["image_sizes"])
            batch_raw_images.append(instance["raw_image"])
            batch_image_ids.append(instance["image_id"])
        # Pad sequences - right padding for training
        input_ids = pad_sequence(
            batch_input_ids, padding_side='right', padding_value=self.pad_token_id
        )
        attention_mask = input_ids != self.pad_token_id
        labels = pad_sequence(
            batch_label_ids, padding_side='right', padding_value=_IGNORE_INDEX
        )
        
        # Use torch.cat for pixel values and image sizes (matching the official implementation)
        pixel_values = torch.cat(batch_pixel_values, dim=0)
        image_sizes = torch.cat(batch_image_sizes, dim=0)
        image_ids = torch.tensor(batch_image_ids, dtype=torch.long)
        
        # Stack raw images for adversarial perturbation 
        # (this is specifically for your adversarial delta calculations)
        raw_images = torch.stack(batch_raw_images)
        
        return {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
            "pixel_values": pixel_values, 
            "image_sizes": image_sizes,
            "raw_image": raw_images,
            "image_id": image_ids
        }           

def get_unwrapped_model(model):
    """Helper function to get the base model from possible wrappers like DistributedDataParallel."""
    if hasattr(model, 'module'):
        return model.module
    return model

class InferenceCallback(TrainerCallback):
    """Callback to run minimal logging after each epoch."""
    
    def __init__(self, tokenizer, system_prompt, user_request, grounding_prompt, assistant_answer, device, base_img, image_processor, base_img_pil):
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt
        self.user_request = user_request
        self.grounding_prompt = grounding_prompt
        self.assistant_answer = assistant_answer
        self.device = device
        self.base_img = base_img
        self.image_processor = image_processor
        self.base_img_pil = base_img_pil
        
    def on_epoch_end(self, args, state, control, model=None, **kwargs):
        if hasattr(args, 'local_rank') and args.local_rank not in [0, -1]:
            return
        print("\n[CALLBACK] Running adversarial inference at epoch end...")
        
        try:
            # Get base model from DDP wrapper
            unwrapped_model = get_unwrapped_model(model)
            unwrapped_model.eval()
            
            with torch.no_grad():
                if hasattr(unwrapped_model, 'adversarial') and hasattr(unwrapped_model.adversarial, 'delta'):
                    print("Applying adversarial perturbation...")
                    # Apply adversarial perturbation to base image
                    perturbed = self.base_img.squeeze(0).clone()
                    perturbed[:, 50:345, 1085:1408] += unwrapped_model.adversarial.delta.detach().to(self.base_img.device, dtype=self.base_img.dtype).squeeze(0)
                    adv_img_tensor = perturbed.clone().cpu().squeeze(0).clamp(0, 1)
                    adv_img_pil = to_pil_image(adv_img_tensor)

                    chat = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"<|image_1|>\n{self.user_request}"},
                        {"role": "assistant", "content": self.assistant_answer},
                        {"role": "user", "content": self.grounding_prompt}
                    ]
                    
                    # Build prompt using processor's chat template
                    prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    if prompt.endswith('<|endoftext|>'):
                        prompt = prompt.rstrip('<|endoftext|>')
                    
                    print("Processing inputs...")
                    # Prepare inputs and run inference
                    inputs = self.image_processor(prompt, [adv_img_pil], return_tensors='pt').to(self.device)
                    
                    print("Generating output...")
                    try:
                        # Generate output
                        generate_ids = unwrapped_model.generate(
                            **inputs,
                            max_new_tokens=300,
                            do_sample=False,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        
                        # Decode generated text
                        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
                        response = self.tokenizer.batch_decode(
                            generate_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        
                        print(f"\nEpoch {state.epoch} inference result:")
                        print(f'>>> Response\n{response}')
                    except Exception as e:
                        print(f"Error during generation: {str(e)}")
                        print(f"Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[CALLBACK] No adversarial delta found on model.")
                    return
        except Exception as e:
            print(f"Error in inference callback: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()

    def on_epoch_end2(self, args, state, control, model=None, **kwargs):
        if hasattr(args, 'local_rank') and args.local_rank not in [0, -1]:
            return
        print("\n[CALLBACK] Running direct adversarial inference at epoch end...")
        
        try:
            # Get base model from DDP wrapper
            unwrapped_model = get_unwrapped_model(model)
            unwrapped_model.eval()
            
            with torch.no_grad():
                if hasattr(unwrapped_model, 'adversarial') and hasattr(unwrapped_model.adversarial, 'delta'):
                    # Apply adversarial perturbation to base image
                    delta = unwrapped_model.adversarial.delta.detach().to(self.base_img.device)
                    
                    # Generate perturbed image
                    perturbed_img = self.base_img + delta * mask
                    perturbed_img = torch.clamp(perturbed_img, 0, 1)
                    
                    # Process image directly using differentiable pipeline
                    processed = unwrapped_model.adversarial.transform(perturbed_img)
                    chat = [
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": f"<|image_1|>\n{self.user_request}"},
                        {"role": "assistant", "content": self.assistant_answer},
                        {"role": "user", "content": self.grounding_prompt}
                    ]
                    
                    # Build prompt using processor's chat template
                    prompt = self.tokenizer.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
                    if prompt.endswith('<|endoftext|>'):
                        prompt = prompt.rstrip('<|endoftext|>')
                    
                    inputs = self.image_processor(prompt, images=[self.base_img_pil], return_tensors='pt').to(self.device)
                    
                    # Combine text and processed image inputs
                    model_inputs = {
                        "input_ids": inputs.input_ids,  # Clone to avoid in-place modifications
                        "attention_mask": inputs.attention_mask,
                        "pixel_values": processed.pixel_values,
                        "image_sizes": processed.image_sizes
                    }

                    try:
                        # Generate output
                        generate_ids = unwrapped_model.generate(
                            **model_inputs,
                            max_new_tokens=300,
                            do_sample=False,
                            eos_token_id=self.tokenizer.eos_token_id,
                        )
                        # Decode generated text
                        generate_ids = generate_ids[:, inputs.input_ids.shape[1]:]
                        response = self.tokenizer.batch_decode(
                            generate_ids,
                            skip_special_tokens=True,
                            clean_up_tokenization_spaces=False
                        )[0]
                        print(f"\nEpoch {state.epoch} direct inference result:")
                        print(f'>>> Direct Response\n{response}')
                        
                    except Exception as e:
                        print(f"Error during direct generation: {str(e)}")
                        print(f"Error type: {type(e)}")
                        import traceback
                        traceback.print_exc()
                else:
                    print("[CALLBACK] No adversarial delta found on model.")
                    return
        except Exception as e:
            print(f"Error in direct inference callback: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            traceback.print_exc()

def train(args):
    """Main training function."""
    # Initialize distributed training if needed
    if args.distributed:
        init_process_group(backend="nccl")
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        world_size = torch.distributed.get_world_size()
    else:
        local_rank = 0
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        world_size = 1
    
    # Set random seed
    set_seed(17)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load processor and model
    processor = AutoProcessor.from_pretrained(args.model_path, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map={"": device},
        torch_dtype=torch.bfloat16,
        trust_remote_code=True
    )

    # Create training arguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.num_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        evaluation_strategy="no",  # No evaluation since we use custom callback
        save_strategy="no",
        logging_steps=1,
        gradient_accumulation_steps=1,
        bf16=True,  # Use bfloat16 precision
        local_rank=local_rank,
        dataloader_drop_last=False,  # Don't drop incomplete batches
        remove_unused_columns=False,  # Keep all columns from dataset
        gradient_checkpointing=False,  # We've already enabled it on the model
    )

    # Enable gradient checkpointing for memory efficiency
    model.config.use_cache = False  # Required for gradient checkpointing
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    else:
        def make_inputs_require_grad(module, input, output):
            output.requires_grad_(True)
        model.get_input_embeddings().register_forward_hook(make_inputs_require_grad)

    # First freeze all base model parameters
    for param in model.parameters():
        param.requires_grad = False

    # Add LoRA adapters for memory efficiency
    from peft import LoraConfig, get_peft_model

    def find_all_linear_names(model):
        # Only patch the first linear layer found
        for name, module in model.named_modules():
            if isinstance(module, torch.nn.Linear):
                return [name]
        return []

        if 'lm_head' in lora_module_names:  # needed for 16-bit
            lora_module_names.remove('lm_head')
        return list(lora_module_names)

    # Create and apply LoRA config
    lora_config = LoraConfig(
        r=64,  # LoRA attention dimension
        lora_alpha=16,  # LoRA alpha scaling
        target_modules=find_all_linear_names(model),
        lora_dropout=0.05,  # LoRA dropout
        bias="none",  # LoRA bias type
        task_type="CAUSAL_LM",
    )
    print("Adding LoRA adapters...")
    model = get_peft_model(model, lora_config)

    # Freeze LoRA parameters since we only want to train the adversarial delta
    for name, param in model.named_parameters():
        if "lora_" in name:
            param.requires_grad = False

    # Load conversation data
    with open(args.test_data_path, "r", encoding="utf-8") as f:
        conv_data = json.load(f)
    system_prompt = conv_data["system"]
    user_request = conv_data["user"][0]
    grounding_prompt = conv_data["user"][1]
    assistant_answer = conv_data["assistant"]

    # Load base image and convert to tensor
    base_img_pil_1 = Image.open(args.image_path_1).convert("RGB")
    base_img_tensor_1 = torch.from_numpy(np.array(base_img_pil_1)).permute(2, 0, 1).float() / 255.0
    base_img_tensor_1 = base_img_tensor_1.to(device)
    base_img_tensor_1 = base_img_tensor_1.unsqueeze(0)  # Add batch dimension
    
    base_img_pil_2 = Image.open(args.image_path_2).convert("RGB")
    base_img_tensor_2 = torch.from_numpy(np.array(base_img_pil_2)).permute(2, 0, 1).float() / 255.0
    base_img_tensor_2 = base_img_tensor_2.to(device)
    base_img_tensor_2 = base_img_tensor_2.unsqueeze(0)

    base_img_pil_3 = Image.open(args.image_path_3).convert("RGB")
    base_img_tensor_3 = torch.from_numpy(np.array(base_img_pil_3)).permute(2, 0, 1).float() / 255.0
    base_img_tensor_3 = base_img_tensor_3.to(device)
    base_img_tensor_3 = base_img_tensor_3.unsqueeze(0)
     
    image_shape = base_img_tensor_1.shape

    # Create and add adversarial perturbation module
    adversarial_module = AdversarialPerturbation(
        shape=image_shape,
        device=device,
        processor=processor
    )

    # Add module to base model (before DDP wrapping)
    model.add_module('adversarial', adversarial_module)

    # Ensure only delta requires grad
    model.adversarial.delta.requires_grad_(True)

    # Verify trainable parameters
    trainable_params = [name for name, p in model.named_parameters() if p.requires_grad]
    if training_args.local_rank == 0:
        print("Trainable parameters:", trainable_params)
        # Should only show adversarial.delta
    
    # Load dataset
    dataset = ConversationDataset(
        data_path=args.train_data_path,
        processor=processor,
        device=device,
        system_prompt=system_prompt,
    )
    
    # Wrap in DDP after all model modifications
    if args.distributed:
        model = DistributedDataParallel(model, device_ids=[local_rank])
    
    # Create trainer with our custom data collator
    trainer = AdversarialTrainer(
        pgd_steps=5,
        pgd_alpha=1/255,
        pgd_epsilon=16/255,
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=DataCollatorForAdversarialDataset(processor.tokenizer),
    )

    # Add inference callback
    trainer.add_callback(InferenceCallback(
        tokenizer=processor.tokenizer,
        system_prompt=system_prompt,
        user_request=user_request,
        grounding_prompt=grounding_prompt,
        assistant_answer=assistant_answer,
        device=device,
        base_img=base_img_tensor_1,
        image_processor=processor,
        base_img_pil=base_img_pil_1,
    ))
    
    # Train
    if list(Path(training_args.output_dir).glob("checkpoint-*")):
        trainer.train(resume_from_checkpoint=True)
    else:
        print("Starting training...")
        trainer.train()
        print("Training completed")
    
    # For safety, unwrap model before touching attributes when DDP is enabled
    unwrapped_model = get_unwrapped_model(model)
    unwrapped_model.config.use_cache = False
    
    if local_rank == 0:
        try:
            with torch.no_grad():
                delta = get_unwrapped_model(model).adversarial.delta.data.to(device)
                perturbed_1 = base_img_tensor_1.squeeze(0).clone()
                perturbed_1[:, 50:345, 1085:1408] += delta.squeeze(0)
                adv_img_tensor_1 = perturbed_1.clone().cpu().squeeze(0).clamp(0, 1)
                pil_img_1 = to_pil_image(adv_img_tensor_1.squeeze(0).detach().cpu())
                image_path_1 = os.path.join(args.output_dir, f"perturbed_sample_1.png")
                pil_img_1.save(image_path_1)
                print(f"Saved perturbed image to {image_path_1}")

                patch_x, patch_y = 1085, 50
                patch_w, patch_h = 323, 295
                adversarial_card_1 = pil_img_1.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
                card_path_1 = os.path.join(args.output_dir, 'adversarial_product_1.png')
                adversarial_card_1.save(card_path_1)
                print(f"[INFO] Adversarial product (patch) saved to {card_path_1}")

                patch_1 = Image.open(card_path_1).convert('RGB')

                perturbed_2 = base_img_tensor_2.squeeze(0).clone()
                perturbed_2[:, 50:345, 730:1053] += delta.squeeze(0)
                adv_img_tensor_2 = perturbed_2.clone().cpu().squeeze(0).clamp(0, 1)
                pil_img_2 = to_pil_image(adv_img_tensor_2.squeeze(0).detach().cpu())
                image_path_2 = os.path.join(args.output_dir, f"perturbed_sample_2.png")
                pil_img_2.save(image_path_2)
                print(f"Saved perturbed image to {image_path_2}")

                patch_x, patch_y = 730, 50
                patch_w, patch_h = 323, 295
                adversarial_card_2 = pil_img_2.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
                card_path_2 = os.path.join(args.output_dir, 'adversarial_product_2.png')
                adversarial_card_2.save(card_path_2)
                print(f"[INFO] Adversarial product (patch) saved to {card_path_2}")

                patch_2 = Image.open(card_path_2).convert('RGB')

                perturbed_3 = base_img_tensor_3.squeeze(0).clone()
                perturbed_3[:, 50:345, 375:698] += delta.squeeze(0)
                adv_img_tensor_3 = perturbed_3.clone().cpu().squeeze(0).clamp(0, 1)
                pil_img_3 = to_pil_image(adv_img_tensor_3.squeeze(0).detach().cpu())
                image_path_3 = os.path.join(args.output_dir, f"perturbed_sample_3.png")
                pil_img_3.save(image_path_3)
                print(f"Saved perturbed image to {image_path_3}")

                patch_x, patch_y = 375, 50
                patch_w, patch_h = 323, 295
                adversarial_card_3 = pil_img_3.crop((patch_x, patch_y, patch_x + patch_w, patch_y + patch_h))
                card_path_3 = os.path.join(args.output_dir, 'adversarial_product_3.png')
                adversarial_card_3.save(card_path_3)
                print(f"[INFO] Adversarial product (patch) saved to {card_path_3}")

                patch_3 = Image.open(card_path_3).convert('RGB')
                
                # Apply adversarial patch to test samples
                test_dirs = [
                    ('test_samples_1_unseen', 40),
                    ('test_samples_2_unseen', 40), 
                    ('test_samples_3_unseen', 40),
                    ('train_samples', 30)
                ]
                
                for dir_name, num_samples in test_dirs:
                    test_samples_dir = os.path.join(args.output_dir, dir_name)
                    os.makedirs(test_samples_dir, exist_ok=True)
                    
                    applied_count = 0
                    for i in range(num_samples):
                        sample_name = f'sample_{i}.png'
                        sample_path = os.path.join(test_samples_dir, sample_name)
                        if not os.path.exists(sample_path):
                            print(f"[WARNING] {sample_path} does not exist, skipping.")
                            continue
                        
                        try:
                            sample_img = Image.open(sample_path).convert('RGB')
                            # sample_img_tensor = torch.from_numpy(np.array(sample_img)).permute(2, 0, 1).float() / 255.0
                            # sample_img_tensor = sample_img_tensor.to(device)
                            idx = i % 3
                            if idx == 0:
                                x, y = 1085, 50
                                sample_img.paste(patch_1, (x, y))
                            elif idx == 1:
                                x, y = 730, 50
                                sample_img.paste(patch_2, (x, y))
                            elif idx == 2:
                                x, y = 375, 50
                                sample_img.paste(patch_3, (x, y))
                            sample_img.save(sample_path)
                            applied_count += 1

                        except Exception as e:
                            print(f"[WARNING] Failed to process {sample_path}: {e}")
                    
                    print(f"[INFO] Overwrote adversarial patch to {applied_count}/{num_samples} samples in {dir_name}")
            
        except Exception as e:
            print(f"[WARN] Failed to save adversarial image: {e}")
            import traceback
            print(f"[WARN] Full traceback: {traceback.format_exc()}")

    # Clean up distributed training
    if args.distributed:
        destroy_process_group()

def parse_args():
    parser = argparse.ArgumentParser(description="PGD Adversarial Training for Phi-4")
    
    # Model and data paths
    parser.add_argument("--model_path", type=str, default="/home/wanjing/Desktop/lab/Phi-4-multimodal-instruct", 
                        help="Path to Phi-4 model")
    parser.add_argument("--train_data_path", type=str, required=True,
                        help="Path to conversation JSON file")
    parser.add_argument("--test_data_path", type=str, required=True,
                        help="Path to conversation JSON file")
    parser.add_argument("--output_dir", type=str, default="./output",
                        help="Output directory")
    parser.add_argument("--image_path_1", type=str, default=None,
                        help="Path to image file")
    parser.add_argument("--image_path_2", type=str, default=None,
                        help="Path to image file")
    parser.add_argument("--image_path_3", type=str, default=None,
                        help="Path to image file")
    
    # Training parameters
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=1,
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for optimizer")
    parser.add_argument("--eval_interval", type=int, default=5,
                        help="Interval for evaluation and saving checkpoints")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    # Distributed training
    parser.add_argument("--distributed", action="store_true",
                        help="Enable distributed training")
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    train(args) 