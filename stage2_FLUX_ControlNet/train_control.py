#!/usr/bin/env python
# coding=utf-8
# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and

import argparse
import copy
import logging
import math
import os
import random
import shutil
from contextlib import nullcontext
from pathlib import Path

import accelerate
import numpy as np
import torch
import transformers
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedType, ProjectConfiguration, set_seed
from datasets import load_dataset
from huggingface_hub import create_repo, upload_folder
from packaging import version
from peft import LoraConfig, set_peft_model_state_dict
from peft.utils import get_peft_model_state_dict
from PIL import Image
from torchvision import transforms
from tqdm.auto import tqdm
import safetensors

import diffusers
from diffusers import AutoencoderKL, FlowMatchEulerDiscreteScheduler, FluxControlPipeline
from flux_transformer import FluxTransformer2DModel
from diffusers.optimization import get_scheduler
from diffusers.training_utils import (
    cast_training_params,
    compute_density_for_timestep_sampling,
    compute_loss_weighting_for_sd3,
    free_memory,
)
from diffusers.utils import check_min_version, is_wandb_available, load_image, make_image_grid
from diffusers.utils.hub_utils import load_or_create_model_card, populate_model_card
from diffusers.utils.torch_utils import is_compiled_module

from pipeline import FluxControlPipeline
# from processor import FluxRDIMGAttnProcessor2_0_NPU
from utils import get_all_processor_keys
import torch.nn.functional as F


if is_wandb_available():
    import wandb

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.34.0.dev0")

logger = get_logger(__name__)

NORM_LAYER_PREFIXES = ["norm_q", "norm_k", "norm_added_q", "norm_added_k"]


from mmcv.ops import point_sample
from third_parts.mmdet.models.utils import get_uncertain_point_coords_with_randomness
from third_parts.mmdet.models.losses import DiceLoss, CrossEntropyLoss

def sample_points(mask_pred, gt_masks):
    oversample_ratio = 3.0
    num_points=1024
    importance_sample_ratio = 0.75
    gt_masks = gt_masks.unsqueeze(1)
    gt_masks = gt_masks.to(mask_pred)
    mask_pred = mask_pred.unsqueeze(1)
    
    # (N, 1, h, w)

    with torch.no_grad():
        points_coords = get_uncertain_point_coords_with_randomness(
            mask_pred.to(torch.float32), None, num_points,
            oversample_ratio, importance_sample_ratio)
        # shape (num_total_gts, h, w) -> (num_total_gts, num_points)
        mask_point_targets = point_sample(
            gt_masks.float(), points_coords).squeeze(1)
    # shape (num_queries, h, w) -> (num_queries, num_points)
    mask_point_preds = point_sample(
        mask_pred.to(torch.float32), points_coords.to(torch.float32)).squeeze(1)
    return mask_point_preds.to(mask_pred.dtype), mask_point_targets.to(mask_pred.dtype)


from pycocotools import mask as mask_utils
def rle_to_pil_image(rle_data, size=(1024, 1024)):
    """Decode RLE data into a PIL image."""
    rle = {
        'counts': rle_data['counts'].encode('utf-8') if isinstance(rle_data['counts'], str) else rle_data['counts'],
        'size': rle_data['size']
    }
    mask_array = mask_utils.decode(rle) * 255  # Convert to the 0-255 range.
    image = Image.fromarray(mask_array.astype(np.uint8))
    image = image.resize(size,Image.BILINEAR)
    return image


def calculate_iou(mask1, mask2):
    """Compute the IoU between two masks."""
    # Convert masks to bool for bitwise operations.
    mask1_bool = mask1 > 0.5
    mask2_bool = mask2 > 0.5
    
    # Compute intersection and union.
    intersection = (mask1_bool & mask2_bool).float().sum()
    union = (mask1_bool | mask2_bool).float().sum()
    
    return intersection / (union + 1e-6)


def encode_images(pixels: torch.Tensor, vae: torch.nn.Module, weight_dtype):
    pixel_latents = vae.encode(pixels.to(vae.dtype)).latent_dist.sample()
    pixel_latents = (pixel_latents - vae.config.shift_factor) * vae.config.scaling_factor
    return pixel_latents.to(weight_dtype)


def log_validation(flux_transformer, args, accelerator, weight_dtype, step, is_final_validation=False):
    logger.info("Running validation... ")

    if not is_final_validation:
        flux_transformer = accelerator.unwrap_model(flux_transformer)
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=flux_transformer,
            torch_dtype=weight_dtype,
        )
    else:
        transformer = FluxTransformer2DModel.from_pretrained(
            args.pretrained_model_name_or_path, subfolder="transformer", torch_dtype=weight_dtype
        )
        initial_channels = transformer.config.in_channels
        pipeline = FluxControlPipeline.from_pretrained(
            args.pretrained_model_name_or_path,
            transformer=transformer,
            torch_dtype=weight_dtype,
        )
        pipeline.load_lora_weights(args.output_dir)
        assert pipeline.transformer.config.in_channels == initial_channels * 2, (
            f"{pipeline.transformer.config.in_channels=}"
        )

    pipeline.to(accelerator.device)
    pipeline.set_progress_bar_config(disable=True)

    if args.seed is None:
        generator = None
    else:
        print(args.seed)
        print("=======")
        generator = torch.Generator(device=accelerator.device).manual_seed(args.seed)

    if len(args.validation_image) == len(args.validation_prompt):
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt
    elif len(args.validation_image) == 1:
        validation_images = args.validation_image * len(args.validation_prompt)
        validation_prompts = args.validation_prompt
    elif len(args.validation_prompt) == 1:
        validation_images = args.validation_image
        validation_prompts = args.validation_prompt * len(args.validation_image)
    else:
        raise ValueError(
            "number of `args.validation_image` and `args.validation_prompt` should be checked in `parse_args`"
        )

    image_logs = []
    if is_final_validation or torch.backends.mps.is_available():
        autocast_ctx = nullcontext()
    else:
        autocast_ctx = torch.autocast(accelerator.device.type, weight_dtype)

    from torchvision import transforms
    mask_image_transforms= transforms.Compose(
        [
            transforms.Resize((1024, 1024), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )

    for validation_prompt, validation_image in zip(validation_prompts, validation_images):
        validation_image = load_image(validation_image)
        # maybe need to inference on 1024 to get a good image
        validation_image = validation_image.resize((args.resolution, args.resolution))

        import json
        with open(validation_prompt, 'r', encoding='utf-8') as f:
            content = f.read()
        data = json.loads(content)
        
        part_prompt =[]
        mask_list = []
        indices_list = []
        for obj in data.get("object_list", []):
            
            # mask_path = os.path.join('/home/five/lisa_test_add',obj["mask"])
            mask_path = os.path.join('/home/five/lisa_test',obj["mask"])
            mask_img = Image.open(mask_path).convert("L")
            mask_tensor = mask_image_transforms(mask_img)  # Optional: customize mask_transforms.
            mask_list.append(mask_tensor)
            part_prompt.append(obj["prompt"])
            indices_list.append(obj["indices"])


        images = []

        for _ in range(args.num_validation_images):
            with autocast_ctx:
                image = pipeline(
                    prompt=[data["long_prompt"]],
                    control_image=validation_image,
                    num_inference_steps=50,
                    guidance_scale=30,
                    generator=generator,
                    max_sequence_length=512,
                    height=args.resolution,
                    width=args.resolution,
                    mask_list =torch.stack([torch.stack(mask_list)]),
                    part_prompt = [part_prompt],
                    indices_list = [indices_list]
                ).images[0]
            image = image.resize((args.resolution, args.resolution))
            images.append(image)
        image_logs.append(
            {"validation_image": validation_image, "images": images, "validation_prompt": validation_prompt}
        )

    tracker_key = "test" if is_final_validation else "validation"
    for tracker in accelerator.trackers:
        if tracker.name == "tensorboard":
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images = []
                formatted_images.append(np.asarray(validation_image))
                for image in images:
                    formatted_images.append(np.asarray(image))
                formatted_images = np.stack(formatted_images)
                tracker.writer.add_images(validation_prompt, formatted_images, step, dataformats="NHWC")

        elif tracker.name == "wandb":
            formatted_images = []
            for log in image_logs:
                images = log["images"]
                validation_prompt = log["validation_prompt"]
                validation_image = log["validation_image"]
                formatted_images.append(wandb.Image(validation_image, caption="Conditioning"))
                for image in images:
                    image = wandb.Image(image, caption=validation_prompt)
                    formatted_images.append(image)

            tracker.log({tracker_key: formatted_images})
        else:
            logger.warning(f"image logging not implemented for {tracker.name}")

        del pipeline
        free_memory()
        return image_logs
    


def compute_iou(pred_mask, gt_mask, epsilon=1e-6):
    """
    Compute batched IoU.
    
    Args:
    - pred_mask (torch.Tensor): Predicted mask with shape (B, N, 1, H, W).
    - gt_mask (torch.Tensor): GT mask with shape (B, N, 1, H, W).

    Returns:
    - iou (torch.Tensor): IoU score with shape (B, N).
    """
    # Ensure float dtype.
    pred_mask = pred_mask.float()
    gt_mask = gt_mask.float()
    
    # Sum along the H and W dimensions (dim=[3, 4]).
    intersection = (pred_mask * gt_mask).sum(dim=[3, 4]) # Shape: (B, N, 1).
    
    sum_masks = pred_mask.sum(dim=[3, 4]) + gt_mask.sum(dim=[3, 4]) # Shape: (B, N, 1).
    union = sum_masks - intersection # Shape: (B, N, 1).
    
    iou = (intersection + epsilon) / (union + epsilon) # Shape: (B, N, 1).
    
    # Remove the extra dimension 1 and return (B, N).
    return iou.squeeze(-1)

def compute_iou_weight(position_mask, position_mask_predict):
    """
    Compute attention_weight with shape (B, 1, H, W) from masks with shape (B, N, ...).
    
    Logic:
    1. Binarize the GT and predicted masks.
    2. Compute the IoU of N objects with shape (B, N).
    3. Set values with IoU > 0.6 to 1.0.
    4. Find the relevant region for each object n, defined as the union of GT and predicted masks.
    5. Create a weight map with shape (B, N, 1, H, W):
       - inside the relevant region, the value is iou[n]
       - outside the region, the value is 1.0
    6. Multiply all weights along the N dimension with torch.prod to obtain the final (B, 1, H, W) weight.
    """
    B, N, _, H, W = position_mask.shape
    device = position_mask.device
    
    # Step 1: Binarize.
    # Create copies so the original inputs are not modified.
    

    position_mask_gt = (position_mask > 0.5).float()
    position_mask_pred = (position_mask_predict > 0.5).float()

    # Step 2: Compute IoU.
    # iou shape: (B, N).
    iou = compute_iou(position_mask_pred, position_mask_gt)
    
    # Step 3: Threshold IoU.
    # .detach() ensures this weight computation does not backpropagate to the mask predictor.
    iou = iou.detach() 
    
    # Step 4: Find the relevant region of N objects, i.e. the union of GT and predicted masks.
    # (A + B > 0) is a fast way to compute the union.
    # region_of_interest shape: (B, N, 1, H, W).
    region_of_interest = (position_mask_gt + position_mask_pred > 0).float()
    
    # Step 5: Create a weight map with shape (B, N, 1, H, W).
    
    # Reshape iou (B, N) for broadcasting: (B, N, 1, 1, 1).
    iou_broadcastable = iou.view(B, N, 1, 1, 1)
    
    # Use torch.where for an efficient implementation:
    # - inside region_of_interest, weight = iou_broadcastable
    # - outside region_of_interest, weight = 1.0
    # weights_per_object shape: (B, N, 1, H, W).
    weights_per_object = torch.where(
        region_of_interest > 0,   # Condition
        iou_broadcastable,        # if True, inside the region.
        torch.tensor(1.0, device=device) # if False, outside the region.
    )
    
    # Step 6: Multiply along the N dimension.
    # (B, N, 1, H, W) -> (B, 1, H, W)
    # torch.prod implements the multiplication that would otherwise be done in a loop.
    # If a pixel overlaps multiple regions, its weight becomes iou[i] * iou[j] * ...
    attention_weight = torch.prod(weights_per_object, dim=1)
    
    return attention_weight





def change_mask(position_mask,indices_list,height,width,max_sequence_length,device):
    image_token_H = height // 16
    image_token_W = width // 16
    atten_mask_list = []
    for bs in range(position_mask.shape[0]):
        instance_num  = position_mask.shape[1]
        HW = (height // 16)* (width // 16)
        seq_len = max_sequence_length 
        atten_mask = torch.zeros(seq_len+HW, seq_len+HW, device=device)
      
        for i in range(instance_num):
            image_instance_text_idxs = indices_list[bs][i]
            instance_img_in_patch_idxs = position_mask[bs,i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0]
            atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], :seq_len] = 1

        for i in range(instance_num):
            image_instance_text_idxs = indices_list[bs][i]
            instance_img_in_patch_idxs = position_mask[bs,i].reshape(image_token_H * image_token_W).nonzero(as_tuple=True)[0]
            soft_mask = position_mask[bs,i].reshape(image_token_H * image_token_W)
            # atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], :seq_len] = 1
            atten_mask[(seq_len + instance_img_in_patch_idxs)[:, None], image_instance_text_idxs] *= (1-soft_mask[instance_img_in_patch_idxs]).unsqueeze(1)
           
        # atten_mask = atten_mask.bool()
    
        atten_mask_list.append(atten_mask)

    atten_mask_list = torch.stack(atten_mask_list,dim=0)
    atten_mask_list = atten_mask_list.unsqueeze(1).repeat(1,24,1,1)

    return atten_mask_list


def save_model_card(repo_id: str, image_logs=None, base_model=str, repo_folder=None):
    img_str = ""
    if image_logs is not None:
        img_str = "You can find some example images below.\n\n"
        for i, log in enumerate(image_logs):
            images = log["images"]
            validation_prompt = log["validation_prompt"]
            validation_image = log["validation_image"]
            validation_image.save(os.path.join(repo_folder, "image_control.png"))
            img_str += f"prompt: {validation_prompt}\n"
            images = [validation_image] + images
            make_image_grid(images, 1, len(images)).save(os.path.join(repo_folder, f"images_{i}.png"))
            img_str += f"![images_{i})](./images_{i}.png)\n"

    model_description = f"""
# control-lora-{repo_id}

These are Control LoRA weights trained on {base_model} with new type of conditioning.
{img_str}

## License

Please adhere to the licensing terms as described [here](https://huggingface.co/black-forest-labs/FLUX.1-dev/blob/main/LICENSE.md)
"""

    model_card = load_or_create_model_card(
        repo_id_or_path=repo_id,
        from_training=True,
        license="other",
        base_model=base_model,
        model_description=model_description,
        inference=True,
    )

    tags = [
        "flux",
        "flux-diffusers",
        "text-to-image",
        "diffusers",
        "control-lora",
        "diffusers-training",
        "lora",
    ]
    model_card = populate_model_card(model_card, tags=tags)

    model_card.save(os.path.join(repo_folder, "README.md"))


def parse_args(input_args=None):
    parser = argparse.ArgumentParser(description="Simple example of a Control LoRA training script.")
    parser.add_argument(
        "--pretrained_model_name_or_path",
        type=str,
        default=None,
        required=True,
        help="Path to pretrained model or model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--model_with_gt_mask",
        type=str,
        default=None,
        required=True,
        help="Path to the model with ground truth mask.",
    )
    parser.add_argument(
        "--variant",
        type=str,
        default=None,
        help="Variant of the model files of the pretrained model identifier from huggingface.co/models, 'e.g.' fp16",
    )
    parser.add_argument(
        "--revision",
        type=str,
        default=None,
        required=False,
        help="Revision of pretrained model identifier from huggingface.co/models.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="control-lora",
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default=None,
        help="The directory where the downloaded models and datasets will be stored.",
    )
    parser.add_argument("--seed", type=int, default=12345, help="A seed for reproducible training.")
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
        help=(
            "The resolution for input images, all the images in the train/validation dataset will be resized to this"
            " resolution"
        ),
    )
    parser.add_argument(
        "--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader."
    )
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument(
        "--max_train_steps",
        type=int,
        default=None,
        help="Total number of training steps to perform.  If provided, overrides num_train_epochs.",
    )
    parser.add_argument(
        "--checkpointing_steps",
        type=int,
        default=500,
        help=(
            "Save a checkpoint of the training state every X updates. Checkpoints can be used for resuming training via `--resume_from_checkpoint`. "
            "In the case that the checkpoint is better than the final trained model, the checkpoint can also be used for inference."
            "Using a checkpoint for inference requires separate loading of the original pipeline and the individual checkpointed model components."
            "See https://huggingface.co/docs/diffusers/main/en/training/dreambooth#performing-inference-using-a-saved-checkpoint for step by step"
            "instructions."
        ),
    )
    parser.add_argument(
        "--checkpoints_total_limit",
        type=int,
        default=None,
        help=("Max number of checkpoints to store."),
    )
    parser.add_argument(
        "--resume_from_checkpoint",
        type=str,
        default=None,
        help=(
            "Whether training should be resumed from a previous checkpoint. Use a path saved by"
            ' `--checkpointing_steps`, or `"latest"` to automatically select the last available checkpoint.'
        ),
    )
    parser.add_argument(
        "--proportion_empty_prompts",
        type=float,
        default=0,
        help="Proportion of image prompts to be replaced with empty strings. Defaults to 0 (no prompt replacement).",
    )
    parser.add_argument(
        "--rank",
        type=int,
        default=4,
        help=("The dimension of the LoRA update matrices."),
    )
    parser.add_argument("--use_lora_bias", action="store_true", help="If training the bias of lora_B layers.")
    parser.add_argument(
        "--lora_layers",
        type=str,
        default=None,
        help=(
            'The transformer modules to apply LoRA training on. Please specify the layers in a comma separated. E.g. - "to_k,to_q,to_v,to_out.0" will result in lora training of attention layers only'
        ),
    )
    parser.add_argument(
        "--gaussian_init_lora",
        action="store_true",
        help="If using the Gaussian init strategy. When False, we follow the original LoRA init strategy.",
    )
    parser.add_argument("--train_norm_layers", action="store_true", help="Whether to train the norm scales.")
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of updates steps to accumulate before performing a backward/update pass.",
    )
    parser.add_argument(
        "--gradient_checkpointing",
        action="store_true",
        help="Whether or not to use gradient checkpointing to save memory at the expense of slower backward pass.",
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=5e-6,
        help="Initial learning rate (after the potential warmup period) to use.",
    )
    parser.add_argument(
        "--scale_lr",
        action="store_true",
        default=False,
        help="Scale the learning rate by the number of GPUs, gradient accumulation steps, and batch size.",
    )
    parser.add_argument(
        "--lr_scheduler",
        type=str,
        default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument(
        "--lr_warmup_steps", type=int, default=0, help="Number of steps for the warmup in the lr scheduler."
    )
    parser.add_argument(
        "--lr_num_cycles",
        type=int,
        default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")
    parser.add_argument(
        "--use_8bit_adam", action="store_true", help="Whether or not to use 8-bit Adam from bitsandbytes."
    )

    parser.add_argument(
        "--dataloader_num_workers",
        type=int,
        default=0,
        help=(
            "Number of subprocesses to use for data loading. 0 means that the data will be loaded in the main process."
        ),
    )
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--push_to_hub", action="store_true", help="Whether or not to push the model to the Hub.")
    parser.add_argument("--hub_token", type=str, default=None, help="The token to use to push to the Model Hub.")
    parser.add_argument(
        "--hub_model_id",
        type=str,
        default=None,
        help="The name of the repository to keep in sync with the local `output_dir`.",
    )
    parser.add_argument(
        "--logging_dir",
        type=str,
        default="logs",
        help=(
            "[TensorBoard](https://www.tensorflow.org/tensorboard) log directory. Will default to"
            " *output_dir/runs/**CURRENT_DATETIME_HOSTNAME***."
        ),
    )
    parser.add_argument(
        "--allow_tf32",
        action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument(
        "--report_to",
        type=str,
        default="tensorboard",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument(
        "--mixed_precision",
        type=str,
        default=None,
        choices=["no", "fp16", "bf16"],
        help=(
            "Whether to use mixed precision. Choose between fp16 and bf16 (bfloat16). Bf16 requires PyTorch >="
            " 1.10.and an Nvidia Ampere GPU.  Default to the value of accelerate config of the current system or the"
            " flag passed with the `accelerate.launch` command. Use this argument to override the accelerate config."
        ),
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=None,
        help=(
            "The name of the Dataset (from the HuggingFace hub) to train on (could be your own, possibly private,"
            " dataset). It can also be a path pointing to a local copy of a dataset in your filesystem,"
            " or to a folder containing files that 🤗 Datasets can understand."
        ),
    )
    parser.add_argument(
        "--dataset_config_name",
        type=str,
        default=None,
        help="The config of the Dataset, leave as None if there's only one config.",
    )
    parser.add_argument(
        "--image_column", type=str, default="image", help="The column of the dataset containing the target image."
    )
    parser.add_argument(
        "--conditioning_image_column",
        type=str,
        default="conditioning_image",
        help="The column of the dataset containing the control conditioning image.",
    )
    parser.add_argument(
        "--caption_column",
        type=str,
        default="text",
        help="The column of the dataset containing a caption or a list of captions.",
    )
    parser.add_argument("--log_dataset_samples", action="store_true", help="Whether to log somple dataset samples.")
    parser.add_argument(
        "--max_train_samples",
        type=int,
        default=None,
        help=(
            "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        ),
    )
    parser.add_argument(
        "--validation_prompt",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of prompts evaluated every `--validation_steps` and logged to `--report_to`."
            " Provide either a matching number of `--validation_image`s, a single `--validation_image`"
            " to be used with all prompts, or a single prompt that will be used with all `--validation_image`s."
        ),
    )
    parser.add_argument(
        "--validation_image",
        type=str,
        default=None,
        nargs="+",
        help=(
            "A set of paths to the control conditioning image be evaluated every `--validation_steps`"
            " and logged to `--report_to`. Provide either a matching number of `--validation_prompt`s, a"
            " a single `--validation_prompt` to be used with all `--validation_image`s, or a single"
            " `--validation_image` that will be used with all `--validation_prompt`s."
        ),
    )
    parser.add_argument(
        "--num_validation_images",
        type=int,
        default=1,
        help="Number of images to be generated for each `--validation_image`, `--validation_prompt` pair",
    )
    parser.add_argument(
        "--validation_steps",
        type=int,
        default=101000,
        help=(
            "Run validation every X steps. Validation consists of running the prompt"
            " `args.validation_prompt` multiple times: `args.num_validation_images`"
            " and logging the images."
        ),
    )
    parser.add_argument(
        "--tracker_project_name",
        type=str,
        default="flux_train_control_lora",
        help=(
            "The `project_name` argument passed to Accelerator.init_trackers for"
            " more information see https://huggingface.co/docs/accelerate/v0.17.0/en/package_reference/accelerator#accelerate.Accelerator"
        ),
    )
    parser.add_argument(
        "--jsonl_for_train",
        type=str,
        default=None,
        help="Path to the jsonl file containing the training data.",
    )

    parser.add_argument(
        "--guidance_scale",
        type=float,
        default=1.0,
        help="the guidance scale used for transformer.",
    )

    parser.add_argument(
        "--upcast_before_saving",
        action="store_true",
        help=(
            "Whether to upcast the trained transformer layers to float32 before saving (at the end of training). "
            "Defaults to precision dtype used for training to save memory"
        ),
    )

    parser.add_argument(
        "--weighting_scheme",
        type=str,
        default="none",
        choices=["sigma_sqrt", "logit_normal", "mode", "cosmap", "none"],
        help=('We default to the "none" weighting scheme for uniform sampling and uniform loss'),
    )
    parser.add_argument(
        "--logit_mean", type=float, default=0.0, help="mean to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--logit_std", type=float, default=1.0, help="std to use when using the `'logit_normal'` weighting scheme."
    )
    parser.add_argument(
        "--mode_scale",
        type=float,
        default=1.29,
        help="Scale of mode weighting scheme. Only effective when using the `'mode'` as the `weighting_scheme`.",
    )
    parser.add_argument(
        "--offload",
        action="store_true",
        help="Whether to offload the VAE and the text encoders to CPU when they are not used.",
    )

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    if args.dataset_name is None and args.jsonl_for_train is None:
        raise ValueError("Specify either `--dataset_name` or `--jsonl_for_train`")

    if args.dataset_name is not None and args.jsonl_for_train is not None:
        raise ValueError("Specify only one of `--dataset_name` or `--jsonl_for_train`")

    if args.proportion_empty_prompts < 0 or args.proportion_empty_prompts > 1:
        raise ValueError("`--proportion_empty_prompts` must be in the range [0, 1].")

    if args.validation_prompt is not None and args.validation_image is None:
        raise ValueError("`--validation_image` must be set if `--validation_prompt` is set")

    if args.validation_prompt is None and args.validation_image is not None:
        raise ValueError("`--validation_prompt` must be set if `--validation_image` is set")

    if (
        args.validation_image is not None
        and args.validation_prompt is not None
        and len(args.validation_image) != 1
        and len(args.validation_prompt) != 1
        and len(args.validation_image) != len(args.validation_prompt)
    ):
        raise ValueError(
            "Must provide either 1 `--validation_image`, 1 `--validation_prompt`,"
            " or the same number of `--validation_prompt`s and `--validation_image`s"
        )

    if args.resolution % 8 != 0:
        raise ValueError(
            "`--resolution` must be divisible by 8 for consistently sized encoded images between the VAE and the Flux transformer."
        )

    return args


def get_train_dataset(args, accelerator):
    dataset = None

    if args.dataset_name is not None:
        dataset = load_dataset(
            args.dataset_name,
            args.dataset_config_name,
            cache_dir=args.cache_dir,
        )
    elif args.jsonl_for_train is not None:
        dataset = load_dataset("json", data_files=args.jsonl_for_train, cache_dir=args.cache_dir)
        dataset = dataset.flatten_indices()
    else:
        raise ValueError("Either dataset_name or jsonl_for_train must be provided.")

    column_names = dataset["train"].column_names

    image_column = args.image_column or "image"
    caption_column = args.caption_column or "text"
    conditioning_image_column = args.conditioning_image_column or "conditioning_image"

    for col in [image_column, caption_column, conditioning_image_column]:
        if col not in column_names:
            raise ValueError(f"Column `{col}` not found in dataset columns: {', '.join(column_names)}")

    with accelerator.main_process_first():
        dataset = dataset["train"].shuffle(seed=args.seed)
        if args.max_train_samples is not None:
            dataset = dataset.select(range(args.max_train_samples))

    return dataset


# 2. prepare_train_dataset: process objects_info in one place.
def prepare_train_dataset(dataset, accelerator):
    image_transforms = transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )
    mask_image_transforms= transforms.Compose(
        [
            transforms.Resize((args.resolution, args.resolution), interpolation=transforms.InterpolationMode.BILINEAR),
            transforms.ToTensor(),
        ]
    )


    def preprocess_train(examples):
        images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.image_column]
        ]
        images = [image_transforms(image) for image in images]

        conditioning_images = [
            (image.convert("RGB") if not isinstance(image, str) else Image.open(image).convert("RGB"))
            for image in examples[args.conditioning_image_column]
        ]
        conditioning_images = [image_transforms(image) for image in conditioning_images]

        examples["pixel_values"] = images
        examples["conditioning_pixel_values"] = conditioning_images

        image_names = [image for image in examples[args.image_column]]
        examples["names"] = image_names

        # is_caption_list = isinstance(examples[args.caption_column][0], list)
        # if is_caption_list:
        #     examples["captions"] = [max(example, key=len) for example in examples[args.caption_column]]
        # else:
        #     examples["captions"] = list(examples[args.caption_column])
        import json
        json_paths = examples[args.caption_column]
        prompts = []
        all_masks = []
        all_masks_gt = []
        all_part_prompt =[]
        all_indices_list=[]
        all_iou_list =[]
        all_gt_flag_list =[]
        print(json_paths)

        for json_path in json_paths:
            with open(json_path, 'r') as f:
                data = json.load(f)
            
            prompts.append(data["long_prompt"])  # Read long_prompt.

            # Extract indices and masks for each object.
            indices_list = []
            part_prompt =[]
            mask_list = []
            mask_gt_list = []
            iou_list =[]
            gt_flag_list =[]
            mask_rle_list =[]
            for obj in data.get("object_list", []):
                
                mask_img = rle_to_pil_image(obj["mask"]).convert("L")
                mask_tensor = mask_image_transforms(mask_img)  # Optional: customize mask_transforms.
                mask_list.append(mask_tensor)

                # print("==========--------------")
                # print(obj["mask_gt"])
                if obj["mask_gt"]=="":
                    gt_flag_list.append(False)
                    mask_gt_tensor = torch.zeros_like(mask_tensor)
                    mask_gt_list.append(mask_gt_tensor)
                else:
                    gt_flag_list.append(True)
                    mask_gt_path = obj["mask_gt"]
                    mask_img_gt = Image.open(mask_gt_path).convert("L")
                    mask_gt_tensor = mask_image_transforms(mask_img_gt)  # Optional: customize mask_transforms.
                    mask_gt_list.append(mask_gt_tensor)

                part_prompt.append(obj["prompt"])
                indices_list.append(obj["indices"])
                iou_list.append(obj["iou"])
                mask_rle_list.append(obj["mask"])
            
            # for i in range(len(mask_list)):
            #     if iou_list[i] == 0.0:  # Skip masks that have already been cleared.
            #         continue
                
            #     for j in range(len(mask_list)):
            #         if mask_rle_list[i] == mask_rle_list[j]:
            #             continue
            #         if i == j or iou_list[j] == 0.0:
            #             continue
                        
            #         iou = calculate_iou(mask_list[i], mask_list[j])
            #         mask_i_bool = mask_list[i] > 0.5
            #         mask_j_bool = mask_list[j] > 0.5
            #         if iou > 0.7:
            #             if j>i:
            #                 # print(i,j, iou)
            #                 mask_list[j] = torch.zeros_like(mask_list[i])
            #                 iou_list[j] = 0.0
            #             else:
            #                 mask_list[i] = torch.zeros_like(mask_list[i])
            #                 iou_list[i] = 0.0
            #             continue
                    
            #         coverage = (mask_i_bool & mask_j_bool).sum().float() / (mask_j_bool.sum().float()+ 1e-6)
            #         # print(i,j, coverage)
            #         if coverage > 0.6:
            #             # Subtract mask_j from mask_i.
            #             mask_list[i] = (mask_i_bool & (~mask_j_bool)).float()


            all_indices_list.append(indices_list)
            all_masks.append(torch.stack(mask_list))
            all_masks_gt.append(torch.stack(mask_gt_list))
            all_part_prompt.append(part_prompt)
            all_iou_list.append(iou_list)
            all_gt_flag_list.append(gt_flag_list)

        examples["captions"] = prompts
        examples["indices"] = all_indices_list   # List[List[int]]
        examples["masks"] = all_masks       # List[List[str]]
        examples["masks_gt"] = all_masks_gt       # List[List[str]]
        examples["ious"] = all_iou_list
        examples["part_prompt"] = all_part_prompt 
        examples["gt_flag"]= all_gt_flag_list


        return examples

    with accelerator.main_process_first():
        dataset = dataset.with_transform(preprocess_train)

    return dataset


# 3. collate_fn function.
def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    pixel_values = pixel_values.to(memory_format=torch.contiguous_format).float()

    conditioning_pixel_values = torch.stack([example["conditioning_pixel_values"] for example in examples])
    conditioning_pixel_values = conditioning_pixel_values.to(memory_format=torch.contiguous_format).float()

    captions = [example["captions"] for example in examples]
    
    masks = torch.stack([example["masks"] for example in examples])
    masks = masks.to(memory_format=torch.contiguous_format).float()

    masks_gt = torch.stack([example["masks_gt"] for example in examples])
    masks_gt = masks_gt.to(memory_format=torch.contiguous_format).float()

    part_promt = [example["part_prompt"] for example in examples]
    indices = [example["indices"] for example in examples]
    ious = torch.tensor([example["ious"] for example in examples])
    ious = ious.to(memory_format=torch.contiguous_format).float()
    names = [example["names"] for example in examples]

    gt_flag = [example["gt_flag"] for example in examples]


    return {
        "pixel_values": pixel_values,
        "conditioning_pixel_values": conditioning_pixel_values,
        "captions": captions,
        "indices": indices,
        "masks":masks,
        "masks_gt":masks_gt,
        "ious":ious,
        "gt_flag":gt_flag,
        "part_prompt":part_promt,
        "names":names
    }


def main(args):
    if args.report_to == "wandb" and args.hub_token is not None:
        raise ValueError(
            "You cannot use both --report_to=wandb and --hub_token due to a security risk of exposing your token."
            " Please use `huggingface-cli login` to authenticate with the Hub."
        )
    if args.use_lora_bias and args.gaussian_init_lora:
        raise ValueError("`gaussian` LoRA init scheme isn't supported when `use_lora_bias` is True.")

    logging_out_dir = Path(args.output_dir, args.logging_dir)

    if torch.backends.mps.is_available() and args.mixed_precision == "bf16":
        # due to pytorch#99272, MPS does not yet support bfloat16.
        raise ValueError(
            "Mixed precision training with bfloat16 is not supported on MPS. Please use fp16 (recommended) or fp32 instead."
        )

    accelerator_project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=str(logging_out_dir))

    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to,
        project_config=accelerator_project_config,
    )

    # Disable AMP for MPS. A technique for accelerating machine learning computations on iOS and macOS devices.
    if torch.backends.mps.is_available():
        logger.info("MPS is enabled. Disabling AMP.")
        accelerator.native_amp = False

    # Make one log on every process with the configuration for debugging.
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        # DEBUG, INFO, WARNING, ERROR, CRITICAL
        level=logging.INFO,
    )
    logger.info(accelerator.state, main_process_only=False)

    if accelerator.is_local_main_process:
        transformers.utils.logging.set_verbosity_warning()
        diffusers.utils.logging.set_verbosity_info()
    else:
        transformers.utils.logging.set_verbosity_error()
        diffusers.utils.logging.set_verbosity_error()

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)

    # Handle the repository creation
    if accelerator.is_main_process:
        if args.output_dir is not None:
            os.makedirs(args.output_dir, exist_ok=True)

        if args.push_to_hub:
            repo_id = create_repo(
                repo_id=args.hub_model_id or Path(args.output_dir).name, exist_ok=True, token=args.hub_token
            ).repo_id

    # Load models. We will load the text encoders later in a pipeline to compute
    # embeddings.
    vae = AutoencoderKL.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="vae",
        revision=args.revision,
        variant=args.variant,
    )
    vae_scale_factor = 2 ** (len(vae.config.block_out_channels) - 1)
    flux_transformer = FluxTransformer2DModel.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="transformer",
        revision=args.revision,
        variant=args.variant,
    )
    logger.info("All models loaded successfully")



    noise_scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        args.pretrained_model_name_or_path,
        subfolder="scheduler",
    )
    noise_scheduler_copy = copy.deepcopy(noise_scheduler)
    vae.requires_grad_(False)
    flux_transformer.requires_grad_(False)

    # cast down and move to the CPU
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # let's not move the VAE to the GPU yet.
    vae.to(dtype=torch.float32)  # keep the VAE in float32.
    
    target_modules = [
            "x_embedder",
            "attn.to_k",
            "attn.to_q",
            "attn.to_v",
            "attn.to_out.0",
            "attn.add_k_proj",
            "attn.add_q_proj",
            "attn.add_v_proj",
            "attn.to_add_out",
            "ff.net.0.proj",
            "ff.net.2",
            "ff_context.net.0.proj",
            "ff_context.net.2",
        ]
    transformer_lora_config = LoraConfig(
        r=args.rank,
        lora_alpha=args.rank,
        init_lora_weights="gaussian" if args.gaussian_init_lora else True,
        target_modules=target_modules,
        lora_bias=args.use_lora_bias,
    )
    flux_transformer.add_adapter(transformer_lora_config)

    from safetensors.torch import load_file
    merge_state_dict ={}
    files = ["diffusion_pytorch_model-00001-of-00003.safetensors", 
            "diffusion_pytorch_model-00002-of-00003.safetensors",
            "diffusion_pytorch_model-00003-of-00003.safetensors"]

    # Iterate over files, load their data, and update merge_state_dict.
    for file in files:
        load_files_dict = load_file(os.path.join(args.model_with_gt_mask, "transformer", file))
        merge_state_dict.update(load_files_dict)
    
    

    for name, param in flux_transformer.named_parameters():
        if name not in merge_state_dict:
            print(f"Initializing missing parameter: {name}")
            merge_state_dict[name] = torch.zeros(param.shape).to(torch.bfloat16)

    print("-----------------------------------load")
    flux_transformer.load_state_dict(merge_state_dict, assign=True)
    flux_transformer.to(dtype=weight_dtype, device=accelerator.device)
    print("-----------------------------------init")

    for name, param in flux_transformer.named_parameters(): 
        if 'mask' in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
    
    # from torch.nn import init
    # for name, param in flux_transformer.named_parameters(): 
    #     if param.requires_grad == True:
    #         init.normal_(param, 0.0, 0.02)


    def unwrap_model(model):
        model = accelerator.unwrap_model(model)
        model = model._orig_mod if is_compiled_module(model) else model
        return model

    # `accelerate` 0.16.0 will have better support for customized saving
    if version.parse(accelerate.__version__) >= version.parse("0.16.0"):

        def save_model_hook(models, weights, output_dir):
            if accelerator.is_main_process:
                transformer_lora_layers_to_save = None

                for model in models:
                    if isinstance(unwrap_model(model), type(unwrap_model(flux_transformer))):
                        model = unwrap_model(model)

                        from collections import OrderedDict
                        from safetensors.torch import  save_file
                        mm_state_dict = OrderedDict()
                        state_dict = model.state_dict()
                        for key in state_dict:
                            if "local" in key:
                                mm_state_dict[key] = state_dict[key]
                        save_file(mm_state_dict, os.path.join(output_dir,'diffusion_part.safetensors'))
                        save_file(state_dict, os.path.join(output_dir,'diffusion_pytorch_model.safetensors'))
                        
                        transformer_norm_layers_to_save = {
                            f"transformer.{name}": param
                            for name, param in model.named_parameters()
                            if 'local' in name
                        }
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")

                    # make sure to pop weight so that corresponding model is not saved again
                    if weights:
                        weights.pop()

                FluxControlPipeline.save_lora_weights(
                    output_dir,
                    transformer_lora_layers=transformer_norm_layers_to_save,
                )

        def load_model_hook(models, input_dir):
            transformer_ = None

            if not accelerator.distributed_type == DistributedType.DEEPSPEED:
                while len(models) > 0:
                    model = models.pop()

                    if isinstance(model, type(unwrap_model(flux_transformer))):
                        transformer_ = model
                    else:
                        raise ValueError(f"unexpected save model: {model.__class__}")
            else:
                transformer_ = FluxTransformer2DModel.from_pretrained(
                    args.pretrained_model_name_or_path, subfolder="transformer"
                ).to(accelerator.device, weight_dtype)

                # Handle input dimension doubling before adding adapter
                with torch.no_grad():
                    initial_input_channels = transformer_.config.in_channels
                    new_linear = torch.nn.Linear(
                        transformer_.x_embedder.in_features * 2,
                        transformer_.x_embedder.out_features,
                        bias=transformer_.x_embedder.bias is not None,
                        dtype=transformer_.dtype,
                        device=transformer_.device,
                    )
                    new_linear.weight.zero_()
                    new_linear.weight[:, :initial_input_channels].copy_(transformer_.x_embedder.weight)
                    if transformer_.x_embedder.bias is not None:
                        new_linear.bias.copy_(transformer_.x_embedder.bias)
                    transformer_.x_embedder = new_linear
                    transformer_.register_to_config(in_channels=initial_input_channels * 2)

                transformer_.add_adapter(transformer_lora_config)

            lora_state_dict = FluxControlPipeline.lora_state_dict(input_dir)
            transformer_lora_state_dict = {
                f"{k.replace('transformer.', '')}": v
                for k, v in lora_state_dict.items()
                if k.startswith("transformer.") and "lora" in k
            }
            incompatible_keys = set_peft_model_state_dict(
                transformer_, transformer_lora_state_dict, adapter_name="default"
            )
            if incompatible_keys is not None:
                # check only for unexpected keys
                unexpected_keys = getattr(incompatible_keys, "unexpected_keys", None)
                if unexpected_keys:
                    logger.warning(
                        f"Loading adapter weights from state_dict led to unexpected keys not found in the model: "
                        f" {unexpected_keys}. "
                    )
            if args.train_norm_layers:
                transformer_norm_state_dict = {
                    k: v
                    for k, v in lora_state_dict.items()
                    if k.startswith("transformer.") and any(norm_k in k for norm_k in NORM_LAYER_PREFIXES)
                }
                transformer_._transformer_norm_layers = FluxControlPipeline._load_norm_into_transformer(
                    transformer_norm_state_dict,
                    transformer=transformer_,
                    discard_original_layers=False,
                )

            # Make sure the trainable params are in float32. This is again needed since the base models
            # are in `weight_dtype`. More details:
            # https://github.com/huggingface/diffusers/pull/6514#discussion_r1449796804
            if args.mixed_precision == "fp16":
                models = [transformer_]
                # only upcast trainable parameters (LoRA) into fp32
                cast_training_params(models)

        accelerator.register_save_state_pre_hook(save_model_hook)
        accelerator.register_load_state_pre_hook(load_model_hook)

    # Make sure the trainable params are in float32.
    if args.mixed_precision == "fp16":
        models = [flux_transformer]
        # only upcast trainable parameters (LoRA) into fp32
        cast_training_params(models, dtype=torch.float32)

    if args.gradient_checkpointing:
        flux_transformer.enable_gradient_checkpointing()

    # Enable TF32 for faster training on Ampere GPUs,
    # cf https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices
    if args.allow_tf32:
        torch.backends.cuda.matmul.allow_tf32 = True

    if args.scale_lr:
        args.learning_rate = (
            args.learning_rate * args.gradient_accumulation_steps * args.train_batch_size * accelerator.num_processes
        )

    # Use 8-bit Adam for lower memory usage or to fine-tune the model in 16GB GPUs
    if args.use_8bit_adam:
        try:
            import bitsandbytes as bnb
        except ImportError:
            raise ImportError(
                "To use 8-bit Adam, please install the bitsandbytes library: `pip install bitsandbytes`."
            )

        optimizer_class = bnb.optim.AdamW8bit
    else:
        optimizer_class = torch.optim.AdamW


    print("-----------------------------------opt")
    # Optimization parameters
    transformer_lora_parameters = list(filter(lambda p: p.requires_grad, flux_transformer.parameters()))
    optimizer = optimizer_class(
        transformer_lora_parameters,
        lr=args.learning_rate,
        betas=(args.adam_beta1, args.adam_beta2),
        weight_decay=args.adam_weight_decay,
        eps=args.adam_epsilon,
    )

    # Prepare dataset and dataloader.
    train_dataset = get_train_dataset(args, accelerator)
    train_dataset = prepare_train_dataset(train_dataset, accelerator)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        shuffle=True,
        collate_fn=collate_fn,
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
    )

    # Scheduler and math around the number of training steps.
    # Check the PR https://github.com/huggingface/diffusers/pull/8312 for detailed explanation.
    if args.max_train_steps is None:
        len_train_dataloader_after_sharding = math.ceil(len(train_dataloader) / accelerator.num_processes)
        num_update_steps_per_epoch = math.ceil(len_train_dataloader_after_sharding / args.gradient_accumulation_steps)
        num_training_steps_for_scheduler = (
            args.num_train_epochs * num_update_steps_per_epoch * accelerator.num_processes
        )
    else:
        num_training_steps_for_scheduler = args.max_train_steps * accelerator.num_processes

    lr_scheduler = get_scheduler(
        args.lr_scheduler,
        optimizer=optimizer,
        num_warmup_steps=args.lr_warmup_steps * accelerator.num_processes,
        num_training_steps=num_training_steps_for_scheduler,
        num_cycles=args.lr_num_cycles,
        power=args.lr_power,
    )
    # Prepare everything with our `accelerator`.
    print("-----------------------------------train")
    flux_transformer, optimizer, train_dataloader, lr_scheduler = accelerator.prepare(
        flux_transformer, optimizer, train_dataloader, lr_scheduler
    )

    # We need to recalculate our total training steps as the size of the training dataloader may have changed.
    num_update_steps_per_epoch = math.ceil(len(train_dataloader) / args.gradient_accumulation_steps)
    if args.max_train_steps is None:
        args.max_train_steps = args.num_train_epochs * num_update_steps_per_epoch
        if num_training_steps_for_scheduler != args.max_train_steps * accelerator.num_processes:
            logger.warning(
                f"The length of the 'train_dataloader' after 'accelerator.prepare' ({len(train_dataloader)}) does not match "
                f"the expected length ({len_train_dataloader_after_sharding}) when the learning rate scheduler was created. "
                f"This inconsistency may result in the learning rate scheduler not functioning properly."
            )
    # Afterwards we recalculate our number of training epochs
    args.num_train_epochs = math.ceil(args.max_train_steps / num_update_steps_per_epoch)

    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        tracker_config = dict(vars(args))

        # tensorboard cannot handle list types for config
        tracker_config.pop("validation_prompt")
        tracker_config.pop("validation_image")

        accelerator.init_trackers(args.tracker_project_name, config=tracker_config)

    # Train!
    total_batch_size = args.train_batch_size * accelerator.num_processes * args.gradient_accumulation_steps

    logger.info("***** Running training *****")
    logger.info(f"  Num examples = {len(train_dataset)}")
    logger.info(f"  Num batches each epoch = {len(train_dataloader)}")
    logger.info(f"  Num Epochs = {args.num_train_epochs}")
    logger.info(f"  Instantaneous batch size per device = {args.train_batch_size}")
    logger.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {total_batch_size}")
    logger.info(f"  Gradient Accumulation steps = {args.gradient_accumulation_steps}")
    logger.info(f"  Total optimization steps = {args.max_train_steps}")
    global_step = 0
    first_epoch = 0

    # Create a pipeline for text encoding. We will move this pipeline to GPU/CPU as needed.
    text_encoding_pipeline = FluxControlPipeline.from_pretrained(
        args.pretrained_model_name_or_path, transformer=None, vae=None, torch_dtype=weight_dtype
    )

    # Potentially load in the weights and states from a previous save
    if args.resume_from_checkpoint:
        if args.resume_from_checkpoint != "latest":
            path = os.path.basename(args.resume_from_checkpoint)
        else:
            # Get the most recent checkpoint
            dirs = os.listdir(args.output_dir)
            dirs = [d for d in dirs if d.startswith("checkpoint")]
            dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
            path = dirs[-1] if len(dirs) > 0 else None

        if path is None:
            logger.info(f"Checkpoint '{args.resume_from_checkpoint}' does not exist. Starting a new training run.")
            args.resume_from_checkpoint = None
            initial_global_step = 0
        else:
            logger.info(f"Resuming from checkpoint {path}")
            accelerator.load_state(os.path.join(args.output_dir, path))
            global_step = int(path.split("-")[1])

            initial_global_step = global_step
            first_epoch = global_step // num_update_steps_per_epoch
    else:
        initial_global_step = 0

    if accelerator.is_main_process and args.report_to == "wandb" and args.log_dataset_samples:
        logger.info("Logging some dataset samples.")
        formatted_images = []
        formatted_control_images = []
        all_prompts = []
        for i, batch in enumerate(train_dataloader):
            images = (batch["pixel_values"] + 1) / 2
            control_images = (batch["conditioning_pixel_values"] + 1) / 2
            prompts = batch["captions"]

            if len(formatted_images) > 10:
                break

            for img, control_img, prompt in zip(images, control_images, prompts):
                formatted_images.append(img)
                formatted_control_images.append(control_img)
                all_prompts.append(prompt)

        logged_artifacts = []
        for img, control_img, prompt in zip(formatted_images, formatted_control_images, all_prompts):
            logged_artifacts.append(wandb.Image(control_img, caption="Conditioning"))
            logged_artifacts.append(wandb.Image(img, caption=prompt))

        wandb_tracker = [tracker for tracker in accelerator.trackers if tracker.name == "wandb"]
        wandb_tracker[0].log({"dataset_samples": logged_artifacts})

    progress_bar = tqdm(
        range(0, args.max_train_steps),
        initial=initial_global_step,
        desc="Steps",
        # Only show the progress bar once on each machine.
        disable=not accelerator.is_local_main_process,
    )

    def get_sigmas(timesteps, n_dim=4, dtype=torch.float32):
        sigmas = noise_scheduler_copy.sigmas.to(device=accelerator.device, dtype=dtype)
        schedule_timesteps = noise_scheduler_copy.timesteps.to(accelerator.device)
        timesteps = timesteps.to(accelerator.device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]

        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    image_logs = None

    loss_dice  =  DiceLoss(use_sigmoid=True, activate=True, reduction='mean', naive_dice=True, eps=1.0,loss_weight=0.5)
    loss_mask = CrossEntropyLoss(use_sigmoid=True, reduction='mean',loss_weight=2.0)
    from torch.utils.tensorboard import SummaryWriter
    writer = SummaryWriter(log_dir='runs/exp1') 


    for epoch in range(first_epoch, args.num_train_epochs):
        flux_transformer.train()
        for step, batch in enumerate(train_dataloader):
            with accelerator.accumulate(flux_transformer):
                # Convert images to latent space
                # vae encode
                pixel_latents = encode_images(batch["pixel_values"], vae.to(accelerator.device), weight_dtype)
                control_latents = encode_images(
                    batch["conditioning_pixel_values"], vae.to(accelerator.device), weight_dtype
                )

                if args.offload:
                    # offload vae to CPU.
                    vae.cpu()

                # Sample a random timestep for each image
                # for weighting schemes where we sample timesteps non-uniformly
                bsz = pixel_latents.shape[0]
                noise = torch.randn_like(pixel_latents, device=accelerator.device, dtype=weight_dtype)
                u = compute_density_for_timestep_sampling(
                    weighting_scheme=args.weighting_scheme,
                    batch_size=bsz,
                    logit_mean=args.logit_mean,
                    logit_std=args.logit_std,
                    mode_scale=args.mode_scale,
                )
                indices = (u * noise_scheduler_copy.config.num_train_timesteps).long()
                timesteps = noise_scheduler_copy.timesteps[indices].to(device=pixel_latents.device)

                # Add noise according to flow matching.
                sigmas = get_sigmas(timesteps, n_dim=pixel_latents.ndim, dtype=pixel_latents.dtype)
                noisy_model_input = (1.0 - sigmas) * pixel_latents + sigmas * noise
                # Concatenate across channels.
                # Question: Should we concatenate before adding noise?
                concatenated_noisy_model_input = torch.cat([noisy_model_input, control_latents], dim=1)

                # pack the latents.
                packed_noisy_model_input = FluxControlPipeline._pack_latents(
                    concatenated_noisy_model_input,
                    batch_size=bsz,
                    num_channels_latents=concatenated_noisy_model_input.shape[1],
                    height=concatenated_noisy_model_input.shape[2],
                    width=concatenated_noisy_model_input.shape[3],
                )

                # latent image ids for RoPE.
                latent_image_ids = FluxControlPipeline._prepare_latent_image_ids(
                    bsz,
                    concatenated_noisy_model_input.shape[2] // 2,
                    concatenated_noisy_model_input.shape[3] // 2,
                    accelerator.device,
                    weight_dtype,
                )

                # handle guidance
                if unwrap_model(flux_transformer).config.guidance_embeds:
                    guidance_vec = torch.full(
                        (bsz,),
                        args.guidance_scale,
                        device=noisy_model_input.device,
                        dtype=weight_dtype,
                    )
                else:
                    guidance_vec = None

                # text encoding.
                captions = batch["captions"]
                text_encoding_pipeline = text_encoding_pipeline.to("cuda")
                with torch.no_grad():
                    _,_,height,width = batch["pixel_values"].shape
                    position_mask = batch["masks"].squeeze(2)  # [1, 5, 1024, 1024]
                    max_sequence_length = 512
                    position_mask = F.interpolate(position_mask, size=(height // 16, width // 16), mode='nearest').unsqueeze(2)

                    position_mask_gt = batch["masks_gt"].squeeze(2)  # [1, 5, 1024, 1024]
                    max_sequence_length = 512
                    position_mask_gt = F.interpolate(position_mask_gt, size=(height // 16, width // 16), mode='nearest').unsqueeze(2)


                    prompt_embeds, pooled_prompt_embeds, text_ids= text_encoding_pipeline.encode_prompt(
                        captions, prompt_2=None,
                        max_sequence_length=max_sequence_length,
                     
                    )
                
      
                
                if args.proportion_empty_prompts and random.random() < args.proportion_empty_prompts:
                    prompt_embeds.zero_()
                    pooled_prompt_embeds.zero_()
                    atten_mask_list[:]= False
                if args.offload:
                    text_encoding_pipeline = text_encoding_pipeline.to("cpu")

                indices_list = batch["indices"]
                iou_list = batch["ious"]
                gt_flag = batch['gt_flag']

                joint_attention_kwargs = {}
                position_mask = position_mask.to(prompt_embeds.device).to(weight_dtype)
                position_mask_gt = position_mask_gt.to(prompt_embeds.device).to(weight_dtype)
                iou_list = iou_list.to(prompt_embeds.device).to(weight_dtype)

                atten_mask_list = change_mask(position_mask,indices_list,height,width,max_sequence_length,device=prompt_embeds.device)
                joint_attention_kwargs['attention_mask'] = 1-atten_mask_list
                joint_attention_kwargs['return_mask'] = True
                

                _, position_mask_predict,selected_indices = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    joint_attention_kwargs=joint_attention_kwargs,
                    position_mask = position_mask,
                    indices_list= indices_list,
                    iou_list = iou_list,
                    return_mask=True,
                )
                position_mask_predict_ = position_mask_predict.sigmoid()
                position_mask_predict_list =[]
                # print(selected_indices,position_mask.shape[1] )
                for object_index in range(position_mask.shape[1]):
                    if object_index in selected_indices:
                        position_mask_predict_list.append(position_mask_predict_[:,object_index:object_index+1])
                    else:
                        position_mask_predict_list.append(position_mask[:,object_index:object_index+1])
                position_mask_predict_two_stage = torch.concat(position_mask_predict_list,dim=1)
                # print(position_mask_predict_two_stage.shape)

                atten_mask_list = change_mask(position_mask_predict_two_stage.to(prompt_embeds.device),indices_list,height,width,max_sequence_length,device=prompt_embeds.device)
                joint_attention_kwargs['attention_mask'] = 1-atten_mask_list.to(prompt_embeds.device)
                joint_attention_kwargs['return_mask'] = False



                # Predict.
                
                model_pred = flux_transformer(
                    hidden_states=packed_noisy_model_input,
                    timestep=timesteps / 1000,
                    guidance=guidance_vec,
                    pooled_projections=pooled_prompt_embeds,
                    encoder_hidden_states=prompt_embeds,
                    txt_ids=text_ids,
                    img_ids=latent_image_ids,
                    return_dict=False,
                    joint_attention_kwargs=joint_attention_kwargs,
                    return_mask=False,
                )[0]
                model_pred = FluxControlPipeline._unpack_latents(
                    model_pred,
                    height=noisy_model_input.shape[2] * vae_scale_factor,
                    width=noisy_model_input.shape[3] * vae_scale_factor,
                    vae_scale_factor=vae_scale_factor,
                )
                weighting = compute_loss_weighting_for_sd3(weighting_scheme=args.weighting_scheme, sigmas=sigmas)

                
                #mask loss
                position_mask_gt = position_mask_gt[:, selected_indices,:, :, :]
                gt_flag = torch.tensor(gt_flag)
                gt_flag = gt_flag[:,selected_indices]
                pred_mask = position_mask_predict[0,:,0][gt_flag[0]]
                gt_mask = position_mask_gt[0,:,0][gt_flag[0]]
                # print(pred_mask.shape, gt_mask.shape)
    
                # flow-matching loss
                target = noise - pixel_latents
                loss = torch.mean(
                    (weighting.float() * (model_pred.float() - target.float()) ** 2).reshape(target.shape[0], -1),
                    1,
                )
                loss_sd = loss.mean()

                sample_mask = False
                if sample_mask:
                    sampled_pred_mask, sampled_gt_mask = sample_points(pred_mask, gt_mask)
                    sam_loss_dice = loss_dice( sampled_pred_mask, sampled_gt_mask, avg_factor=(len(gt_mask) + 1e-4))
                    sam_loss_mask = loss_mask( sampled_pred_mask.reshape(-1), sampled_gt_mask.reshape(-1),avg_factor=(pred_mask.shape[0] * sampled_pred_mask.shape[1] + 1e-4))
                else:
                    sam_loss_dice = loss_dice( pred_mask, gt_mask)
                    sam_loss_mask = loss_mask(  pred_mask, gt_mask)
                loss = loss_sd + sam_loss_dice + sam_loss_mask 
                print(loss,loss_sd,sam_loss_dice,sam_loss_mask )
                writer.add_scalar('Loss/sam_loss_dice', sam_loss_dice, epoch)
                writer.add_scalar('Loss/sam_loss_mask', sam_loss_mask, epoch)
                writer.add_scalar('Loss/loss', loss, epoch)
                accelerator.backward(loss)


                if accelerator.sync_gradients:
                    params_to_clip = flux_transformer.parameters()
                    accelerator.clip_grad_norm_(params_to_clip, args.max_grad_norm)
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()

            # Checks if the accelerator has performed an optimization step behind the scenes
            if accelerator.sync_gradients:
                progress_bar.update(1)
                global_step += 1

                # DeepSpeed requires saving weights on every device; saving weights only on the main process would cause issues.
                if accelerator.distributed_type == DistributedType.DEEPSPEED or accelerator.is_main_process:
                    if global_step % args.checkpointing_steps == 0:
                        # _before_ saving state, check if this save would set us over the `checkpoints_total_limit`
                        flux_transformer_ = unwrap_model(flux_transformer)
                        dtype = (
                            torch.float16
                            if args.mixed_precision == "fp16"
                            else torch.bfloat16
                            if args.mixed_precision == "bf16"
                            else torch.float32
                        )
                        flux_transformer_ = flux_transformer_.to(dtype)
                        save_path = os.path.join(args.output_dir, f"checkpoint-{global_step}")
                        flux_transformer_.save_pretrained(
                            os.path.join(save_path,"transformer"),
                            safe_serialization=True,
                            max_shard_size="10GB",
                        )
                        
                       

                    if args.validation_prompt is not None and global_step % args.validation_steps == 0:
                        image_logs = log_validation(
                            flux_transformer=flux_transformer,
                            args=args,
                            accelerator=accelerator,
                            weight_dtype=weight_dtype,
                            step=global_step,
                        )

            logs = {"loss": loss.detach().item(), "lr": lr_scheduler.get_last_lr()[0]}
            progress_bar.set_postfix(**logs)
            accelerator.log(logs, step=global_step)

            if global_step >= args.max_train_steps:
                break

    # Create the pipeline using using the trained modules and save it.
    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        flux_transformer = unwrap_model(flux_transformer)
        if args.upcast_before_saving:
            flux_transformer.to(torch.float32)
        transformer_lora_layers = get_peft_model_state_dict(flux_transformer)
        if args.train_norm_layers:
            transformer_norm_layers = {
                f"transformer.{name}": param
                for name, param in flux_transformer.named_parameters()
                if any(k in name for k in NORM_LAYER_PREFIXES)
            }
            transformer_lora_layers = {**transformer_lora_layers, **transformer_norm_layers}
        FluxControlPipeline.save_lora_weights(
            save_directory=args.output_dir,
            transformer_lora_layers=transformer_lora_layers,
        )

        del flux_transformer
        del text_encoding_pipeline
        del vae
        free_memory()

        # Run a final round of validation.
        image_logs = None
        if args.validation_prompt is not None:
            image_logs = log_validation(
                flux_transformer=None,
                args=args,
                accelerator=accelerator,
                weight_dtype=weight_dtype,
                step=global_step,
                is_final_validation=True,
            )

        if args.push_to_hub:
            save_model_card(
                repo_id,
                image_logs=image_logs,
                base_model=args.pretrained_model_name_or_path,
                repo_folder=args.output_dir,
            )
            upload_folder(
                repo_id=repo_id,
                folder_path=args.output_dir,
                commit_message="End of training",
                ignore_patterns=["step_*", "epoch_*", "*.pt", "*.bin"],
            )

    accelerator.end_training()


if __name__ == "__main__":
    args = parse_args()
    main(args)
