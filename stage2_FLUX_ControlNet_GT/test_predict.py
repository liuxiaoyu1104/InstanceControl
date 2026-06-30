import argparse
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from peft import LoraConfig
from pycocotools import mask as mask_utils
from safetensors.torch import load_file
from torchvision import transforms
from transformers import T5TokenizerFast

from flux_transformer import FluxTransformer2DModel
from pipeline import FluxControlPipeline


LORA_TARGET_MODULES = [
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
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def parse_args():
    parser = argparse.ArgumentParser(description="Generate images with a GT-mask FLUX ControlNet checkpoint.")
    parser.add_argument("--model_path", required=True, help="Base FLUX model directory.")
    parser.add_argument(
        "--checkpoint_dir",
        required=True,
        help=(
            "Checkpoint directory. Automatically supports either transformer/*.safetensors "
            "or lora.safetensors."
        ),
    )
    parser.add_argument("--canny_pth", required=True, help="Control-image directory.")
    parser.add_argument("--box_pth", required=True, help="JSON directory.")
    parser.add_argument("--result_path", required=True, help="Output image directory.")
    parser.add_argument("--mask_root", default=None, help="Root directory for relative GT mask paths.")
    parser.add_argument("--rank", type=int, default=64, help="LoRA rank used during training.")
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--height", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--guidance_scale", type=float, default=30.0)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def add_lora_adapter(transformer, rank):
    config = LoraConfig(
        r=rank,
        lora_alpha=rank,
        init_lora_weights=False,
        target_modules=LORA_TARGET_MODULES,
        lora_bias=False,
    )
    transformer.add_adapter(config)


def print_incompatible_weights(name, incompatible):
    if incompatible.missing_keys:
        print(f"{name}: missing keys: {len(incompatible.missing_keys)}")
    if incompatible.unexpected_keys:
        print(f"{name}: unexpected keys: {len(incompatible.unexpected_keys)}")


def load_full_transformer(transformer, weights_dir):
    """Load a save_pretrained-style transformer directory."""
    weight_files = sorted(weights_dir.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors files found in {weights_dir}")

    state_dict = {}
    for weight_file in weight_files:
        state_dict.update(load_file(weight_file))

    incompatible = transformer.load_state_dict(state_dict, assign=True)
    print(f"Loaded {len(weight_files)} transformer shard(s) from {weights_dir}")
    print_incompatible_weights("Transformer checkpoint", incompatible)


def load_split_checkpoint(transformer, checkpoint_path):
    """Load a separately exported LoRA adapter."""
    lora_file = checkpoint_path / "lora.safetensors"
    if not lora_file.is_file():
        raise FileNotFoundError(lora_file)

    lora_state_dict = load_file(lora_file)
    if not lora_state_dict:
        raise ValueError(f"No LoRA keys found in {lora_file}")

    incompatible = transformer.load_state_dict(lora_state_dict, strict=False, assign=True)
    print(f"Loaded LoRA weights from {lora_file}")
    print_incompatible_weights("LoRA checkpoint", incompatible)


def load_transformer(model_path, checkpoint_dir, rank):
    transformer = FluxTransformer2DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
    )
    add_lora_adapter(transformer, rank)

    checkpoint_path = Path(checkpoint_dir)
    if not checkpoint_path.is_dir():
        raise NotADirectoryError(checkpoint_path)

    weights_dir = checkpoint_path / "transformer"
    if any(weights_dir.glob("*.safetensors")):
        print("Detected full transformer checkpoint layout.")
        load_full_transformer(transformer, weights_dir)
    else:
        print("Detected split LoRA checkpoint layout.")
        load_split_checkpoint(transformer, checkpoint_path)
    return transformer


def rle_to_pil_image(rle_data, size):
    rle = {
        "counts": rle_data["counts"].encode("utf-8") if isinstance(rle_data["counts"], str) else rle_data["counts"],
        "size": rle_data["size"],
    }
    mask_array = mask_utils.decode(rle) * 255
    image = Image.fromarray(mask_array.astype(np.uint8))
    return image.resize(size, Image.BILINEAR)


def resolve_mask_path(mask_path, json_path, mask_root):
    path = Path(mask_path)
    if path.is_absolute():
        return path

    candidates = []
    if mask_root is not None:
        candidates.append(Path(mask_root) / path)
    candidates.extend([json_path.parent / path, json_path.parent.parent / path])

    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return candidates[0]


def get_token_indices(offsets, start_char, end_char):
    return [
        index
        for index, (start, end) in enumerate(offsets)
        if start < end_char and end > start_char
    ]


def get_mask_inputs(json_path, tokenizer, mask_transform, image_size):
    with open(json_path, "r", encoding="utf-8") as file:
        data = json.load(file)

    caption = data["caption"]
    encoding = tokenizer(caption, return_offsets_mapping=True, return_tensors="pt", add_special_tokens=False)
    offsets = encoding["offset_mapping"][0].tolist()

    masks, phrases, indices = [], [], []
    start_index = 0
    for phrase, mask_data in zip(data.get("phrases", []), data.get("pred_masks", [])):
        if phrase in {"", "[SEG]", "<p>"} or len(phrase) < 2:
            continue
        try:
            start_char = caption.lower().index(phrase.lower(), start_index)
        except ValueError:
            print(f"Skipping unmatched phrase in {json_path.name}: {phrase}")
            continue
        end_char = start_char + len(phrase)
        start_index = end_char

        masks.append(mask_transform(rle_to_pil_image(mask_data["rle"], image_size)))
        phrases.append(phrase)
        indices.append(get_token_indices(offsets, start_char, end_char))

    if not masks:
        return None

    return {
        "prompts": [caption],
        "mask_list": torch.stack(masks).unsqueeze(0).to(torch.bfloat16),
        "part_prompt": [phrases],
        "indices_list": [indices],
    }


def main():
    args = parse_args()
    canny_dir = Path(args.canny_pth)
    json_dir = Path(args.box_pth)
    result_dir = Path(args.result_path)
    for directory in (canny_dir, json_dir):
        if not directory.is_dir():
            raise NotADirectoryError(directory)
    result_dir.mkdir(parents=True, exist_ok=True)

    transformer = load_transformer(args.model_path, args.checkpoint_dir, args.rank)
    pipeline = FluxControlPipeline.from_pretrained(
        args.model_path,
        transformer=transformer,
        torch_dtype=torch.bfloat16,
    ).to(args.device)
    tokenizer = T5TokenizerFast.from_pretrained(args.model_path, subfolder="tokenizer_2")
    mask_transform = transforms.Compose([
        transforms.Resize((args.height, args.width), interpolation=transforms.InterpolationMode.BILINEAR),
        transforms.ToTensor(),
    ])

    image_paths = sorted(path for path in canny_dir.iterdir() if path.suffix.lower() in IMAGE_SUFFIXES)
    for image_path in image_paths:
        json_path = json_dir / f"{image_path.stem}.json"
        if not json_path.exists():
            print(f"Skipping {image_path.name}: no JSON at {json_path}")
            continue

        inputs = get_mask_inputs(json_path, tokenizer, mask_transform, (args.width, args.height))
        if inputs is None:
            print(f"Skipping {image_path.name}: no valid masks")
            continue

        output_path = result_dir / f"{image_path.stem}.png"
        if output_path.exists():
            print(f"Skipping {image_path.name}: output already exists")
            continue

        control_image = Image.open(image_path).convert("RGB")
        image = pipeline(
            prompt=inputs["prompts"],
            prompt_2=inputs["prompts"],
            control_image=control_image,
            height=args.height,
            width=args.width,
            num_inference_steps=args.num_inference_steps,
            guidance_scale=args.guidance_scale,
            generator=torch.Generator().manual_seed(args.seed),
            mask_list=inputs["mask_list"],
            part_prompt=inputs["part_prompt"],
            indices_list=inputs["indices_list"],
        ).images[0]
        image.save(output_path)
        print(f"Saved {output_path}")


if __name__ == "__main__":
    main()
