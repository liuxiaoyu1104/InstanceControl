import os
import cv2
import json
import numpy as np
from pycocotools import mask as mask_utils
import matplotlib.pyplot as plt
import random
import argparse

def random_color():
    return [random.randint(0, 255) for _ in range(3)]

def visualize_mask_on_image(image_path, json_path, save_dir=None, alpha=0.5):
    # Read the image.
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image,(512,512))
    

    # Read the JSON file.
    with open(json_path, "r") as f:
        data = json.load(f)

    phrases = data.get("phrases", [])
    rle_masks = data.get("pred_masks", [])

    # Iterate over each mask.
    for idx, pred in enumerate(rle_masks):
        if pred is None:
            continue
        if phrases[idx] in ["<p>", "", "[SEG]"]:
            continue
        rle = pred.get("rle", None)
        if rle is None or "counts" not in rle:
            print(f"[Skip] mask {idx} has no valid RLE")
            continue
        mask = mask_utils.decode(rle)  # HxW binary mask.
        color = random_color()

        # Copy the original image and draw one mask separately.
        overlay = image.copy()
        overlay[mask == 1] = (
            overlay[mask == 1] * (1 - alpha) + np.array(color) * alpha
        ).astype(np.uint8)
        print("====")
        mask = (mask*255).astype(np.uint8)
        
        iou = pred.get("iou")
        score = pred.get("score")
        iou_str = f"{iou:.2f}".replace('.', '_')   # 0.71 -> "0_71"
        score_str = f"{score:.2f}".replace('.', '_')


        # Save the individual mask image.
        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(image_path))[0]
            os.makedirs(os.path.join(save_dir, base_name), exist_ok=True)
            save_path = os.path.join(save_dir, base_name, f"{idx+1}_{phrases[idx][:100]}_IoU{iou_str}_Score{score_str}.png")
            cv2.imwrite(save_path, cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
            save_path = os.path.join(save_dir, base_name, f"{idx+1}_{phrases[idx][:100]}_IoU{iou_str}_Score{score_str}_mask.png")
            cv2.imwrite(save_path, cv2.cvtColor(mask, cv2.COLOR_RGB2BGR))
            print(f"mask {idx} saved to: {save_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--image_dir", 
            default="/hdd5/controlnet/a_test_vis_all/a_test_vis/hed", 
            help="Directory containing the original images"
        )
    parser.add_argument(
        "--json_dir", 
        default="/hdd5/controlnet/a_test_vis_all/a_test_vis/json_pred_init_hed", 
        help="Directory containing the inference JSON files"
    )
    parser.add_argument(
        "--save_dir", 
        default="/hdd5/controlnet/a_test_vis_all/a_test_vis/json_pred_init_hed_vis", 
        help="Directory for saving visualization results"
    )    
    args = parser.parse_args()
    img_list = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith((".jpg", ".png", ".jpeg"))])
    for img_name in img_list[:]:
        img_path = os.path.join(args.image_dir, img_name)

        # Find the corresponding JSON file, assuming the same base name with a different suffix.
        json_name = os.path.splitext(img_name)[0] + ".json"
        json_path = os.path.join(args.json_dir, json_name)

        # Output path for saving results.
        # save_path = os.path.join(args.save_dir, img_name)
        save_path = args.save_dir

        if not os.path.exists(json_path):
            print(f"[Skip] corresponding JSON file not found: {json_path}")
            continue

        print(f"[Process] {img_path} -> {save_path}")
        try:
            visualize_mask_on_image(img_path, json_path, save_path)
        except:
            print("error")
