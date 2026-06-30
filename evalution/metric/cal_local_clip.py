import json
import math
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
import numpy as np
from PIL import Image
import torch
import os
from transformers import T5TokenizerFast
import argparse
import cv2
from collections import defaultdict
import glob
import torch
from PIL import Image
from transformers import (
    AutoImageProcessor,
    AutoTokenizer,
    AutoModelForCausalLM,
)

def mask2box(mask):
    if isinstance(mask,torch.Tensor):
        ys, xs = torch.where(mask==1)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    else:
        ys, xs = np.where(mask==1)
        y0, y1 = ys.min(), ys.max()
        x0, x1 = xs.min(), xs.max()
    return x0, y0, x1, y1


def save_similarity_map(similarity_map, label_name, output_dir):
    patch_size = int(math.sqrt(similarity_map.shape[0]))
    similarity_map_reshaped = similarity_map.reshape(patch_size, patch_size)

    # Normalize to the 0-255 range.
    similarity_map_norm = cv2.normalize(similarity_map_reshaped, None, 0, 255, cv2.NORM_MINMAX)
    similarity_map_uint8 = similarity_map_norm.astype(np.uint8)

    # Apply pseudo color.
    heatmap = cv2.applyColorMap(similarity_map_uint8, cv2.COLORMAP_JET)

    # Save the image.
    output_path = os.path.join(output_dir, f"{label_name}.png")
    cv2.imwrite(output_path, heatmap)


# base_pth= "/root/autodl-tmp/controlnet/data/test/our_test_add"
# def load_mask(mask_path, image_size):
#     mask = Image.open(mask_path).convert("L").resize((image_size, image_size))
#     return np.array(mask) > 127

def get_similarity_map(image, model, processor, text_feature):
    image_input = processor.preprocess(image, return_tensors='pt')['pixel_values'].to(text_feature.device)
    with torch.no_grad():
        image_feat = model.get_image_dense_features(image_input)
    image_feat = image_feat / image_feat.norm(p=2, dim=-1, keepdim=True)
    similarity = image_feat.squeeze(0) @ text_feature.squeeze(0).T
    return similarity.cpu().numpy()

def compute_mask_score(similarity_map, mask_bool):
    patch_size = int(math.sqrt(similarity_map.shape[0]))
    similarity_map = similarity_map.reshape(patch_size, patch_size)
    mask_resized = Image.fromarray(mask_bool.astype(np.uint8)*255).resize((patch_size, patch_size), resample=Image.NEAREST)
    mask_bool_resized = np.array(mask_resized) > 127
    if mask_bool_resized.sum() == 0:
        return 0.0
    return similarity_map[mask_bool_resized].mean()

def extract_local_prompt(long_prompt, indices, tokenizer_2):
    text_inputs = tokenizer_2(
        [long_prompt],
        padding="max_length",
        max_length=512,
        truncation=True,
        return_length=False,
        return_overflowing_tokens=False,
        return_tensors="pt",
    )
    text_input_ids = text_inputs.input_ids[0]
    instance_text_index = [torch.tensor(indices, dtype=torch.int32)]
    part_prompt = tokenizer_2.convert_ids_to_tokens(text_input_ids[instance_text_index])
    part_prompt = "".join(part_prompt).replace("▁", " ").strip()
    
    return part_prompt

def analyze_objects_similarity(json_path, image_path, model, tokenizer,processor, device,image_size,ori_image_path,output_path,mask_path_base):
    # with open(json_path, 'r', encoding='utf-8') as f:
    #     content = f.read()
    
    # data = json.loads(content)
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    ori_image_w, ori_image_h = Image.open(ori_image_path).convert("RGB").size
    image = Image.open(image_path).convert("RGB")
    image = image.resize((1024, 1024))

    results = []
    # output_dir = os.path.join(output_path,image_path.split('/')[-1].split('.')[0])
    # print(output_dir)
    # os.makedirs(output_dir, exist_ok=True)
    for obj in data["object_list"]:
        mask_path = os.path.join(mask_path_base, obj["mask"])
        if "Bbox" in obj:
            x_min, y_min, w, h = obj["Bbox"] 
            # ori_image_w = 1800 
            # ori_image_h = 1800
            x_min = x_min / ori_image_w * 1024
            w = w / ori_image_w * 1024
            y_min = y_min / ori_image_h * 1024
            h = h / ori_image_h * 1024
            x_max = x_min + w
            y_max= y_min + h
        else:
            mask_path = os.path.join(mask_path_base,obj["mask"])
            mask_img = Image.open(mask_path).convert("L") 
            mask_img_np = (np.array(mask_img) / 255.0).astype(np.uint8)

            mask = (mask_img_np==1)
            x_min, y_min, x_max, y_max = mask2box(mask)
            x_min = x_min / ori_image_w * 1024
            x_max = x_max / ori_image_w * 1024
            y_min = y_min / ori_image_h * 1024
            y_max = y_max / ori_image_h * 1024



        # Crop the image region.
        mask = Image.open(mask_path).convert("L").resize((1024, 1024))
        mask_part = mask.crop((x_min, y_min, x_max, y_max))
        mask_part = np.array(mask_part) > 127
        image_part = image.crop((x_min, y_min, x_max, y_max))
        image_part = image_part.resize((image_size, image_size))

        # image_part_cv = cv2.cvtColor(np.array(image_part), cv2.COLOR_RGB2BGR)
        # part_save_path = os.path.join(output_dir, f"{obj['Label Name']}_{obj['Instance ID']}_image.png")
        # cv2.imwrite(part_save_path, image_part_cv)

        text_input = tokenizer(obj["prompt"], max_length=77, padding="max_length", truncation=True, return_tensors="pt").to(device)
        with torch.no_grad():
            text_feat = model.get_text_features(text_input.input_ids, walk_short_pos=True)
        text_feat = text_feat / text_feat.norm(p=2, dim=-1, keepdim=True)

        sim_map = get_similarity_map(image_part, model, processor, text_feat)
        score = compute_mask_score(sim_map, mask_part)

        # Save the sim_map heatmap.
        # save_similarity_map(sim_map, f"{obj['Label Name']}_{obj['Instance ID']}_simmap", output_dir)

        results.append({
            "Instance ID": obj["Instance ID"],
            "Label Name": obj["Label Name"],
            "local_prompt": obj["prompt"],
            "similarity_score": round(float(score), 5)
        })

    return results



def cal_clip(args):
    model_root = args.fg_clip_path
    image_size = 224
    model = AutoModelForCausalLM.from_pretrained(model_root, trust_remote_code=True).cuda()
    device = model.device
    tokenizer = AutoTokenizer.from_pretrained(model_root)
    image_processor = AutoImageProcessor.from_pretrained(model_root)

    score = 0
    count = 0
    
    # Create the save directory in advance.
    save_dir = os.path.join(args.output_dir, "local_clip")
    os.makedirs(save_dir, exist_ok=True)

    for image_name in os.listdir(args.image_root)[:]:
        image_name = image_name.split('/')[-1]
        file_id = image_name.split('.')[0]
        txt_path = os.path.join(save_dir, f"{file_id}.txt")
        
        # --- Resume logic: directly read the txt file if it already exists. ---
        if os.path.exists(txt_path):
            try:
                with open(txt_path, 'r') as f:
                    avg_score = float(f.read().strip())
                print(f"Loaded {image_name}: {avg_score}")
            except:
                # Recompute if reading fails, for example when the file is empty.
                pass
        else:
            # Original computation logic.
            image_path = os.path.join(args.gen_img_dir, file_id + '.jpg')
            if not os.path.exists(image_path):
                image_path = os.path.join(args.gen_img_dir, file_id + '.png')
                if not os.path.exists(image_path):
                    continue
            
            ori_image_path = os.path.join(args.image_root, image_name)
            json_path = os.path.join(args.json_path, file_id + '.json')
            mask_path_base = args.image_root.split('image')[0]
            
            results = analyze_objects_similarity(json_path, image_path, model, tokenizer, image_processor, device, image_size, ori_image_path, args.output_dir, mask_path_base)

            avg_score = sum([r['similarity_score'] for r in results]) / len(results)
            
            # Save the single-image result in 'w' mode to avoid duplicate appends.
            with open(txt_path, "w") as f:
                f.write(f"{round(avg_score, 4)}")
            print(f"Computed {image_name}: {avg_score}")

        score += avg_score
        count += 1

    return (score / count) if count > 0 else 0

if __name__ =="__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image_root",
        type=str,
        default='/root/autodl-tmp/controlnet/data/test/our_test_add/image',
    )
    
    parser.add_argument(
        "--json_path",
        type=str,
        default='/root/autodl-tmp/controlnet/data/test/our_test_add/json',
    )

    parser.add_argument(
        "--gen_img_dir",
        type=str,
        default="/root/autodl-tmp/controlnet/compare/result/seg2any/image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/controlnet/compare/result/seg2any"
    )
    parser.add_argument(
        "--fg_clip_path",
        type=str,
        default="/root/autodl-tmp/controlnet/compare/metric/huggingface/fg-clip"
    )


    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True) 

    average_iou = cal_clip(args)
    print(f"Average IoU across all GPUs: {average_iou}")
    with open(os.path.join(args.output_dir,"local_clip.txt"), "a") as f:
        f.write(f"miou: {round(average_iou, 4)}\n")
