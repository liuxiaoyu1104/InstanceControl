import os
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = "2"
import torch
import torch.distributed as dist
from torch.nn import functional as F
import os
from tqdm import tqdm
import PIL
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
import cv2
import multiprocessing
from multiprocessing import Manager
import copy
import random
import argparse
from accelerate.utils import set_seed

from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
import json



def save_overlay_cv2(img_input, masks_input, folder_path, prefix, index, ious=None, alpha=0.5):
    """
    img_input: (H, W, C) ndarray (RGB)
    masks_input: (N, H, W) ndarray
    folder_path: Subdirectory for saving results.
    prefix: "pred" or "gt".
    index: Image index in the batch.
    ious: IoU list corresponding to each mask, only meaningful for predictions.
    """
    if torch.is_tensor(img_input):
        img_np = img_input.permute(1, 2, 0).cpu().numpy()
    else:
        img_np = img_input.copy()

    if img_np.max() <= 1.1:
        img_np = (img_np * 255).astype(np.uint8)
    
    # Convert to BGR for OpenCV.
    img_bgr = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
    os.makedirs(folder_path, exist_ok=True)

    if masks_input.ndim == 2:
        masks_input = masks_input[None, :, :]

    for j in range(masks_input.shape[0]):
        mask = masks_input[j].astype(np.bool_)
        overlay = img_bgr.copy()
        
        # Fix the color seed so the same object has consistent GT and Pred colors.
        np.random.seed(j) 
        color = np.random.randint(0, 255, (3,)).tolist()
        overlay[mask] = color
        result = cv2.addWeighted(overlay, alpha, img_bgr, 1 - alpha, 0)
        
        # Draw IoU on the image if available.
        if ious is not None and j < len(ious):
            iou_val = ious[j].item() if torch.is_tensor(ious[j]) else ious[j]
            text = f"IoU: {iou_val:.4f}"
            cv2.putText(result, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            save_filename = f"obj{j}_{prefix}_iou{iou_val:.3f}.png"
        else:
            save_filename = f"obj{j}_{prefix}.png"
            
        cv2.imwrite(os.path.join(folder_path, save_filename), result)


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

def worker(args):
    set_seed(42)
    
    model_cfg = "configs/sam2.1/sam2.1_hiera_l.yaml"
    predictor = SAM2ImagePredictor(build_sam2(model_cfg, args.sam2_checkpoint))
    
    score = 0 
    num_batches = 0
    
    # Create the directory in advance.
    res_dir = os.path.join(args.output_dir, "miou")
    os.makedirs(res_dir, exist_ok=True)

    for image_name in os.listdir(args.image_root)[:]:
        # --- Check whether this sample has already been evaluated. ---
        file_id = image_name.split('.')[0]
        txt_path = os.path.join(res_dir, f"{file_id}.txt")
        
        if os.path.exists(txt_path) and image_name.endswith(('.jpg','.png')):
            # print("image name", image_name, image_name.split('.')[0]+".json")
            # print(txt_path)
            with open(txt_path, 'r') as f:
                try:
                    
                    line = f.read().strip()
                    score_image, num_batches_image = map(float, line.split(','))
                    print(f"Loaded existing result for {image_name}: {score_image},{num_batches_image}")
                    score += score_image
                    num_batches += int(num_batches_image)
                    continue # Skip subsequent computation.
                except:
                    pass # Recompute if the file is corrupted.
        # -----------------------------
        

        json_path = os.path.join(args.json_path,image_name.split('.')[0]+".json")
        with open(json_path, 'r') as f:
            data = json.load(f)
        masks_all = []
        boxes_all = []
        image_path = os.path.join(args.image_root, image_name)
        image = Image.open(image_path).convert('RGB')
        img_w, img_h = image.size
        score_image = 0 
        num_batches_image = 0
        for obj in data.get("object_list", []):
            # mask_path = os.path.join('/root/autodl-tmp/lisa_test_coco',obj["mask"])
            mask_path_base = args.image_root.split('image')[0]
            mask_path = os.path.join(mask_path_base,obj["mask"])
            mask_img = Image.open(mask_path).convert("L") 
            mask_img_np = (np.array(mask_img) / 255.0).astype(np.uint8)

            mask = (mask_img_np==1)
            x0, y0, x1, y1 = mask2box(mask)
            box = np.array([
                x0 / img_w,
                y0 / img_h,
                x1 / img_w ,
                y1 / img_h ,
            ])
            boxes_all.append(box)
            masks_all.append(mask)
        
        masks_all = np.stack(masks_all, axis=0) # n,h,w
        masks_all = torch.from_numpy(masks_all)
        masks_all = masks_all[None,...]
        masks_all = F.interpolate(masks_all.float(), size=args.resolution, mode='nearest-exact')
        masks_all = masks_all[0,...].long() # n,h,w
        print(masks_all.shape)



        masks_all = [masks_all]
        boxes_all = [boxes_all]
        image_names = [image_name]
        import glob
        if len(glob.glob(os.path.join(args.gen_img_dir,image_name.split('.')[0]+'.png')))>0:
            gen_img_paths = [os.path.join(args.gen_img_dir,name.split('.')[0]+'.png') for name in image_names]
        else:
            gen_img_paths = [os.path.join(args.gen_img_dir,name.split('.')[0]+'.jpg') for name in image_names]
        
        gen_imgs  = [Image.open(path).convert('RGB').resize([args.resolution,args.resolution],resample=Image.BICUBIC) for path in gen_img_paths]
        gen_imgs = [np.array(img) for img in gen_imgs]


        gt_imgs  = [image.resize([args.resolution,args.resolution],resample=Image.BICUBIC)]
        gt_imgs = [np.array(img) for img in gt_imgs]
        
        mask_input_batch = [] 
        box_batch = []
        point_coords_batch = []
        gts = []
        for label,boxes in zip(masks_all,boxes_all):
            gt_label = label.cpu().numpy()
            resized_label = F.interpolate(label[None,...].float(),size=[256,256],mode='nearest-exact')
            resized_label = resized_label[0,...].long().cpu().numpy() # n,256,256
            
            region_num = len(gt_label)
            masks = []
            temp_boxes = []
            points = []
            
            for i in range(region_num):
                mask = resized_label[i:i+1] # 1,256,256
                masks.append(mask)
                
                box = boxes[i]*args.resolution
                temp_boxes.append(box)
                
                point  = None
                points.append(point)
                
            box_batch.append(temp_boxes)
            mask_input_batch.append(masks)
            gts.append(gt_label)
            point_coords_batch.append(points)
        
        for i,(img,gt_img,masks,boxes) in enumerate(zip(gen_imgs,gt_imgs,mask_input_batch,box_batch)):
            points = point_coords_batch[i]
            with torch.inference_mode():
                image_batch = [img] * len(boxes)
                predictor.set_image_batch(image_batch)

                masks_batch, scores_batch, _ = predictor.predict_batch(
                    point_coords_batch = None,
                    point_labels_batch = None,
                    box_batch=boxes,
                    mask_input_batch=masks,
                    multimask_output=False
                )

            masks_batch = [m[0].astype(np.bool_) for m in masks_batch] # list of (h,w)



            with torch.inference_mode():
                image_batch = [gt_img] * len(boxes)
                predictor.set_image_batch(image_batch)

                gt_masks_batch, scores_batch, _ = predictor.predict_batch(
                    point_coords_batch = None,
                    point_labels_batch = None,
                    box_batch=boxes,
                    mask_input_batch=masks,
                    multimask_output=False
                )

            gt_masks_batch = [m[0].astype(np.bool_) for m in gt_masks_batch] # list of (h,w)

            
            
            # target = torch.from_numpy(gts[i]).long()
            target = torch.from_numpy(np.stack(gt_masks_batch,axis=0)).long()
            preds = torch.from_numpy(np.stack(masks_batch,axis=0)).long()

            intersection = torch.sum(preds & target, dim=[1,2])
            target_sum = torch.sum(target, dim=[1,2])
            pred_sum = torch.sum(preds, dim=[1,2])
            union = target_sum + pred_sum - intersection
            iou = torch.where(union != 0, intersection / union, 0)


            # vis_folder = os.path.join(args.output_dir, "visualizations", image_name.split('.')[0])
            # os.makedirs(vis_folder, exist_ok=True) 
            # save_overlay_cv2(img, np.stack(masks_batch,axis=0), vis_folder, "pred", i, ious=iou)
            # # Save Ground Truth.
            # save_overlay_cv2(img,  gts[i], vis_folder, "gt", i)
            
            score_image = torch.sum(iou).item()
            num_batches_image = len(masks_batch)
        score += score_image
        num_batches += num_batches_image
        os.makedirs(os.path.join(args.output_dir,"miou"), exist_ok=True)
        with open(os.path.join(args.output_dir,"miou", image_name.split('.')[0]+'.txt'), "a") as f:
            # f.write(f"{round(average_iou_image, 4)}")
            f.write(f"{round(score_image, 4)},{num_batches_image}")
        os.makedirs(os.path.join(args.output_dir,"miou_image"), exist_ok=True)
        with open(os.path.join(args.output_dir,"miou_image", image_name.split('.')[0]+'.txt'), "a") as f:
            # f.write(f"{round(average_iou_image, 4)}")
            f.write(f"{round(score_image, 4)/num_batches_image}")

        
    return score, num_batches
    
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
        "--sam2_checkpoint",
        type=str,
        default='/root/autodl-tmp/controlnet/compare/metric/huggingface/sam2.1_hiera_large.pt',
    )
    parser.add_argument(
        "--gen_img_dir",
        type=str,
        default="/root/autodl-tmp/controlnet/compare/result/seg2any/image"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/root/autodl-tmp/controlnet/compare/result/seg2any/"
    )
    parser.add_argument(
        "--resolution",
        type=int,
        default=1024,
    )

    parser.add_argument(
        "--num_replicas",
        type=int,
        default=1
    )
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)  
    
    
    score,num_batches = worker(args)
    average_iou = (score/num_batches)
    print(f"Average IoU across all GPUs: {average_iou}")
    with open(os.path.join(args.output_dir,"miou.txt"), "a") as f:
        f.write(f"miou: {round(average_iou, 4)}\n")
