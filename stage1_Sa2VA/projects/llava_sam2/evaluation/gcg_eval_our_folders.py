import argparse
import math
import os
import torch
import tqdm
from pycocotools import mask as mask_utils

from transformers import (AutoModel, AutoModelForCausalLM, AutoTokenizer,
                          BitsAndBytesConfig, CLIPImageProcessor,
                          CLIPVisionModel, GenerationConfig)

from utils import _init_dist_pytorch, get_dist_info, collect_results_cpu
from PIL import Image
import re
import json
import numpy as np

def parse_args():
    parser = argparse.ArgumentParser(description='GCG')
    parser.add_argument('--model_path', default="/hdd5/wanghuan/controlnet/Sa2Va/Sa2VA_7_4gpus_encode_lora/work_dirs/hf_model_22500", help='hf model path.')
    parser.add_argument(
        '--split',
        default='val',
        help='Specify a split')
    parser.add_argument(
        '--save_dir',
        default='/root/autodl-tmp/a_test_visi_6/json_pred_init_canny_ga',
        help='save path')
    parser.add_argument(
        '--image_dir',
        required=True,
        help='Image directory.')
    parser.add_argument(
        '--json_dir',
        required=True,
        help='JSON directory.')
    parser.add_argument(
        '--launcher',
        choices=['none', 'pytorch', 'slurm', 'mpi'],
        default='none',
        help='job launcher')
    parser.add_argument('--local_rank', '--local-rank', type=int, default=0)
    args = parser.parse_args()
    if 'LOCAL_RANK' not in os.environ:
        os.environ['LOCAL_RANK'] = str(args.local_rank)
    return args


class GCGInferenceDataset:
    def __init__(self,
                 image_folder,
                 json_folder,
                 save_dir=None,
                 ):
        self.image_folder = image_folder
        self.json_folder = json_folder

        self.images = sorted(os.listdir(image_folder))[:]

        if save_dir is not None:
            # filter evaluated
            self.save_dir = save_dir
            exsits_files = os.listdir(self.save_dir)
            exsits_files = [_file[:-5] for _file in exsits_files]
            print(exsits_files)
            _images = []
            for i, item in enumerate(self.images):
                if item[:-4] not in exsits_files:
                    _images.append(item)
            self.images = _images

    def __len__(self):
        return len(self.images)

    def get_questions(self,prompt):
    #         DEFAULT_IMAGE_TOKEN + 'Please identify each object described in {prompt}, and provide the detailed description along with its corresponding interleaved segmentation mask in image.',
    # DEFAULT_IMAGE_TOKEN + 'Locate all objects mentioned in {prompt} and generate the interleaved segmentation mask for each one in image.',
    # DEFAULT_IMAGE_TOKEN + 'Using both the provided image and the text in {prompt}, locate each mentioned object, then generate a thorough description and an interleaved segmentation mask for every instance',
    # DEFAULT_IMAGE_TOKEN + 'Based on the image and the description in {prompt}, identify all described objects and provide the description along with its corresponding interleaved segmentation mask for each one',
    # DEFAULT_IMAGE_TOKEN + 'Analyze the image along with the description in {prompt}, extract each object, and return the description along with a corresponding segmentation mask',
    # DEFAULT_IMAGE_TOKEN + 'Recognize all objects in the image based on the content of {prompt} and provide extended descriptions accompanied by interleaved segmentation masks'
        question = "Please identify each object described in {prompt}, and provide the rewirte description along with its corresponding interleaved segmentation mask in image."
        question = question.format(prompt=prompt)
        return question

    def __getitem__(self, index):
        data_dict = {}
        image_file = self.images[index]
        json_file = os.path.join(self.json_folder, image_file.split('.')[0]+'.json')
        if not os.path.exists(json_file):
            data_dict = None
        else:
            with open(json_file, 'r') as f:
                data = json.load(f)
                prompt=data["long_prompt"]
            questions = self.get_questions(prompt)
            
            data_dict['image_file'] = image_file

            image_file = os.path.join(self.image_folder, image_file)
            image = Image.open(image_file).convert('RGB')
            image = image.resize((512,512))

            data_dict['image'] = image
            data_dict['text'] = "<image>\n" + questions

            data_dict['img_id'] = image_file
        return data_dict

def main():
    args = parse_args()

    if args.launcher != 'none':
        _init_dist_pytorch('nccl')
        rank, world_size = get_dist_info()
        torch.cuda.set_device(rank)
    else:
        rank = 0
        world_size = 1

    # build model
    model = AutoModel.from_pretrained(
        args.model_path,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        use_flash_attn=True,
        trust_remote_code=True,
        local_files_only=True
    ).eval().cuda()

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        trust_remote_code=True,
    )

    os.makedirs(args.save_dir, exist_ok=True)

    image_folder = args.image_dir
    json_folder = args.json_dir
    print(image_folder)
    # try:
    dataset = GCGInferenceDataset(
        image_folder=image_folder,
        json_folder=json_folder,
        save_dir=args.save_dir,
    )

    results = []
    n_samples = len(dataset)
    per_rank_samples = math.ceil(n_samples / world_size) + 1
    per_rank_ids = range(per_rank_samples * rank,
                        min(n_samples, per_rank_samples * (rank + 1)))
    count = 0
    for idx in tqdm.tqdm(per_rank_ids):
        data_batch = dataset[idx]
        if data_batch is None:
            continue
        else:
            count += 1
        
        
        prediction = {'img_id': data_batch['img_id'], 'image_file': data_batch['image_file']}
        del data_batch['img_id'], data_batch['image_file']

        w, h = data_batch['image'].size
        # print(w, h)
        pred_dict = model.predict_forward(**data_batch, tokenizer=tokenizer)
        if 'prediction_masks' not in pred_dict.keys() or pred_dict['prediction_masks'] is None or len(pred_dict['prediction_masks']) == 0:
            print("No SEG !!!")
            prediction['prediction_masks'] = torch.zeros((0, h, w), dtype=torch.bool, device='cuda')
        else:
            # Convert numpy.ndarray values to torch.Tensor and move them to GPU.
            tensor_list = []
            for m in pred_dict['prediction_masks']:
                if isinstance(m, np.ndarray):
                    tensor_list.append(torch.from_numpy(m).to('cuda'))
                elif isinstance(m, torch.Tensor):
                    tensor_list.append(m.to('cuda'))
                else:
                    raise TypeError(f"Unsupported type {type(m)} in prediction_masks")
            
            prediction['prediction_masks'] = torch.stack(tensor_list, dim=0)[:, 0]
        process_and_save_output(
            args.save_dir,
            prediction['image_file'],
            pred_dict['prediction'],
            prediction['prediction_masks'],
            pred_dict['ious'],
            pred_dict['scores']
        )
        results.append(pred_dict['prediction'])

    results = collect_results_cpu(results, len(dataset), tmpdir='./gcg_eval_tmp')
    # except:
    #     print("===============")


def process_and_save_output(output_dir, image_name, text_output, pred_masks, ious, scores):
# def process_and_save_output(output_dir, image_name, text_output, pred_masks):
    if not os.path.exists(output_dir):
        os.mkdir(output_dir)

    text_output = text_output.replace("<s>", "").replace("\n", "").replace("  ", " ")
    text_output = text_output.split("ASSISTANT: ")[-1]

    cleaned_str = re.sub(r'<.*?>', '', text_output)
    pattern = re.compile(r'<p>(.*?)<\/p>')
    phrases = pattern.findall(text_output)
    phrases = [p.strip() for p in phrases]

    pattern_id = re.compile(r'\[SEG\] <(\d+)>')
    ids = pattern_id.findall(text_output)
    ids = [i.strip() for i in ids]
    # print(ids)
    

    # Remove the [SEG] token
    cleaned_str = cleaned_str.replace('[SEG]', '')

    # Strip unnecessary spaces
    cleaned_str = ' '.join(cleaned_str.split()).strip("'")
    cleaned_str = cleaned_str.strip()

    # Convert the predicted masks into RLE format
    pred_masks_tensor = pred_masks.cpu()
    uncompressed_mask_rles = mask_to_rle_pytorch(pred_masks_tensor)
    rle_masks = []
    for idx, m in enumerate(uncompressed_mask_rles):
        rle_masks.append(coco_encode_rle(idx, m, ious, scores))
        # rle_masks.append(coco_encode_rle(idx, m))
    # for rle_mask in rle_masks:

    # Create results dictionary
    # print(f"clean_str: {cleaned_str}")
    result_dict = {
        "image_id": image_name[:-4],
        "caption": cleaned_str,
        "phrases": phrases,
        "ids": ids,
        "pred_masks": rle_masks,
    }

    # print(cleaned_str)
    # print(phrases)

    output_path = f"{output_dir}/{image_name[:-4]}.json"

    with open(output_path, 'w') as f:
        json.dump(result_dict, f, indent=2)

    return

def mask_to_rle_pytorch(tensor: torch.Tensor):
    """
    Encodes masks to an uncompressed RLE, in the format expected by
    pycoco tools.
    """
    # Put in fortran order and flatten h,w
    b, h, w = tensor.shape
    tensor = tensor.permute(0, 2, 1).flatten(1)

    # Compute change indices
    diff = tensor[:, 1:] ^ tensor[:, :-1]
    change_indices = diff.nonzero()

    # Encode run length
    out = []
    for i in range(b):
        cur_idxs = change_indices[change_indices[:, 0] == i, 1]
        cur_idxs = torch.cat(
            [torch.tensor([0], dtype=cur_idxs.dtype, device=cur_idxs.device), cur_idxs + 1,
             torch.tensor([h * w], dtype=cur_idxs.dtype, device=cur_idxs.device), ]
        )
        btw_idxs = cur_idxs[1:] - cur_idxs[:-1]
        counts = [] if tensor[i, 0] == 0 else [0]
        counts.extend(btw_idxs.detach().cpu().tolist())
        out.append({"size": [h, w], "counts": counts})

    return out

def coco_encode_rle(idx, uncompressed_rle, ious, scores):
# def coco_encode_rle(idx, uncompressed_rle):
    h, w = uncompressed_rle["size"]
    rle = mask_utils.frPyObjects(uncompressed_rle, h, w)
    rle["counts"] = rle["counts"].decode("utf-8")  # Necessary to serialize with json
    result = {
        "rle": rle,
        "iou": float(ious[idx]),     # Convert to float to avoid Tensor serialization issues.
        "score": float(scores[idx])  # Same as above.
    }
    # return rle
    return result

if __name__ == '__main__':
    main()
