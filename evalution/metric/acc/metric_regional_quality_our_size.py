import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["VLLM_LOGGING_LEVEL"] = "ERROR"
import argparse
import json
import cv2
import os
import numpy as np
from PIL import Image

def mask2box(mask: np.ndarray) -> tuple[int, int, int, int]:
    """
    Compute the bounding box from a boolean mask.
    
    Returns coordinates in (x0, y0, x1, y1) format, where the end index is exclusive.
    The result can be used directly for NumPy slicing, such as image[y0:y1, x0:x1].
    """
    y_indices, x_indices = np.where(mask)
    if y_indices.size == 0 or x_indices.size == 0:
        return 0, 0, 0, 0  # Return an empty region if the mask is empty.
        
    x0, x1 = x_indices.min(), x_indices.max()
    y0, y1 = y_indices.min(), y_indices.max()
    
    # Return coordinates with an exclusive end index for easier slicing.
    return int(x0), int(y0), int(x1) + 1, int(y1) + 1

def adjust_bbox_to_min_side(x0, y0, x1, y1, image_shape, min_side=400):
    """
    Adjust the bounding box so each side is at least min_side and stays within image bounds.
    """
    img_height, img_width = image_shape[:2]
    
    # --- Adjust width. ---
    box_width = x1 - x0
    if box_width < min_side:
        center_x = (x0 + x1) // 2
        x0 = center_x - min_side // 2
        x1 = x0 + min_side
        
        # Boundary correction.
        if x0 < 0:
            x0, x1 = 0, min_side
        if x1 > img_width:
            x1 = img_width
            x0 = img_width - min_side

    # --- Adjust height. ---
    box_height = y1 - y0
    if box_height < min_side:
        center_y = (y0 + y1) // 2
        y0 = center_y - min_side // 2
        y1 = y0 + min_side
        
        # Boundary correction.
        if y0 < 0:
            y0, y1 = 0, min_side
        if y1 > img_height:
            y1 = img_height
            y0 = img_height - min_side
            
    # --- Finally ensure all coordinates are valid, including images smaller than min_side. ---
    final_x0 = max(0, x0)
    final_y0 = max(0, y0)
    final_x1 = min(img_width, x1)
    final_y1 = min(img_height, y1)

    return int(final_x0), int(final_y0), int(final_x1), int(final_y1)

def draw_edge_overlay(image: np.ndarray, mask: np.ndarray, color=(0, 255, 0), thickness=3) -> np.ndarray:
    """
    Draw the mask contour on the image.
    
    Args:
        image: Image as a NumPy array, either RGB or grayscale.
        mask: Boolean or 0/1 NumPy array with the same size as image.
        color: Contour color in BGR format. Defaults to green.
        thickness: Contour line width.

    Returns:
        A new image copy with the contour drawn.
    """
    # Ensure the mask is in an OpenCV-compatible format, 8-bit single-channel.
    mask_uint8 = mask.astype(np.uint8) * 255

    kernel = np.ones((15, 15), np.uint8)
    mask_uint8 = cv2.dilate(mask_uint8, kernel, iterations=1)
    
    # Find contours.
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Draw contours on a copy of the image.
    overlay_image = image.copy()
    cv2.drawContours(overlay_image, contours, -1, color, thickness)
    
    return overlay_image


def worker(args):

    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "29500"  
    # CUDA_VISIBLE_DEVICES must be set before importing torch. Otherwise, CUDA_VISIBLE_DEVICES will have no effect.
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams
    from accelerate.utils import set_seed
    import torch
    from torch.nn import functional as F
    from utils.visualizer import save_image_with_caption,Visualizer
    from utils.utils import mask2box
    from qwen_processor import QwenProcessor
    
    set_seed(42)

    processor = AutoProcessor.from_pretrained(args.model_id)
    qwen_processor = QwenProcessor(processor,
                               min_pixels=224*224,max_pixels=1280 * 28 * 28)
    
    llm = LLM(
        model=args.model_id,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        limit_mm_per_prompt={"image": 1, "video": 1},
        gpu_memory_utilization = args.gpu_memory_utilization,
        tensor_parallel_size=args.tensor_parallel_size,
    )
    sampling_params = SamplingParams(
        temperature=0.1,
        top_p=0.001,
        repetition_penalty=1.05,
        max_tokens=int(1e8),  # Set to a large value so that the effective limit is solely determined by `max_model_len`.
        stop_token_ids=[],
    )

    
    visualizer = Visualizer()
    
    # skip the regional caption where `attribute` is not mentioned.
    skip_anno_ids = {
        k: list() for k in args.questions.keys()
    }
    p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids.json")
    if hasattr(args,'pre_questions') and not os.path.exists(p):        
        
    
        for image_name in os.listdir(args.image_root)[990:]:
            llm_inputs = []
            json_path = os.path.join(args.json_path,image_name.split('.')[0]+".json")
            with open(json_path, 'r') as f:
                data = json.load(f)

            for obj in data.get("object_list", []):
                for key in args.pre_questions.keys():
                    short_caption = obj["prompt_noun"]
                    caption = obj["prompt"]
                    prompt = args.pre_questions[key].format(short_caption=short_caption,caption=caption)
                    llm_input = qwen_processor.process([],prompt)
                    llm_input["question"] = prompt
                    llm_input["anno_id"] = image_name.split('.')[0]+"_"+ str(obj['Instance ID'])
                    llm_inputs.append(llm_input)
        
            for i in range(0,len(llm_inputs),args.batch_size):
                outputs = llm.generate(llm_inputs[i:i+args.batch_size], sampling_params=sampling_params,use_tqdm=False)
                for j, output in enumerate(outputs):
                    output = output.outputs[0]
                    answer = output.text.strip()
                    llm_inputs[i+j]["answer"] = answer
                    
            assert len(llm_inputs) % len(args.pre_questions) == 0
            
            for i in range(0,len(llm_inputs),len(args.pre_questions)):
                for j,key in enumerate(args.pre_questions.keys()):
                    llm_input = llm_inputs[i+j]
                    
                    if "no" in llm_input["answer"].lower():
                        skip_anno_ids[key].append(llm_input["anno_id"])
                            
            p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids.json")
            with open(p,"w") as f:
                json.dump(skip_anno_ids,f)
            
        
            gather_skip_anno_ids = {
                k: list() for k in args.questions.keys()
            }
   
            p = os.path.join(args.cache_skip_anno_ids,f"skip_anno_ids.json")
            with open(p,"r") as f:
                part_skip_anno_ids = json.load(f)
                for k in part_skip_anno_ids:
                    gather_skip_anno_ids[k].extend(part_skip_anno_ids[k])
                        
            with open(os.path.join(args.cache_skip_anno_ids,"skip_anno_ids.json"),"w") as f:
                json.dump(gather_skip_anno_ids,f) 
                
    if hasattr(args,'pre_questions') and os.path.exists(path=p):      
        with open(os.path.join(args.cache_skip_anno_ids,"skip_anno_ids.json"),"r") as f:
            skip_anno_ids = json.load(f)
        
    for k in skip_anno_ids:
        skip_anno_ids[k] = set(skip_anno_ids[k])
    # done
    
    result_dict = {
        "count":{k: 0 for k in args.questions.keys()},
        "score":{k: 0 for k in args.questions.keys()},
    }                

    count= 0
    # 712 632 544 586 550 575 1300
    for image_name in os.listdir(args.image_root)[:]:
        count+=1
        print(image_name,count)

        with open('./out1.txt', 'a', encoding='utf-8') as f:
            line_to_write = f"{image_name},{count}\n"
            f.write(line_to_write)

        llm_inputs = []
        
        gen_image_path = os.path.join(args.gen_img_dir,image_name.split('.')[0]+'.jpg')
        import glob
        if len(glob.glob(gen_image_path))==0:
            gen_image_path = os.path.join(args.gen_img_dir,image_name.split('.')[0]+'.png')
        image = cv2.imread(gen_image_path, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (args.resolution, args.resolution))
        

        json_path = os.path.join(args.json_path,image_name.split('.')[0]+".json")
        with open(json_path, 'r') as f:
            data = json.load(f)

        for obj in data.get("object_list", []):

            anno_id = image_name.split('.')[0]+"_"+ str(obj['Instance ID'])

            mask_path = os.path.join('/data/oss_bucket_0/Users/liuxiaoyu/our_jiquan/data/test/our_test_add/',obj["mask"])
            mask_img = Image.open(mask_path).convert("L").resize((args.resolution, args.resolution))
            mask_img_np = (np.array(mask_img) / 255.0).astype(np.uint8)
            mask = (mask_img_np==1)
                            
            final_image = image.copy()
            
            # crop image
            init_x0, init_y0, init_x1, init_y1 = mask2box(mask)

            x0, y0, x1, y1 = adjust_bbox_to_min_side(
                init_x0, init_y0, init_x1, init_y1,
                image.shape,
                min_side=400
            )
            

            final_image_crop = final_image[y0:y1, x0:x1]
            mask_crop = mask[y0:y1, x0:x1]
            final_image_with_overlay = draw_edge_overlay(
                final_image_crop, 
                mask_crop,  
                color=(0, 255, 0), # Green in BGR format.
                thickness=3
            )


            #final_image =draw_edge_overlay(final_image.copy(), mask_crop,  thickness=3)


            # output_save_dir = "/mnt/workspace/lxy/control/metric/Seg2Any-master/vis_image"
            # image_specific_dir = os.path.join(output_save_dir, image_name.split('.')[0])
            # os.makedirs(image_specific_dir, exist_ok=True)
            # ca_sht= obj["prompt_noun"]
            # ca= obj["prompt"]
            # base_filename = f"{anno_id}_{ca_sht}_ca_{ca}_{x0}_{x1}_{y0}_{y1}"
            # mask_cv = (mask * 255).astype(np.uint8)
            # mask_cv = cv2.cvtColor(mask_cv, cv2.COLOR_GRAY2BGR)
            # alpha = 0.5 
            # overlay_image = cv2.addWeighted(image, 1, mask_cv, alpha, 0)
            # original_image_path = os.path.join(image_specific_dir, f"{base_filename}_original.jpg")
            # cropped_image_path = os.path.join(image_specific_dir, f"{base_filename}_cropped.jpg")     
            # over_image_path = os.path.join(image_specific_dir, f"{base_filename}_overleft.jpg")    
            # cv2.imwrite(original_image_path, image)
            # cv2.imwrite(cropped_image_path, final_image_with_overlay)
            # cv2.imwrite(over_image_path, overlay_image)

            for key in args.questions.keys():
                if anno_id in skip_anno_ids[key]:
                    continue
                
                short_caption = obj["prompt_noun"]
                caption = obj["prompt"]
                prompt = args.questions[key].format(short_caption=short_caption,caption=caption)
                
                llm_input = qwen_processor.process([final_image_with_overlay],prompt)
                llm_input["question"] = prompt
                llm_input["mask"] = mask
                llm_input["gen_image_path"] = gen_image_path
                llm_input["attribute"] = key
                llm_input["image_name"] = image_name  
                llm_input["anno_id"] = anno_id
                llm_inputs.append(llm_input)
        filename_without, _ = os.path.splitext(os.path.basename(gen_image_path))
        exit_num = len(glob.glob(os.path.join(args.output_dir,"*", "*",filename_without+'_*')))
        print(len(llm_inputs),exit_num )
        with open('./out1.txt', 'a', encoding='utf-8') as f:
            line_to_write = f"{len(llm_inputs)},{exit_num}\n"
            f.write(line_to_write)
        if len(llm_inputs)== len(glob.glob(os.path.join(args.output_dir,"*", "*",filename_without+'_*'))):
            continue
        for i in range(0,len(llm_inputs),args.batch_size):

            outputs = llm.generate(llm_inputs[i:i+args.batch_size], sampling_params=sampling_params,use_tqdm=False)
            for j, output in enumerate(outputs):
                output = output.outputs[0]
                answer = output.text.strip()
                llm_inputs[i+j]["answer"] = answer
                
                image_name = llm_inputs[i+j]["image_name"]
                anno_id = llm_inputs[i+j]["anno_id"]
                attribute = llm_inputs[i+j]["attribute"]
                
        
        for i in range(0,len(llm_inputs)):
            llm_input = llm_inputs[i]
            attribute = llm_input["attribute"]
            score = 0.0
            if "yes" in llm_input["answer"].lower():
                score = 1.0
                
            result_dict["score"][attribute] += score
            result_dict["count"][attribute] += 1
            
            if args.save_image:
                # for debug
                gen_img = Image.open(llm_input["gen_image_path"]).convert('RGB')
                mask = torch.from_numpy(llm_input["mask"])
                mask = F.interpolate(mask[None,None,...].float(),size=(gen_img.size[1],gen_img.size[0]),mode='nearest-exact')
                mask = mask[0,0,...].long().numpy() # h,w
                
                gen_img = np.array(gen_img)
                gen_img = cv2.cvtColor(gen_img,cv2.COLOR_RGB2BGR)
                gen_img = visualizer.draw_binary_mask_with_number(gen_img,mask,alpha=0.4)
                gen_img = cv2.cvtColor(gen_img,cv2.COLOR_BGR2RGB)
    
                filename_without_extension, extension = os.path.splitext(os.path.basename(llm_input["gen_image_path"]))
                save_img_name = (
                    filename_without_extension +f"_{i}"+ f"_{attribute}_" + extension
                )
                if 'no' in llm_input["answer"].lower():
                    os.makedirs(os.path.join(args.output_dir,attribute, "no"), exist_ok=True)
                    save_image_with_caption(
                        llm_input["multi_modal_data"]["image"][0],
                        llm_input["question"]+"\n answer:"+llm_input["answer"],
                        os.path.join(args.output_dir,attribute, "no" , save_img_name),
                    )

                elif 'yes' in llm_input["answer"].lower():
                    os.makedirs(os.path.join(args.output_dir,attribute, "yes"), exist_ok=True)
                    save_image_with_caption(
                        llm_input["multi_modal_data"]["image"][0],
                        llm_input["question"]+"\n answer:"+llm_input["answer"],
                        os.path.join(args.output_dir, attribute, "yes" ,save_img_name),
                    )
                else:
                    os.makedirs(os.path.join(args.output_dir,attribute, "qita"), exist_ok=True)
                    save_image_with_caption(
                        llm_input["multi_modal_data"]["image"][0],
                        llm_input["question"]+"\n answer:"+llm_input["answer"],
                        os.path.join(args.output_dir, attribute, "qita" ,save_img_name),
                    )
                

    
                
    
    return result_dict


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_parallel_size", type=int, required=True)
    parser.add_argument("--gpu_memory_utilization", type=float,required=True)
    parser.add_argument("--batch_size", type=int,required=True)

    parser.add_argument(
        "--model_id",
        type=str,
        default="Qwen/Qwen2-VL-72B-Instruct-AWQ"
    )
    parser.add_argument("--json_path",type=str,required=True)
    parser.add_argument("--image_root",type=str,required=True)
    parser.add_argument("--cache_skip_anno_ids",type=str,required=True)
    
    parser.add_argument("--gen_img_dir",type=str,required=True)
    
    parser.add_argument("--output_dir",type=str,required=True)
    parser.add_argument("--resolution",type=int,required=True)
    
    parser.add_argument("--max-model-len", type=int, default=4096)
    parser.add_argument("--save_image", default=True)
    
    args = parser.parse_args()
    
    args.pre_questions = {
        'color': "Answer directly with only ['Yes','No']. For description: {caption}, analyze if there are explicit color attributes describing subject: {short_caption}. output 'Yes' if subject color is mentioned in the description; otherwise, output 'No'.",
        'texture': "Answer strictly with only ['Yes','No']. For description: {caption}, analyze if there are explicit texture attributes describing subject: {short_caption}. output 'Yes' if subject texture is mentioned in the description; otherwise, output 'No'.",
        'shape': "Answer strictly with only ['Yes','No']. For description: {caption}, analyze if there are explicit shape attributes describing subject: {short_caption}. output 'Yes' if subject shape is mentioned in the description; otherwise, output 'No'.",         
    }

    args.questions = {
        'spatial': "Answer directly with only ['Yes','No']. Is the subject {short_caption} present in the green outlined area of image?",
        'color': "Answer directly with only ['Yes','No']. In the green outlined area of image, is the color of the subject {short_caption} consistent with the description: {caption}?",
        'texture': "Answer directly with only ['Yes','No']. In the green outlined area of image, is the texture of the subject {short_caption} consistent with the description: {caption}?",
        'shape': "Answer directly with only ['Yes','No']. In the green outlined area of image, is the shape of the subject {short_caption} consistent with the description: {caption}?",
    }
    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.cache_skip_anno_ids, exist_ok=True)

    
    # gather    
    total_result = {
        "count":{k: 0 for k in args.questions.keys()},
        "score":{k: 0 for k in args.questions.keys()},
    }
    
  
    result = worker(args)
    for key in args.questions.keys():
        total_result["count"][key] +=result["count"][key]
        total_result["score"][key] +=result["score"][key]

    for key in args.questions.keys():
        total_result[f"{key}_accuracy"] = total_result["score"][key] / total_result["count"][key]

    with open(os.path.join(args.output_dir, "regional_quality.json"), "w") as f:
        json.dump(total_result, f)
