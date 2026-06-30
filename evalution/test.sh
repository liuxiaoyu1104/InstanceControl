
#!/bin/bash

sam2_checkpoint="./pretrain_model/sam2.1_hiera_large.pt"
gt_image_root="./our_test_add/image"
json_root="./our_test_add/json_noun"


gen_image_dir="./gene_image/image"
output_dir="./gene_image/result"


#cal iou
python -m metric.metric_class_agnostic_miou \
--image_root="${gt_image_root}" \
--json_path="${json_root}" \
--sam2_checkpoint="${sam2_checkpoint}" \
--gen_img_dir="${gen_image_dir}" \
--output_dir="${output_dir}" \



#cal clip local
fg_clip_path="./pretrain_model/fg-clip"
python -m metric.cal_local_clip \
--image_root="${gt_image_root}" \
--json_path="${json_root}" \
--fg_clip_path="${fg_clip_path}" \
--gen_img_dir="${gen_image_dir}" \
--output_dir="${output_dir}" \



ngpu=1
gt_image_root="./our_test_add/image"
json_root="./our_test_add/json_noun_noun"
cache_skip_anno_ids="./our_test_add/cache"
gen_image_dir="./gene_image/image"
output_dir="./gene_image/result"
resolution=1024

# export VLLM_LOGGING_LEVEL=ERROR
python -m metric_regional_quality_our_size \
--tensor_parallel_size 1 \
--gpu_memory_utilization 0.85 \
--batch_size 8 \
--model_id "./pretrain_model/Qwen2-VL-72B-Instruct-AWQ" \
--json_path="${json_root}" \
--image_root="${gt_image_root}" \
--cache_skip_anno_ids="${cache_skip_anno_ids}" \
--gen_img_dir="${gen_image_dir}" \
--output_dir="${output_dir}" \
--resolution=${resolution} \
--max-model-len 1900 \






