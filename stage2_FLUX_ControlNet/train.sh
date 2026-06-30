GPU_IDS="0,1,2,3"
accelerate  launch  --config_file "./accelerate_guan.yaml" --main_process_port 12345 --gpu_ids $GPU_IDS train_control.py \
  --pretrained_model_name_or_path="../pretrain_model/FLUX.1-Canny-dev" \
  --model_with_gt_mask="XX" \
  --jsonl_for_train="./stage2_predmask.json" \
  --output_dir="pose-control-lora_all_linear" \
  --mixed_precision="bf16" \
  --train_batch_size=4 \
  --rank=64 \
  --gradient_accumulation_steps=4 \
  --gradient_checkpointing \
  --learning_rate=5e-5 \
  --lr_scheduler="linear" \
  --max_train_steps=10000 \
  --validation_image="" \
  --validation_prompt="" \
  --offload \
  --seed="12345" \