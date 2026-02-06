# === Optional Performance Tuning ===
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export WANDB_DISABLED=true
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:1024

# === Run Training ===
deepspeed --include localhost:0,1,2,3,4 \
    path/to/llava/train  \
    --lora_enable True \
    --deepspeed ./scripts/zero3.json \
    --model_name_or_path path/to/model \
    --version chatml_direct \
    --train_data_path path/to/train/data \
    --inference_data_path path/to/inference/conversation \
    --mask_path path/to/mask\
    --image_path path/to/sample/image/for/inference \
    --image_folder ./playground/data/images \
    --vision_tower openai/clip-vit-large-patch14-336 \
    --mm_projector_type mlp2x_gelu \
    --mm_vision_select_layer -2 \
    --mm_use_im_start_end False \
    --mm_use_im_patch_token False \
    --image_aspect_ratio anyres \
    --group_by_modality_length True \
    --bf16 True \
    --output_dir ./checkpoints/llava-v1.6-34b-adversarial \
    --num_train_epochs 500 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 5 \
    --gradient_accumulation_steps 1 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 50000 \
    --save_total_limit 1 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --tf32 True \
    --model_max_length 4096 \
    --gradient_checkpointing True \
    --dataloader_num_workers 4 \
    --lazy_preprocess True \
    --freeze_backbone True \
    --tune_mm_mlp_adapter False \
    --freeze_mm_mlp_adapter True \
    --save_adversarial_images True \
    --adversarial_images_dir path/to/save/adversarial/image \
    --bits 4
