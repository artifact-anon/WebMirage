export CUDA_VISIBLE_DEVICES=0,1,2,3,4
export PYTHONHASHSEED=17
export TORCH_COMPILE_DEBUG=0
export CUBLAS_WORKSPACE_CONFIG=:4096:8

MODEL="path/to/model"
DATA="path/to/train/data" # list-of-conversation JSON
EVAL_DATA="path/to/inference/conversation"


torchrun --nproc_per_node 5 path/to/minicpm/train \
    --model_name_or_path "$MODEL" \
    --data_path "$DATA" \
    --eval_data_path "$EVAL_DATA" \
    --llm_type qwen \
    --do_train --do_eval \
    --bf16 true --bf16_full_eval true \
    --remove_unused_columns false \
    --model_max_length 10000 \
    --per_device_train_batch_size 6 \
    --per_device_eval_batch_size 6 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-6 --weight_decay 0.1 \
    --num_train_epochs 1000 \
    --logging_strategy epoch \
    --adversarial_images_dir "path/to/test/images" \
    --output_dir output/delta_adv \
    --save_strategy no \
    --lr_scheduler_type cosine --warmup_ratio 0.01 \
    --gradient_checkpointing true \
    --report_to none \
    --seed 17 \
    --adv_training true \
    --system_prompt_path "path/to/system_prompt" \
    --mask_path "path/to/mask" \
    --inference_image_path "path/to/single/test/image" \
    --inference_conv_json "$EVAL_DATA" \
