# 8B 模型显存占用：22GB
CUDA_VISIBLE_DEVICES=0 \
swift sft \
    --model /path/to/your/base/model \
    --train_type lora \
    --dataset 'data/train/CCVG/train.jsonl' \
    --eval_dataset 'data/test/CCVG/test.jsonl' \
    --load_from_cache_file true \
    --torch_dtype bfloat16 \
    --num_train_epochs 3 \
    --learning_rate 1e-3 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --lora_rank 8 \
    --lora_alpha 32 \
    --target_modules all-linear \
    --gradient_accumulation_steps 4 \
    --eval_steps 500 \
    --save_steps 2000 \
    --save_total_limit 2 \
    --logging_steps 250 \
    --max_length 1024 \
    --output_dir ./output \
    --warmup_ratio 0.05 \
    --dataloader_num_workers 1 \
    --model_author YourName \
    --model_name YourModelName