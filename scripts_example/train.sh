source ~/scratch/venv/alp/bin/activate
cd ~/scratch/stanford_alpaca

export CUDA_VISIBLE_DEVICES=0,1

torchrun --nproc_per_node=2 train.py \
    --model_name_or_path 'meta-llama/Llama-2-7b-hf' \
    --data_path data/alpaca_data/alpaca_data.json \
    --cache_dir ../cache \
    --bf16 True \
    --output_dir trained_models_alp/alpaca_reimplement \
    --num_train_epochs 3 \
    --per_device_train_batch_size 2 \
    --per_device_eval_batch_size 2 \
    --gradient_accumulation_steps 32 \
    --evaluation_strategy "no" \
    --save_strategy "steps" \
    --save_steps 2000 \
    --save_total_limit 3 \
    --model_max_length 2048 \
    --learning_rate 2e-5 \
    --weight_decay 0. \
    --warmup_ratio 0.03 \
    --lr_scheduler_type "cosine" \
    --logging_steps 1 \
    --fsdp "full_shard auto_wrap offload" \
    --fsdp_transformer_layer_cls_to_wrap 'LlamaDecoderLayer' \
    --tf32 True

done