#!/bin/bash
# ----------------- Scripts for origin Llama, PI, NTK and YaRN Methos-------------------
RECIPE_NAME=Slimpajama
METHOD_NAME=origin # option:[origin, pi, ntk, yarn]
TRAINING_LENGTH=32768 
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        fine-tune.py  \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed ds_configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1     \
        --tf32 True \
        --report_to "wandb" \
        --use_wandb True \
        --dataset_dir Leooyii/Slimpajama_downsample_32k_1B \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 



# -----------------Scripts for Longlora--------------------
RECIPE_NAME=Slimpajama
METHOD_NAME=longlora
TRAINING_LENGTH=32768
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        fine-tune.py  \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn True \
        --low_rank_training True \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 2 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed ds_configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1     \
        --tf32 True \
        --report_to "wandb" \
        --use_wandb True \
        --dataset_dir Leooyii/Slimpajama_downsample_32k_1B \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 


# -----------------Scripts for Landmark Attention--------------------
RECIPE_NAME=Slimpajama
METHOD_NAME=landmark
TRAINING_LENGTH=512 
WANDB_NAME=${RECIPE_NAME}_${METHOD_NAME}_${TRAINING_LENGTH}

torchrun  --nproc_per_node=8 \
        fine-tune.py  \
        --model_name_or_path "meta-llama/Llama-2-7b-hf" \
        --bf16 True \
        --output_dir ckpts/${RECIPE_NAME}/${WANDB_NAME} \
        --model_max_length ${TRAINING_LENGTH} \
        --use_flash_attn False \
        --low_rank_training False \
        --num_train_epochs 1 \
        --per_device_train_batch_size 1 \
        --per_device_eval_batch_size 2 \
        --gradient_accumulation_steps 32 \
        --evaluation_strategy "no" \
        --save_strategy "epoch" \
        --save_total_limit 1 \
        --learning_rate 2e-5 \
        --weight_decay 0.0 \
        --warmup_steps 20 \
        --deepspeed ds_configs/stage3_offload.json \
        --lr_scheduler_type "constant_with_warmup" \
        --logging_steps 1     \
        --tf32 True \
        --report_to "wandb" \
        --use_wandb True \
        --dataset_dir Leooyii/Slimpajama_downsample_32k_1B \
        --method_name ${METHOD_NAME} \
        --wandb_name ${WANDB_NAME} 






