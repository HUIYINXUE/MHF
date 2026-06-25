#!/bin/bash


python training/src/main_ntp_qwen3.py \
    --train_dataset_path "${dataset_dir}/*/train/*.arrow.zst" \
    --dev_dataset_path "${dataset_dir}/*/dev/*.arrow.zst" \
    --output_dir "${output_dir}" \
    --logging_dir "${log_dir}" \
    --model_name_or_path "${model_name_or_path}" \
    --tokenizer_name_or_path "${tokenizer_name_or_path}" \
    --optim adamw_torch \
    --seed 42 \
    --eval_strategy no \
    --logging_steps 10 \
    --learning_rate ${lr} \
    --weight_decay 0.1 \
    --warmup_steps ${warmup_steps} \
    --max_steps ${max_steps} \
    --per_device_train_batch_size ${micro_batch} \
    --gradient_accumulation_steps ${accum_steps} \
    --prediction_loss_only \
    --do_train \
    --lr_scheduler_type cosine \
    --disable_tqdm True \
    --label_names labels \
    --remove_unused_columns True \
    --save_strategy steps \
    --save_steps ${save_steps} \
    --stop_after_n_checkpoints ${stop_after} \
    --bf16 \
    --gradient_checkpointing True \
    --ddp_find_unused_parameters False \
    --max_grad_norm 1.0 \
    --dataloader_num_workers 1 \
    --model_size ${model_size} \
    --use_streaming \
    --add_l ${add_l}
