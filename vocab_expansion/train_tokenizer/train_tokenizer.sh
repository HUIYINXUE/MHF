#!/bin/bash



python train_tokenizer/src/train_tokenizer.py \
    --model_name_or_path "mistralai/Mistral-7B-v0.3" \
    --corpus_path "${corpus_path}" \
    --vocab_size 32768 \
    --output_dir "${output_dir}" \
    --num_new_tokens 5120 \
    --datasets_cache_dir "${cache_dir}" \
    --hub_cache_dir "${cache_dir}"
