#!/bin/bash


python src/generate_tokenizer_taining_data.py \
  --output_file "${output_dir}" \
  --num_workers "${num_proc}"