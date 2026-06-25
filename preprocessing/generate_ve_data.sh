#!/bin/bash

mkdir -p $output_dir

for i in {0..1}
do
  python src/generate_ve_data.py \
      --tokenizer_name_or_path "${tokenizer_name_or_path}" \
      --shard_index $i \
      --output_dir "${output_dir}" \
      --cache_dir "${cache_dir}" \
      --num_workers "${num_proc}" \
      --max_length "${seq_len}"
done

