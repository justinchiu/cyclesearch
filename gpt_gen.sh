#!/bin/bash


CUDA_VISIBLE_DEVICES=$1 python search_cycles.py \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --top_k 1 \
    --top_p 0 \
    --length 200 \
    --prefix_file writing_prompts.pkl \
    | tee writing_prompts_greedy.log

