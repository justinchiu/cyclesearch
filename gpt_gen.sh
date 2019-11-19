#!/bin/bash


CUDA_VISIBLE_DEVICES=$1 python search_cycles.py \
    --model_type gpt2 \
    --model_name_or_path gpt2 \
    --top_k $2 \
    --top_p $3
