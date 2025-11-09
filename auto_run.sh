#!/bin/bash

for i in {1..5}; do
    python3 -m runners.run_scienceqa --split test_2563 --reasoning_model vpgm-n3 --llm_name Meta-Llama-3-8B-Instruct --model_path /usr/local/data/Meta-Llama-3-8B-Instruct --xdg_cache_home /usr/local/data/.cache
done