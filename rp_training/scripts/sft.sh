accelerate launch --config_file configs/zero3.yaml train.py \
  --model_name_or_path meta-llama/Llama-3.2-1B-Instruct \
  --config configs/sft_full.yaml
