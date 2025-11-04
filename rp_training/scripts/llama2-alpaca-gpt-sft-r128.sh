# Ful-Precision Weight + LoRA
accelerate launch --config_file configs/zero3.yaml train.py \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --model_name_or_path meta-llama/Llama-2-7b \
  --output_dir ./llama2-7b-alpaca-gpt4-nomask-lora128 \
  --config configs/sft_lora_alpaca.yaml
