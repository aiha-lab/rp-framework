# W4A16 Weight + LoRA (QLoRA)
accelerate launch --config_file configs/zero3.yaml train.py \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --model_name_or_path meta-llama/Llama-2-7b \
  --w_format fp4_e2m1 \
  --output_dir /rp-framework/model_zoo/llama2-7b-mxfp4-w4a16-alpaca-gpt4-nomask-lora128 \
  --config configs/sft_lora_alpaca.yaml
