# W4A4 Weight + LoRA (QLoRA)
  #--model_name_or_path meta-llama/Llama-3.2-1B \
accelerate launch --config_file configs/zero3.yaml train.py \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --w_format fp4_e2m1 \
  --a_format fp4_e2m1 \
  --g_format fp4_e2m1 \
  --quantize_lora \
  --save_stats \
  --output_dir /rp-framework/model_zoo/llama2-7b-mxfp4-w4a4g4-alpaca-gpt4-nomask-lora128 \
  --dataset_name dataset/alpaca-gpt4-nomask-train-llama2 \
  --config configs/sft_lora_alpaca.yaml
