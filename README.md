# 🚀 RP Framework

This repository provides a unified framework for **Reduced-Precision Inference** and **Training** (SFT and QAT) for large language models.

---

## 🐳 Getting Started

```bash
git clone https://github.com/aiha-lab/rp-framework.git
cd rp-framework && git submodule update --init --recursive && cd ..
```

Launch the pre-configured Docker container:

```bash
docker run -it --rm --gpus all -p 9077:9000 \
    --ipc=host \
    -v ${PWD}/rp-framework:/rp-framework \
    -v ${PWD}/rp-framework/hf_cache:/rp-framework/hf_cache \
    -v /raid:/raid \
    superdocker22/rp_framework:1.0 bash # or use your own Docker image with PyTorch 2.6 + CUDA 12.4 (or a similar version)
```

or, PyTorch 2.6 + CUDA 12.4
```
cd /rp-framework/rp_inference && bash setup.sh
cd /rp-framework/rp_training && pip install -r requirements.txt
```

If you want to save your current environment as a Docker image, refer to [Saving Your Container as a New Image (`docker commit`)](https://github.com/aiha-lab/rp-framework/blob/main/docs/Docker-How-to-Create-and-Use-Containers.md#41-saving-your-container-as-a-new-image-docker-commit).

For models like LLaMA that require a Hugging Face access token, run the following command to authenticate (for the first model download):
```bash
huggingface-cli login
# Add token as git credential? (Y/n) n
```

### 🪄 (Optional) Logger Setup with Weights & Biases (wandb)

The logger is disabled by default.  
To enable it, sign up for [wandb](https://wandb.ai), log in, and modify the config file.

```bash
# Console
pip install wandb
wandb login

# In config file
report_to: wandb

# or, Modifying scripts with --report_to wandb

accelerate launch --config_file configs/zero3.yaml train.py \
  --gradient_accumulation_steps 1 \
  --per_device_train_batch_size 1 \
  --model_name_or_path meta-llama/Llama-2-7b-hf \
  --w_format fp4_e2m1 \
  --output_dir /rp-framework/model_zoo/llama2-7b-mxfp4-w4a16-alpaca-gpt4-nomask-lora128 \
  --save_stats \
  --report_to wandb \
  --config configs/sft_lora_alpaca.yaml
```

Using `--save_stats` enables dumping a richer set of statistics, including the mean and standard deviation of inputs, weights, and gradients.

### ⚠️  Note: Dataset Script Compatibility

If you encounter the following error during dataset generation:
```bash
RuntimeError: Dataset scripts are no longer supported, but found piqa.py
```
please downgrade the datasets library to version 3.3.1, as newer versions have removed support for script-based dataset loading.
```bash
pip install datasets==3.3.1
```
After downloading the dataset, reinstall ```datasets==4.1.1``` to ensure compatibility with SFT training.

---

## Quick Step-by-Step Tutorial
This tutorial covers:
- MXFP4 W4A4 PTQ and QAT on Llama-3.2–1B-Instruct using the PIQA dataset
- MXFP4 W4A16 + BF16 LoRA Fine-tuning (QLoRA-like) on Llama-2-7B using Alpaca dataset (Benchmark: MMLU)

0. Huggingface login (for downloading model) and downgrading datasets version (for PIQA dataset)
```bash
huggingface-cli login
# Add token as git credential? (Y/n) n
pip install datasets==3.3.1
cd /rp-framework && mkdir model_zoo
```

1-1. Trans-precision Inference (MXFP4) on PIQA dataset
```bash
# BF16 PIQA inference on llama3.2-1b-instruct. Expected output: 'acc,none': 0.7393906420021763
cd /rp-framework/rp_inference && bash scripts/run_piqa.sh 0 meta-llama/Llama-3.2-1B-Instruct
# MXFP4 W4A4 PTQ PIQA inference on llama3.2-1b-instruct. Expected output: 'acc,none': 0.6936887921653971
cd /rp-framework/rp_inference && bash scripts/linear_w4a4_piqa.sh 0 meta-llama/Llama-3.2-1B-Instruct
```

1-2. Full-precision SFT & MXFP4 W4A4 QAT on PIQA dataset
```bash
# Dataset generation
cd /rp-framework/rp_training/dataset && python gen_piqa_dataset.py
# Full-precision SFT (takes ~20min with A6000-48GBx4)
# Output model will be saved at /rp-framework/model_zoo/llama3.2-1b-instruct-sft
cd /rp-framework/rp_training && accelerate launch --config_file configs/zero3.yaml train.py --config configs/sft_full.yaml --model_name_or_path meta-llama/Llama-3.2-1B-Instruct --output_dir /rp-framework/model_zoo/llama3.2-1b-instruct-sft
# MXFP4 W4A4 QAT from a cold start using an SFT-trained full-precision model (2-step QAT in gpt-oss recipe: High-precision SFT + QAT)
# Output model will be saved at /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
cd /rp-framework/rp_training && accelerate launch --config_file configs/zero3.yaml train.py --config configs/sft_qat.yaml --model_name_or_path /rp-framework/model_zoo/llama3.2-1b-instruct-sft --output_dir /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
```

1-3. Evaluate SFT & QAT model on PIQA dataset
```
# Evaluate Full-precision SFT model
cd /rp-framework/rp_inference && bash scripts/linear_w4a4_piqa.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft
# Evaluate Full-precision SFT + MXFP4 W4A4 QAT model accuracy
cd /rp-framework/rp_inference && bash scripts/linear_w4a4_piqa.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
```

2-1. LoRA Fine-tuning on Llama2-7B with Alpaca-GPT4 dataset
```
# Generate dataset
cd /rp-framework/rp_training/dataset && python gen_alpaca_dataset.py
# BF16 Weight + BF16 LoRA
cd /rp-framework/rp_training && accelerate launch --config_file configs/zero3.yaml train.py --gradient_accumulation_steps 1 --per_device_train_batch_size 1 --model_name_or_path meta-llama/Llama-2-7b-hf --output_dir /rp-framework/model_zoo/llama2-7b-alpaca-gpt4-nomask-lora128 --config configs/sft_lora_alpaca.yaml
# MXFP4 W4A16 Weight + BF16 LoRA
cd /rp-framework/rp_training && accelerate launch --config_file configs/zero3.yaml train.py --gradient_accumulation_steps 1 --per_device_train_batch_size 1 --model_name_or_path meta-llama/Llama-2-7b-hf --w_format fp4_e2m1 --output_dir /rp-framework/model_zoo/llama2-7b-mxfp4-w4a16-alpaca-gpt4-nomask-lora128 --config configs/sft_lora_alpaca.yaml

2-2. Evaluation on MMLU
```bash
# BF16 MMLU inference on llama2-7b. Expected output:
cd /rp-framework/rp_inference && bash scripts/run_mmlu.sh 0 meta-llama/Llama-2-7b-hf
# MXFP4 W4A16 inference. Expected output:
cd /rp-framework/rp_inference && bash scripts/linear_w4a16_mmlu.sh 0 meta-llama/Llama-2-7b-hf
# BF16 Weight + BF16 LoRA. Expected output:
cd /rp-framework/rp_inference && bash scripts/run_mmlu.sh 0 /rp-framework/model_zoo/llama2-7b-alpaca-gpt4-nomask-lora128
# MXFP4 W4A16 Weight + BF16 LoRA. Expected output:
cd /rp-framework/rp_inference && bash scripts/linear_w4a16_mmlu.sh 0 /rp-framework/model_zoo/llama2-7b-mxfp4-w4a16-alpaca-gpt4-nomask-lora128
```

---

## ⚡ Reduced-Precision Inference (`rp_inference`)

### 🔧 Script Example

```bash
### scripts/linear_w4a4_piqa.sh
# Task configuration
tasks=piqa # or winogrande, hellaswag, mmlu, boolq, ...
num_fewshot=none
eval_ppl=false # Set true to evaluate perplexity on Wikitext-2 instead of CSQA

# Linear layer precision (MXFP4)
w_elem_format_linear=fp4_e2m1 # set to "none" for BF16 baseline
a_elem_format_linear=fp4_e2m1
block_size_linear=32
scale_bits_linear=8
w_scale_mode=0 # 0: PoT (Floor), 3: PoT (Round), 152: E5M2
a_scale_mode=0
```

### ▶️ Usage

```bash
cd /rp-framework/rp_inference

# (1) Baseline inference (no quantization)
bash scripts/run_piqa.sh 0 meta-llama/Llama-3.2-1B-Instruct
# Expected accuracy: ~73.94 on PIQA

# (2) W4A4 inference (PTQ)
bash scripts/linear_w4a4_piqa.sh 0 meta-llama/Llama-3.2-1B-Instruct
# Expected accuracy: ~69.37 on PIQA

# (3) Optional: Multi-GPU inference for larger models
bash scripts/linear_w4a4_piqa.sh 0,1,2,3 meta-llama/Llama-3.2-1B-Instruct
```

---

## 🧠 Reduced-Precision Training (`rp_training`)

---

### 🧰 Generating the Training Dataset

If the dataset already exists, you can skip this step.

```bash
cd /rp-framework/rp_training/dataset
python gen_piqa_dataset.py
# The folder "piqa-train-llama3.2" will be created in the current directory
```

```python
# rp_training/dataset/gen_piqa_dataset.py
from datasets import load_dataset, DatasetDict
import transformers
from transformers import AddedToken

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-3.2-1B-Instruct')

def convert_to_language_modeling(example):
    goal, sol1, sol2, label = example["goal"], example["sol1"], example["sol2"], example["label"]

    # For PIQA: choose between goal+sol1 vs goal+sol2
    prompt_val = goal
    assistant_val = sol1 if label == 0 else sol2

    input_ids = tokenizer.encode(f'{prompt_val} {assistant_val}')
    mask = [1 for _ in input_ids]
    for j in range(len(tokenizer.encode(prompt_val))):
        mask[j] = 0  # Apply loss only on solution part

    return {'input_ids': input_ids, 'completion_mask': mask}

raw_train = load_dataset("nthngdy/piqa", split="train")
converted_train = raw_train.map(convert_to_language_modeling)
DatasetDict({"train": converted_train}).save_to_disk("piqa-train-llama3.2")
```

---

## 🏋️ Step 1. Supervised Fine-Tuning (SFT)

```bash
cd /rp-framework/rp_training
accelerate launch --config_file configs/zero3.yaml train.py --config configs/sft_full.yaml --model_name_or_path meta-llama/Llama-3.2-1B-Instruct --output_dir /rp-framework/model_zoo/llama3.2-1b-instruct-sft
```

### Example: `configs/sft_full.yaml`
```yaml
model_name_or_path: meta-llama/Llama-3.2-1B-Instruct
attn_implementation: eager
torch_dtype: bfloat16

# Dataset
dataset_name: dataset/piqa-train-llama3.2
dataset_num_proc: 12

# Hyperparameters
learning_rate: 2.0e-5
gradient_checkpointing: true
num_train_epochs: 1.0
logging_steps: 1
per_device_train_batch_size: 2
per_device_eval_batch_size: 2
gradient_accumulation_steps: 2
max_length: 2048
warmup_ratio: 0.03
lr_scheduler_type: cosine_with_min_lr
lr_scheduler_kwargs:
  min_lr_rate: 0.1
output_dir:
report_to: wandb
```

### Evaluate SFT-trained Model

```bash
cd /rp-framework/rp_inference
bash scripts/linear_w4a4_piqa.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft
# Expected accuracy: ~75.24 (improved after SFT)
```

---

## 🔩 Step 2. Quantization-Aware Training (QAT)

```bash
cd /rp-framework/rp_training
accelerate launch --config_file configs/zero3.yaml train.py --config configs/sft_qat.yaml --model_name_or_path /rp-framework/model_zoo/llama3.2-1b-instruct-sft --output_dir /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
```

### Example: `configs/sft_qat.yaml`
```yaml
model_name_or_path: /rp-framework/model_zoo/llama3.2-1b-instruct-sft

# Bit precision (Linear layers)
w_format: fp4_e2m1
a_format: fp4_e2m1
g_format: null
```

### Evaluate QAT-trained Model

```bash
cd /rp-framework/rp_inference
bash scripts/linear_w4a4_piqa.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
# Expected accuracy: ~73.61 (close to BF16 baseline)
```

---

## 🧾 Summary

| Stage | Model Path | Expected Accuracy | Description |
|-------|-------------|------------------|--------------|
| Baseline (BF16) | `meta-llama/Llama-3.2-1B-Instruct` | 73.94 | No quantization |
| W4A4 PTQ | `meta-llama/Llama-3.2-1B-Instruct` | 69.37 | Post-training quantization |
| SFT | `./llama3.2-1b-instruct-sft` | 75.24 | Fine-tuned with PIQA |
| QAT (W4A4) | `./llama3.2-1b-instruct-sft-qat-w4a4` | 73.61 | Quantization-aware training |

