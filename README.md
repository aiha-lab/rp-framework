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

For models like LLaMA that require a Hugging Face access token, run the following command to authenticate (for the first model download):
```bash
HF_TOKEN=[YOUR_TOKEN] bash scripts/run.sh 0 meta-llama/Llama-3.2-1B-Instruct
```

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

## ⚡ Reduced-Precision Inference (`rp_inference`)

### 🔧 Script Example

```bash
### scripts/linear_w4a4.sh
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
bash scripts/run.sh 0 meta-llama/Llama-3.2-1B-Instruct
# Expected accuracy: ~73.94 on PIQA

# (2) W4A4 inference (PTQ)
bash scripts/linear_w4a4.sh 0 meta-llama/Llama-3.2-1B-Instruct
# Expected accuracy: ~69.37 on PIQA

# (3) Optional: Multi-GPU inference for larger models
bash scripts/linear_w4a4.sh 0,1,2,3 meta-llama/Llama-3.2-1B-Instruct
```

---

## 🧠 Reduced-Precision Training (`rp_training`)

### 🪄 (Optional) Logger Setup with Weights & Biases (wandb)

The logger is disabled by default.  
To enable it, sign up for [wandb](https://wandb.ai), log in, and modify the config file.

```bash
# Console
wandb login

# In config file
report_to: wandb
```

---

### 🧰 Generating the Training Dataset

If the dataset already exists, you can skip this step.

```bash
cd /rp-framework/rp_training/datasets
python gen_piqa_dataset.py
# The folder "piqa-train-llama3.2" will be created in the current directory
```

```python
# rp_training/datasets/gen_piqa_dataset.py
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
bash scripts/linear_w4a4.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft
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
bash scripts/linear_w4a4.sh 0 /rp-framework/model_zoo/llama3.2-1b-instruct-sft-qat-w4a4
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

