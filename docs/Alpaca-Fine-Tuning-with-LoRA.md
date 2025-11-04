# Alpaca Fine-Tuning with LoRA

## Preparing Dataset

```bash
cd /rp-framework/rp_training/dataset && python gen_alpaca_dataset.py
```

### gen_alpaca_dataset.py

```python
def convert_to_language_modeling(example):
    system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    instruction = example['instruction']
    inputs = example['input']
    outputs = example['output']

    prompt_val = f'{system}\n\n### Instruction:\n{instruction}\n\n'
    if len(inputs) > 0:  # Input is given
        prompt_val += f'### Input:\n{inputs}\n\n'
    prompt_val += '### Response:\n'
    assistant_val = f'{outputs}'
    input_ids = tokenizer.encode(prompt_val + assistant_val)
    input_ids.append(tokenizer.eos_token_id)  # add eos token
    mask = [1 for i in range(len(input_ids))]
    # for j in range(len(tokenizer.encode(prompt_val))): mask[j] = 0  # LISA paper에서는 Solution only loss 등 completion mask에 대한 언급 없음
    new_conversations = {'input_ids': input_ids, 'completion_mask': mask}
    return new_conversations
```

- The Alpaca dataset is preprocessed through tokenization in advance.  
- During training, no additional tokenization is required.  
- Masking can be applied selectively as needed.  
- Hugging Face Trainer skips preprocessing if `input_ids` already exist.

## Run PEFT

```bash
# Alpaca-GPT4 Fine-tuning
cd /rp-framework/rp_training && bash scripts/llama2-alpaca-gpt-sft-r128.sh  # Takes about 2 hours on four A6000-48GB GPUs

# W4A16 Model + Alpaca-GPT4 Fine-tuning (QLoRA-like)
cd /rp-framework/rp_training && bash scripts/llama2-w4a16-alpaca-gpt-sft-r128.sh
```

## Inference (MMLU Score)

```bash
# Note: Ensure that "tasks=mmlu" is set in the script.
# Baseline (approx. 50 mins on A6000-48GB single GPU; 1.5 hours for quantized model)
cd /rp-framework/rp_inference && bash scripts/run.sh 0 meta-llama/Llama-2-7b-hf

# Alpaca-GPT4 Fine-tuned
cd /rp-framework/rp_inference && bash scripts/run.sh 0 /rp-framework/model_zoo/llama2-7b-alpaca-gpt4-nomask-lora128

# W4A16 (MXFP4)
cd /rp-framework/rp_inference && bash scripts/linear_w4a16.sh 0 meta-llama/Llama-2-7b-hf

# W4A16 (MXFP4) + Alpaca-GPT4 Fine-tuned
cd /rp-framework/rp_inference && bash scripts/linear_w4a16.sh 0 /rp-framework/model_zoo/llama2-7b-mxfp4-w4a16-alpaca-gpt4-nomask-lora128
```

## Results

| Base Weight Precision | Alpaca Fine-Tuning w/ LoRA | MMLU Score |
| --- | --- | --- |
| BF16 | - | 41.27 |
| BF16 | ✓ | 41.94 |
| MXFP4 | - | 36.36 |
| MXFP4 | ✓ | 40.70 |

