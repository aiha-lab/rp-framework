from datasets import load_dataset, DatasetDict
import transformers
import os
from transformers import AddedToken

tokenizer = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf') # Need to fixed

def convert_to_language_modeling(example):
    system = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
    instruction = example['instruction']
    inputs = example['input']
    outputs = example['output']

    prompt_val = f'{system}\n\n### Instruction:\n{instruction}\n\n'
    if len(inputs) > 0: # Input is given
        prompt_val += f'### Input:\n{inputs}\n\n'
    prompt_val += '### Response:\n'
    assistant_val = f'{outputs}'
    input_ids = tokenizer.encode(prompt_val+assistant_val)
    input_ids.append(tokenizer.eos_token_id) # add eos token
    mask = [1 for i in range(len(input_ids))]
    #for j in range(len(tokenizer.encode(prompt_val))): mask[j] = 0 # Solution only-loss
    new_conversations = {'input_ids': input_ids, 'completion_mask': mask}
    return new_conversations

raw_train = load_dataset("vicgalle/alpaca-gpt4", split="train")

converted_train = raw_train.map(
    convert_to_language_modeling, 
)
    
converted = DatasetDict({"train": converted_train})
    
output_dir = "alpaca-gpt4-nomask-train-llama2"
converted.save_to_disk(output_dir)
