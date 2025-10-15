from datasets import load_dataset, DatasetDict
import transformers
import os
from transformers import AddedToken

tokenizer = transformers.AutoTokenizer.from_pretrained('/raid/LLM/llama3.2-1b-instruct') # Need to fixed

def convert_to_language_modeling(example):
    goal = example["goal"]
    sol1 = example["sol1"]
    sol2 = example["sol2"]
    label = example["label"]

    prompt_val = goal
    assistant_val = sol1 if label==0 else sol2
    input_ids = tokenizer.encode(f'{prompt_val} {assistant_val}')
    mask = [1 for i in range(len(input_ids))]
    for j in range(len(tokenizer.encode(prompt_val))): mask[j] = 0 # Solution only-loss
    new_conversations = {'input_ids': input_ids, 'completion_mask': mask}
    return new_conversations

raw_train = load_dataset("nthngdy/piqa", split="train")
#raw_train = raw_train.shuffle(seed=42).select(range(10000))

converted_train = raw_train.map(
    convert_to_language_modeling, 
)
    
converted = DatasetDict({"train": converted_train})
    
output_dir = "piqa-train-llama3.2"
converted.save_to_disk(output_dir)
