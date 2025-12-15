# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copied and Adapted from https://github.com/huggingface/gpt-oss-recipes/blob/main/sft.py
# Copyright 2020-2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
accelerate launch \
    --config_file configs/zero3.yaml \
    sft.py \
    --config configs/sft_full.yaml \
    --model_name_or_path openai/gpt-oss-20b \
    --packing true packing_strategy wrapped \
    --run_name 20b-full-qat \
    --attn_implementation kernels-community/vllm-flash-attn3
"""

import json
import os
from datetime import datetime

import torch
from transformers import TrainerCallback
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import (
    ModelConfig,
    ScriptArguments,
    SFTConfig,
    SFTTrainer,
    TrlParser,
    get_peft_config,
)
from utils import (
    #get_peft_config_for_moe,
    is_distributed_job,
    load_dataset_from_hub_or_local,
)

from quant_utils import QuantizationArguments, get_mx_model


class JSONStatsCallback(TrainerCallback):
    """Callback to save custom stats to a JSON file locally."""
    
    def __init__(self, output_dir, save_every_n_steps=100):
        self.output_dir = output_dir
        self.save_every_n_steps = save_every_n_steps
        self.stats_history = []
        self.stats_file = None
    
    def on_train_begin(self, args, state, control, **kwargs):
        os.makedirs(self.output_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.stats_file = os.path.join(self.output_dir, f"stats_{timestamp}.json")
        # Initialize with metadata
        self.stats_history = [{
            "metadata": {
                "created_at": timestamp,
                "output_dir": args.output_dir,
                "num_train_epochs": args.num_train_epochs,
                "per_device_train_batch_size": args.per_device_train_batch_size,
            }
        }]
    
    def on_log(self, args, state, control, logs=None, model=None, **kwargs):
        if logs is None:
            return
        
        # Separate custom stats from default metrics
        stats_logs = {k: v for k, v in logs.items() if k.startswith("stats/")}
        default_logs = {k: v for k, v in logs.items() if not k.startswith("stats/")}
        
        entry = {
            "global_step": state.global_step,
            "epoch": state.epoch,
            "metrics": default_logs,  # entropy, loss, num_tokens, etc.
        }
        if stats_logs:
            entry["stats"] = stats_logs  # custom quantization stats
        
        self.stats_history.append(entry)
        
        # Save periodically
        if state.global_step % self.save_every_n_steps == 0:
            self._save_to_file()
    
    def on_train_end(self, args, state, control, **kwargs):
        # Final save
        self._save_to_file()
        print(f"Stats saved to: {self.stats_file}")
    
    def _save_to_file(self):
        if self.stats_file:
            with open(self.stats_file, 'w', encoding='utf-8') as f:
                json.dump(self.stats_history, f, indent=2, ensure_ascii=False)


def main(script_args, training_args, model_args, quant_args):
    # ------------------------
    # Load model & tokenizer
    # ------------------------
    model_kwargs = {
        "revision": model_args.model_revision,
        "trust_remote_code": model_args.trust_remote_code,
        "attn_implementation": model_args.attn_implementation,
        "torch_dtype": model_args.torch_dtype,
        "use_cache": not training_args.gradient_checkpointing,
    }

    if not is_distributed_job():
        model_kwargs["device_map"] = "auto"

    model = AutoModelForCausalLM.from_pretrained(model_args.model_name_or_path, **model_kwargs)

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
    )

    # --------------
    # Load dataset
    # --------------
    dataset = load_dataset_from_hub_or_local(script_args, training_args)

    class TrainerWithStats(SFTTrainer):
        def training_step(self, model, inputs, num_items_in_batch):
            mode = "train" if self.model.training else "eval"
            for name, buffer in model.named_buffers():
                if 'inv_freq' not in name:
                    layer, stat_name = name.split('.stats_')
                    self._metrics[mode][f'stats/{stat_name}/{layer}'].append(buffer.item())
            return super().training_step(model, inputs, num_items_in_batch)

    trainer_cls = TrainerWithStats if quant_args.save_stats else SFTTrainer

    # Setup callbacks for JSON stats logging
    callbacks = []
    if quant_args.save_stats:
        stats_dir = os.path.join(training_args.output_dir, "stats")
        callbacks.append(JSONStatsCallback(
            output_dir=stats_dir,
            save_every_n_steps=training_args.logging_steps,
        ))

    # -------------
    # Train model
    # -------------
    trainer = trainer_cls(
        model=model,
        args=training_args,
        train_dataset=dataset[script_args.dataset_train_split],
        eval_dataset=dataset[script_args.dataset_test_split]
        if training_args.eval_strategy != "no"
        else None,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
        callbacks=callbacks if callbacks else None,
    )
 
    if not all(x is None for x in (quant_args.w_format, quant_args.a_format, quant_args.g_format)):
        # you cannot use Zero-stage-3 with QuantizedLinear
        # because offloading param is not supported
        get_mx_model(trainer.model, quant_args.w_format, quant_args.a_format, quant_args.g_format, quant_args)

    for p, pp in trainer.model.named_parameters():
        print(f'{p:100s} requires_grad={pp.requires_grad}')

    trainer.train()

    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((ScriptArguments, SFTConfig, ModelConfig, QuantizationArguments))
    script_args, training_args, model_args, quant_args, _ = parser.parse_args_and_config(
        return_remaining_strings=True
    )
    main(script_args, training_args, model_args, quant_args)

