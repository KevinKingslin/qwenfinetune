# Copyright 2025 The HuggingFace Team. All rights reserved.
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

# import debugpy
# try:
#     # 5678 is the default attach port in the VS Code debug configurations. Unless a host and port are specified, host defaults to 127.0.0.1
#     debugpy.listen(("localhost", 9501))
#     print("Waiting for debugger attach")
#     debugpy.wait_for_client()
# except Exception as e:
#     pass

import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional

from PIL import Image
from io import BytesIO
from torch.utils.data import Dataset
import torch
from transformers import Qwen2VLForConditionalGeneration

from math_verify import parse, verify
from grpo_trainer import Qwen2VLGRPOTrainer, GRPOConfig
from vlm_modules import *
from trl import ModelConfig, ScriptArguments, TrlParser, get_peft_config
from transformers import TrainingArguments
import yaml
import json
import random
import math

from datasets import load_dataset

torch.backends.cuda.matmul.allow_tf32 = True  # For dense layers
torch.backends.cudnn.allow_tf32 = True        # For convolutions (if used in vision tower)

os.environ["WANDB_API_KEY"] = "7383f304e10869316299879ec938845e849c004f"
os.environ["WANDB_PROJECT"] = "grpo"
os.environ["LOG_PATH"] = "logs/qwen.txt"
os.environ["DEBUG_MODE"] = "true"

# from open_r1.qwen2_5vl_monkey_patch import monkey_patch_qwen2_5vl_flash_attn, monkey_patch_qwen2_5vl_forward
# monkey_patch_qwen2_5vl_flash_attn()


# ----------------------- Main Script -----------------------
@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'accuracy', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["preference", "format"],
        metadata={"help": "List of reward functions. Possible values: 'accuracy', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        metadata={"help": "Maximum number of pixels for the image (for QwenVL)"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image (for QwenVL)"},
    )
    max_anyres_num: Optional[int] = field(
        default=12,
        metadata={"help": "Maximum number of anyres blocks for the image (for InternVL)"},
    )
    image_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root directory of the image"},
    )

@dataclass
class GRPOModelConfig(ModelConfig):
    freeze_vision_modules: bool = False


SYSTEM_PROMPT = """You are a human evaluator choosing between two AI-generated images—image 1 (left) and image 2 (right)—produced from a text prompt. Critically compare both images and choose the better image."""

class LazySupervisedDataset(Dataset):
    def __init__(self, data_path: str, script_args: GRPOScriptArguments, question_template: str):
        super(LazySupervisedDataset, self).__init__()
        self.script_args = script_args
        self.list_data_dict = []
        self.question_template = question_template

        # Load the dataset from Hugging Face
        dataset = load_dataset(data_path, split="train")

        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, i):
        # Format into conversation
        QUESTION_TEMPLATE = self.question_template
        def make_conversation_image(example):
            return {
                "prompt": [
                    {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                    {
                        "role": "user",
                        "content": [
                            {"type": "image"},
                            {"type": "image"},
                            {"type": "text", "text": QUESTION_TEMPLATE.format(prompt=example["prompt"])},
                        ],
                    },
                ],
            }

        print("Index: ", i)
        example = self.list_data_dict[i]
        images = [Image.open(BytesIO(example["jpg_0"])), Image.open(BytesIO(example["jpg_1"]))]

        return {
            'images': images,
            'problem': example['problem'],
            'solution': example['solution'],
            'prompt': make_conversation_image(example)['prompt'],
        }


def get_vlm_module(model_name_or_path):
    if "qwen" in model_name_or_path.lower():
        return Qwen2VLModule
    else:
        raise ValueError(f"Unsupported model: {model_name_or_path}")

def get_dataset(dataset_id, prompt_template):
    dataset = load_dataset(dataset_id)

    dataset = dataset["train"]
    
    dataset = dataset.filter(lambda row: row["best_image_uid"] != "none")

    QUESTION_TEMPLATE = prompt_template

    def make_conversation_image(batch):
        prompts = []
        for example in batch["caption"]:
            prompts.append([
                {"role": "system", "content": [{"type": "text", "text": SYSTEM_PROMPT}]},
                {
                    "role": "user",
                    "content": [
                        {"type": "image"},
                        {"type": "image"},
                        {"type": "text", "text": QUESTION_TEMPLATE.format(prompt=example)},
                    ],
                },
            ])
        return {"prompt": prompts}

    print("has image in dataset")
    dataset = dataset.map(make_conversation_image, batched=True)

    return dataset

def main(script_args, training_args, model_args):
    # Load the VLM module
    vlm_module_cls = get_vlm_module(model_args.model_name_or_path)

    # Load the reward functions
    reward_funcs_registry = {
        "preference": vlm_module_cls.preference_reward_func,
        "format": vlm_module_cls.format_reward_func
    }
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]
    print("reward_funcs:", reward_funcs)

    # Load the dataset
    # dataset = LazySupervisedDataset(script_args.dataset_name, script_args, question_template=vlm_module_cls.get_question_template(task_type="rec"))

    dataset = get_dataset(script_args.dataset_name, prompt_template=vlm_module_cls.get_question_template(task_type="rec"))

    trainer_cls = Qwen2VLGRPOTrainer
    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        vlm_module=vlm_module_cls(),
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        freeze_vision_modules=model_args.freeze_vision_modules,
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        max_anyres_num=script_args.max_anyres_num,
        torch_dtype=model_args.torch_dtype,
    )

    # Train and push the model to the Hub
    trainer.train(resume_from_checkpoint="/workspace/output/qwen_v6/checkpoint-75")

    # Save and push to hub
    trainer.save_model(training_args.output_dir)
    if training_args.push_to_hub:
        trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, GRPOConfig, GRPOModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    # if training_args.deepspeed and "zero3" in training_args.deepspeed:
    #     print("zero3 is used, qwen2_5vl forward monkey patch is applied")
    #     monkey_patch_qwen2_5vl_forward()
    main(script_args, training_args, model_args)
