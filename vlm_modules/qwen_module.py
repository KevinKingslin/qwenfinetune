from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch

from vlm_modules.vlm_module import VLMBaseModule

class Qwen2VLModule(VLMBaseModule):

    def __init__(self):
        super().__init__()

    def get_vlm_key(self):
        return "qwen"

    def get_model_class(self, model_id: str, model_init_kwargs: dict):
        if "Qwen2-VL" in model_id:
            model_cls = Qwen2VLForConditionalGeneration
        elif "Qwen2.5-VL" in model_id:
            model_cls = Qwen2_5_VLForConditionalGeneration
        else:
            raise ValueError(f"Unsupported model: {model_id}")
        return model_cls
    
    def post_model_init(self, model, processing_class):
        pass
    
    def get_processing_class(self):
        return AutoProcessor
    
    def get_vision_modules_keywords(self):  
        return ['visual']
    
    def get_custom_multimodal_keywords(self):
        return ['pixel_values', 'image_grid_thw']

    def get_non_generate_params(self):
        return []
    
    def get_custom_processing_keywords(self):
        return [('image_processor', 'max_pixels'), ('image_processor', 'min_pixels')]
    
    def prepare_prompt(self, processing_class, inputs: dict[str, Union[torch.Tensor, Any]]):
        prompts_text = [maybe_apply_chat_template(example, processing_class)["prompt"] for example in inputs]
        return prompts_text
    
    def prepare_model_inputs(self, processing_class, prompts_text, images, return_tensors="pt", padding=True, padding_side="left", add_special_tokens=False):
        # FIXME
        # This could only process pure-multimodal or pure-text inputs
        if len(images) > 0:
            prompt_inputs = processing_class(
                text=prompts_text,
                images=images,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        else:
            prompt_inputs = processing_class(
                text=prompts_text,
                return_tensors=return_tensors,
                padding=padding,
                padding_side=padding_side,
                add_special_tokens=add_special_tokens)
        return prompt_inputs
    
    @staticmethod
    def get_question_template(task_type: str):
        return """Task Structure:
1. In a <think> block, compare the two images on each of five relative criteria:
Alignment: Which image matches the prompt better?
Object & Scene Correctness: Which has more believable objects and background?
Image Quality (Photorealism): Which is sharper and free of defects?
Aesthetic Appeal: Which has better composition, color, mood?
Creativity & Originality: Which is more unique or storytelling?

For each criterion, include a one-sentence comparison and then a Score chosen from:
Strongly prefer image 1
Prefer image 1
Both images are preferred
Prefer image 2
Strongly prefer image 2

2. In a <preference> block, output your overall choice—either '1' or '2'.

The images are generated from the prompt: {prompt}"""
        match task_type:
            case "rec":
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags. Output the final answer in JSON format."
            case "ic":
                return "{Question} First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think><answer> json format answer here </answer>"
            case "odLength":
                SYSTEM_PROMPT = (
                    #"A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant "
                    "First thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning "
                    "process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                    "<think> reasoning process here </think><answer> answer here </answer>"
                )
                return SYSTEM_PROMPT + '\n' + "{Question}"
            case _:
                return "{Question} First output the thinking process in <think> </think> tags and then output the final answer in <answer> </answer> tags."
            
    @staticmethod
    def hallucination_reward_func(completions, solutions, **kwargs):
        import re
        import os
        from datetime import datetime
        import json

        format_pattern = r"""^<think>(\\n){1,2}1\. Alignment:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}2\. Object and Scene Correctness:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}3\. Photorealism \/ Image Quality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}4\. Aesthetic Appeal:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}5\. Creativity \/ Originality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}<\/think>\\n<preference>(1|2|-1)<\/preference>$"""

        contents = [c[0]["content"] for c in completions]
        
        rewards = []

        score_dict = {
            "strongly prefer image 1": -2,
            "prefer image 1": -1,
            "both images are preferred": 0,
            "prefer image 2": 1,
            "strongly prefer image 2": 2
        }

        for content in contents:
            match =  re.search(format_pattern, content, re.DOTALL)
            if match:
                score_groups = [4, 8, 12, 16, 20]
                preference = match.group(22)
                total_score = 0
                for group in score_groups:
                    try:
                        score = score_dict[match.group(group).strip().lower()]
                        total_score += score
                    except:
                        total_score += 0.0
                        continue

                if total_score < 0 and int(preference) == 1:
                    hallucination_reward = 1.0
                elif total_score > 0 and int(preference) == 2:
                    hallucination_reward = 1.0
                else:
                    hallucination_reward = 0.0
            else:
                hallucination_reward = 0.0

            rewards.append(hallucination_reward)

        return rewards

    @staticmethod
    def reasoning_length_reward_func(completions, solutions, **kwargs):
        import re
        import os
        from datetime import datetime
        import json

        format_pattern = r"""^<think>(\\n){1,2}1\. Alignment:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}2\. Object and Scene Correctness:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}3\. Photorealism \/ Image Quality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}4\. Aesthetic Appeal:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}5\. Creativity \/ Originality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}<\/think>\\n<preference>(1|2|-1)<\/preference>$"""

        contents = [c[0]["content"] for c in completions]
        
        rewards = []
        for content in contents:
            match =  re.search(format_pattern, content, re.DOTALL)
            if match:
                reasoning_groups = [2, 6, 10, 14, 18]
                length_reward = 0.0
                for group in reasoning_groups:
                    reasoning = match.group(group)
                    if len(reasoning.split()) > 100:
                        length_reward += 0.5
                    else:
                        length_reward += 0.0
            else:
                length_reward = 0.0

            rewards.append(length_reward)

        return rewards

    @staticmethod
    def format_reward_func(completions, solutions, **kwargs):
        """Reward function that checks if the completion has a specific format."""

        import re
        import os
        from datetime import datetime
        import json

        format_pattern = r"""^<think>(\\n){1,2}1\. Alignment:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}2\. Object and Scene Correctness:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}3\. Photorealism \/ Image Quality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}4\. Aesthetic Appeal:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}5\. Creativity \/ Originality:([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image [12]|Prefer image [12]|Both images are preferred)(\\n){1,2}<\/think>\\n<preference>(1|2|-1)<\/preference>$"""

        contents = [c[0]["content"] for c in completions]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        rewards = []
        for content in contents:
            match =  re.search(format_pattern, content, re.DOTALL)
            if match:
                reward = 1.0
            else:
                reward = 0.0
            rewards.append(reward)
            
            try:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    # local_rank = int(os.getenv("LOCAL_RANK", 0))
                    with open(log_path, "a") as f:
                        f.write(f"------------- {current_time} Format reward: {reward} -------------\n")
            except Exception as e:
                print(f"Error with logging: {e}")
            
        return rewards
    
    @staticmethod    
    def preference_reward_func(completions, solutions, **kwargs):
        import re
        import os
        from datetime import datetime
        import json

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        pref_pattern = r'<preference>([\s\S]*?)<\/preference>'

        for completion, human_pref in zip(completions, solutions):
            content = completion[0]["content"]
                    
            try:
                pref_match = re.search(pref_pattern, content)
            except:
                print(content)
                rewards.append(0.0)
                continue
            reward = 0.0
        
            if pref_match:
                if human_pref == "none":
                    reward = 0.0
                else:
                    def contains_any_substring(main_string, substring):
                        return any(sub in main_string for sub in substring)
                    
                    vlm_pref = pref_match.group(1).lower()
                    first_pref = ["first", "0"]
                    second_pref = ["second", "1"]
                    if contains_any_substring(vlm_pref, first_pref) and human_pref == 0:
                        reward = 1.0
                    elif contains_any_substring(vlm_pref, second_pref) and human_pref == 1:
                        reward = 1.0
                    else:
                        reward = 0.0
                    
            rewards.append(reward)
            
            try:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    # local_rank = int(os.getenv("LOCAL_RANK", 0))
                    with open(log_path, "a") as f:
                        f.write(f"------------- {current_time} Preference reward: {reward} -------------\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {solutions}\n")
            except Exception as e:
                print(f"Error with logging: {e}")
            
        return rewards

    @staticmethod
    def select_reward_func(func: str, task_type: str):
        if func == "accuracy":
            match task_type:
                case "rec":
                    return Qwen2VLModule.iou_reward
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        elif func == "format":
            match task_type:
                case "rec":
                    return Qwen2VLModule.format_reward_rec
                case _:
                    raise ValueError(f"Unsupported reward function: {func}")
        else:
            raise ValueError(f"Unsupported reward function: {func}")