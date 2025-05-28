from transformers import Qwen2_5_VLForConditionalGeneration, Qwen2VLForConditionalGeneration, AutoProcessor
from typing import Dict, Any, Union
from trl.data_utils import maybe_apply_chat_template
import torch
import math

import re
import os
from datetime import datetime
import json
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
Object and Scene Correctness: Which has more believable objects and background?
Photorealism / Image Quality: Which is sharper and free of defects?
Aesthetic Appeal: Which has better composition, color, mood?
Creativity / Originality: Which is more unique or storytelling?

For each criterion, include a one-sentence comparison and then a Score chosen from:
Strongly prefer image 1
Prefer image 1
Both images are preferred
Prefer image 2
Strongly prefer image 2

2. In a <preference> block, output your overall choice—either '1' or '2'.

3. Always think in English. Do not modify the criteria headings.

The images are generated from the prompt: {prompt}."""
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

        alignment_pattern = r"(?i)Alignment\**:\s*(.*?)\s*Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"
        correctness_pattern = r"(?i)Object and Scene Correctness\**:\s*(.*?)\s*Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"
        quality_pattern = r"(?i)Photorealism \/ Image Quality\**:\s*(.*?)\s*Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"
        aesthetic_pattern = r"(?i)Aesthetic Appeal\**:\s*(.*?)\s*Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"
        creativity_pattern = r"(?i)Creativity \/ Originality\**:\s*(.*?)\s*Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"

        patterns = [alignment_pattern, correctness_pattern, quality_pattern, aesthetic_pattern, creativity_pattern]

        contents = [c[0]["content"] for c in completions]
        
        rewards = []

        score_dict = {
            "strongly prefer image 1": -2.0,
            "prefer image 1": -1.0,
            "both images are preferred": 0.0,
            "neutral": 0.0,
            "prefer image 2": 1.0,
            "strongly prefer image 2": 2.0
        }

        for content in contents:
            total_score = 0

            pref_match = re.search(r"<preference>(1|2|-1)<\/preference>", content)
            if not pref_match:
                hallucination_reward = 0.0
                rewards.append(hallucination_reward)
                continue
                
            try:
                preference = int(pref_match.group(1))
            except:
                hallucination_reward = 0.0
                rewards.append(hallucination_reward)
                continue

            matches = []
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    matches.append(match)
                    try:
                        score = score_dict[match.group(2).strip().lower()]
                        total_score += score
                    except:
                        total_score += 0.0
                        continue

            score_magnitude = abs(total_score) / (2.0 * len(patterns))  # Normalize to [0,1]
            score_sign = 1 if total_score > 0 else -1 if total_score < 0 else 0

            # Damping factor for overconfident answers
            if score_magnitude >= 0.9:
                score_magnitude *= 0.8
            
            if preference in [1, 2]:
                pref_sign = 1 if preference == 2 else -1
                hallucination_reward = score_magnitude * (1.0 if pref_sign == score_sign else 0.0)
            else:
                # Neutral preference gets 0 reward
                hallucination_reward = 0.0

            rewards.append(hallucination_reward)

            try:
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    # local_rank = int(os.getenv("LOCAL_RANK", 0))
                    with open(log_path, "a") as f:
                        f.write(f"------------- Format reward: {hallucination_reward} -------------\n")
                        f.write(f"Matches: {matches}| Pref Sign: {pref_sign}| Score Sign: {score_sign}")
            except Exception as e:
                print(f"Error with logging: {e}")

        return rewards

    @staticmethod
    def reasoning_length_reward_func(completions, solutions, **kwargs):

        alignment_pattern = r"(?i)Alignment\**:\s*(.*?)\s*Score:"
        correctness_pattern = r"(?i)Object and Scene Correctness\**:\s*(.*?)\s*Score:"
        quality_pattern = r"(?i)Photorealism \/ Image Quality\**:\s*(.*?)\s*Score:"
        aesthetic_pattern = r"(?i)Aesthetic Appeal\**:\s*(.*?)\s*Score:"
        creativity_pattern = r"(?i)Creativity \/ Originality\**:\s*(.*?)\s*Score:"

        patterns = [alignment_pattern, correctness_pattern, quality_pattern, aesthetic_pattern, creativity_pattern]

        contents = [c[0]["content"] for c in completions]
        
        rewards = []
        for content in contents:
            length_reward = 0.0
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    reasoning = match.group(1)
                    length = len(reasoning.split())
                    length_reward += min(math.log1p(length) / math.log1p(150), 1.0) * 0.1
                else:
                    length_reward = 0.0
            
            rewards.append(length_reward)

        return rewards

    @staticmethod
    def format_reward_func(completions, solutions, per_token_logps, **kwargs):
        """Reward function that checks if the completion has a specific format."""

        base_pattern = r"""^<think>(.*?)<\/think>(\\n){1,2}<preference>(1|2)<\/preference>$"""

        alignment_pattern = r"""(?i)Alignment\**:\s*([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"""
        correctness_pattern = r"""(?i)Object and Scene Correctness\**:\s*([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"""
        quality_pattern = r"""(?i)Photorealism \/ Image Quality\**:\s*([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"""
        aesthetic_pattern = r"""(?i)Aesthetic Appeal\**:\s*([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"""
        creativity_pattern = r"""(?i)Creativity \/ Originality\**:\s*([A-Za-z0-9\s.,'";:!?\-\(\)]+)(\\n){1,2}Score:\s*(Strongly prefer image \d|Prefer image \d|Both images are preferred|Neutral)"""

        patterns = [alignment_pattern, correctness_pattern, quality_pattern, aesthetic_pattern, creativity_pattern]

        contents = [c[0]["content"] for c in completions]
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        
        rewards = []
        for content in contents:
            reward = 0.0
            
            base_match =  re.search(base_pattern, content, re.DOTALL)
            if base_match:
                reward += 0.5
            else:
                reward -= 0.2

            match_count = 0
            for pattern in patterns:
                match = re.search(pattern, content, re.DOTALL)
                if match:
                    match_count += 1
                    reward += 0.1
                else:
                    reward += 0.0

            # if match_count == 5 and base_match:
            #     reward += 0.2
            
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
    def preference_reward_func(completions, solutions, per_token_logps, **kwargs):

        rewards = []
        current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
        pref_pattern = r'<preference>(1|2)<\/preference>'

        score_dict = {
            "strongly prefer image 1": -1.0,
            "prefer image 1": -0.5,
            "both images are preferred": 0.0,
            "neutral": 0.0,
            "prefer image 2": 0.5,
            "strongly prefer image 2": 1.0
        }

        for completion, human_pref, logps in zip(completions, solutions, per_token_logps):
            content = completion[0]["content"]
            logps = logps[::-1]

            try:
                for item in logps:
                    if item[0] in ['1', '2']:
                        logps_match = item
                        break
                        
                confidence = logps_match[1] if logps_match else None
                confidence = math.exp(confidence)
            except:
                print(confidence)
                rewards.append(0.0)
                continue
                    
            try:
                pref_match = re.search(pref_pattern, content)
                vlm_pref = int(pref_match.group(1))
            except:
                rewards.append(-0.5)
                continue

            if vlm_pref in [1, 2] and vlm_pref == int(human_pref):                    
                preference_reward = confidence
            else:
                # Penalize overconfident wrong answers
                if confidence >= 0.5 and vlm_pref != int(human_pref):
                    preference_reward = -confidence * 0.8
                else:
                    preference_reward = 0.0

            rewards.append(preference_reward)
            
            try:
                torch.set_printoptions(profile="full")
                if os.getenv("DEBUG_MODE") == "true":
                    log_path = os.getenv("LOG_PATH")
                    # local_rank = int(os.getenv("LOCAL_RANK", 0))
                    with open(log_path, "a") as f:
                        f.write(f"------------- {current_time} Preference reward: {preference_reward} -------------\n")
                        f.write(f"Content: {content}\n")
                        f.write(f"Solution: {solutions} | VLM Pref: {vlm_pref} | Human pref: {human_pref} | Confidence: {confidence}\n")
                        f.write(f"Logps: {logps}\n")
 
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
