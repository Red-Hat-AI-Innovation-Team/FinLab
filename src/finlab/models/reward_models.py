#!/usr/bin/env python
# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

from itertools import accumulate

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer,
    AutoModel
)
from multiprocessing import Process, Manager
from finlab.config import Config
import torch.nn.functional as F
import re
from vllm import LLM
import math
from finlab.models.logprobs import LogprobsVLLM
from tqdm import tqdm
import numpy as np


CANDIDATE_TOKENS = [648, 387]
STEP_TAG_ID = 12902

def batched_math_shepherd_inference(
    model: PreTrainedModel,
    tokenizer: PreTrainedTokenizer,
    inputs: list[str],
    batch_size: int,
) -> list[list[float]]:
    output_scores = []
    for i in range(0, len(inputs), batch_size):
        inputs_batch = inputs[i : i + batch_size]
        inputs_batch = tokenizer(inputs_batch, padding=True, return_tensors="pt").to(
            model.device
        )
        with torch.no_grad():
            logits = model(**inputs_batch).logits[:, :, CANDIDATE_TOKENS]
            scores = logits.softmax(dim=-1)[:, :, 0]
            step_scores_flat = scores[inputs_batch.input_ids == STEP_TAG_ID].tolist()
            # Split scores into sublist based on number of \n in the input
            step_scores = []
            counter = 0
            for i in range(len(inputs_batch.input_ids)):
                count = inputs_batch.input_ids[i].tolist().count(STEP_TAG_ID)
                step_scores.append(step_scores_flat[counter : counter + count])
                counter += count

        # Store the step scores for this batch
        output_scores.extend(step_scores)

        # Clear GPU memory
        del inputs_batch, logits, scores
        torch.cuda.empty_cache()

    return output_scores


class PRM:
    def __init__(self, search_config: Config, **model_kwargs):
        self.search_config = search_config
        if search_config.prm_path == "PRIME-RL/EurusPRM-Stage2":
            self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)
        else:
            self.model, self.tokenizer = self.load_model_and_tokenizer(**model_kwargs)

    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        raise NotImplementedError

    def score(
        self, questions: list[str], outputs: list[list[str]]
    ) -> list[list[float]]:
        raise NotImplementedError




class INTERMLM_ORM(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name = "internlm/internlm2-7b-reward"
        model = AutoModel.from_pretrained(model_name,
                                        torch_dtype=torch.bfloat16,
                                        device_map="cuda", 
                                        trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]], **kwargs
    ) -> list[list[float]]:
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]
                reward_score = self.model.get_score(self.tokenizer, messages)
                steps_scores = [reward_score]
                all_step_scores.append(steps_scores)
            all_scores.append(all_step_scores)
        return all_scores



class QWEN_ORM(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer()

    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_name = "Qwen/Qwen2.5-Math-RM-72B"
        model = AutoModel.from_pretrained(model_name,
                                        device_map="auto",
                                        torch_dtype=torch.bfloat16,
                                        trust_remote_code=True).eval()
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        tokenizer.truncation_side = "left" 
        return model, tokenizer
    
    # implement score to not error out when passed additional arguments
    def score(
        self, questions: list[str], outputs: list[list[str]], max_token_length: int = 4096, **kwargs
    ) -> list[list[float]]:
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                message = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ]
                conversation_str = self.tokenizer.apply_chat_template(
                    message, 
                    tokenize=False, 
                    add_generation_prompt=False
                )

                input_ids = self.tokenizer(
                    conversation_str, 
                    return_tensors="pt",
                    add_special_tokens=False, 
                    truncation=True, 
                    max_length=max_token_length
                ).input_ids.to(self.model.device)
                raw_outputs = self.model(input_ids=input_ids)
                reward_scores = [raw_outputs[0].item()]
                all_step_scores.append(reward_scores)
            all_scores.append(all_step_scores)
        return all_scores


# class QWEN_ORM(PRM):
#     def __init__(self, search_config: Config, **model_kwargs):
#         self.model, self.tokenizer = self.load_model_and_tokenizer()

#     def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
#         model_name = "Qwen/Qwen2.5-Math-RM-72B"
#         num_gpus = torch.cuda.device_count()
#         print(f"Number of GPUs: {num_gpus}")

#         model = LLM(model=model_name, 
#                     task="reward",
#                     gpu_memory_utilization=0.5,
#                     tensor_parallel_size=num_gpus,
#                     )
#         tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
#         return model, tokenizer

#     def score(
#         self, questions: list[str], outputs: list[list[str]], **kwargs
#     ) -> list[list[float]]:
#         all_scores = []
#         for question, answers in zip(questions, outputs, strict=True):
#             all_step_scores = []
#             conversation_strs = []
#             for ans in answers:

#                 QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
#                 message = [
#                     {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
#                     {"role": "user", "content": question},
#                     {"role": "assistant", "content": ans},
#                 ]
#                 conversation_str = self.tokenizer.apply_chat_template(
#                     message, 
#                     tokenize=False, 
#                     add_generation_prompt=False
#                 )
#                 conversation_strs.append(conversation_str)
#             raw_outputs = self.model.encode(conversation_strs)
#             # print(raw_outputs[0].outputs.data[-1])

#             reward_scores = [[out.outputs.data[-1].item()] for out in raw_outputs]
#             all_step_scores = reward_scores
#             all_scores.append(all_step_scores)
#         return all_scores



class QWEN_PRM(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer(search_config.prm_path)

    def load_model_and_tokenizer(self, model_name: str = None) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        import os
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        # if model_name == "Qwen/Qwen2.5-Math-PRM-7B":
        model = LLM(model=model_name, 
                    task="reward",
                    device=num_gpus-1, # use the last gpu
                    gpu_memory_utilization=0.8,
                    tensor_parallel_size=1,
                    )
        # else:
        #     model = LLM(model=model_name, 
        #                 task="reward",
        #                 gpu_memory_utilization=0.7,
        #                 tensor_parallel_size=num_gpus,
        #                 )
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        tokenizer.truncation_side = "left"
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]], outputs_is_single_step: bool = False,
        aggregate_method: str = "model_aggregate", max_token_length=4096, rm_batch_size=16, **kwargs
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question.
        aggregate_method: "product", "lowest", "last", "model_aggregate"
        '''
        if aggregate_method == "model_aggregate":
            outputs_is_single_step = True
        else:
            outputs_is_single_step = False
        # TODO: implement QWEN-PRM scoring
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            conversation_strs = []
            for ans in answers:
                # we assume here that the answers use "\n\n" to separate steps. 
                if outputs_is_single_step:
                    ans = re.sub(r'\n+', '\n', ans)

                steps_list = ans.split("\n\n")
                QWEN_PRM_SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."
                messages = [
                    {"role": "system", "content": QWEN_PRM_SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "<extra_0>".join(steps_list) + "<extra_0>"},
                ] # 0.88671875

                # Prepare conversation for scoring
                conversation = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=False, 
                    add_generation_prompt=False
                )
                conversation_strs.append(conversation)

            # TODO: tokenize each batch independently so there is less padding and more memory efficient
            # Process in batches of rm_batch_size
            all_batch_scores = []
            for i in range(0, len(conversation_strs), rm_batch_size):
                batch = conversation_strs[i:i + rm_batch_size]
                batch_input_ids = self.tokenizer(
                    batch,
                    return_tensors="pt", 
                    truncation=True,
                    padding=True,
                    max_length=max_token_length
                ).input_ids
                batch_decoded = self.tokenizer.batch_decode(batch_input_ids, skip_special_tokens=False)
                batch_outputs = self.model.encode(batch_decoded)
                batch_scores = [[d[-1].item() for d in ex.outputs.data] for ex in batch_outputs]
                all_batch_scores.extend(batch_scores)
            out_scores = all_batch_scores

            for step_scores in out_scores:
                # make the scores cumulative through multiplication
                if aggregate_method == "product":
                    step_scores = [math.prod(step_scores[:i+1]) for i in range(len(step_scores))]
                elif aggregate_method == "lowest":
                    step_scores = [min(step_scores)]
                elif aggregate_method == "last":
                    step_scores = [step_scores[-1]]
                elif aggregate_method == "model_aggregate":
                    pass
                else:
                    raise ValueError(f"Invalid aggregate method: {aggregate_method}")

                all_step_scores.append(step_scores)
            all_scores.append(all_step_scores)
        return all_scores

from transformers import AutoModelForSequenceClassification
class Skywork_ORM(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # super().__init__(search_config, **model_kwargs)
        self.model, self.tokenizer = self.load_model_and_tokenizer(search_config.prm_path)

    def load_model_and_tokenizer(self, model_name: str = None) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        import os
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        num_gpus = torch.cuda.device_count()
        print(f"Number of GPUs: {num_gpus}")
        # if model_name == "Qwen/Qwen2.5-Math-PRM-7B":
        model = AutoModelForSequenceClassification.from_pretrained(
                model_name,
                torch_dtype=torch.bfloat16,
                device_map=num_gpus-1, # use the last gpu
                # attn_implementation="flash_attention_2",
                num_labels=1,
            )
        tokenizer = AutoTokenizer.from_pretrained(model_name)

        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]],
        aggregate_method: str = "model_aggregate", max_token_length=4096, rm_batch_size=4, **kwargs
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question.
        aggregate_method: "product", "lowest", "last", "model_aggregate"
        '''
        # TODO: implement QWEN-PRM scoring
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            question_scores = []
            for ans in answers:
                messages = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": ans},
                ] # 0.88671875

                # Prepare conversation for scoring
                conversation_tokenized = self.tokenizer.apply_chat_template(
                    messages, 
                    tokenize=True, 
                    return_tensors="pt"
                ).to(self.model.device)
                # Get the reward scores
                with torch.no_grad():
                    score = self.model(conversation_tokenized).logits[0][0].item()
                    question_scores.append([score])

            all_scores.append(question_scores)

        return all_scores


def get_logprob_of_output_given_question(model, question, answer, proposal_prompt, temperature, vllm_object):
    if proposal_prompt is None:
        messages = [
            [ 
                {"role": "user", "content":  f"{question}"},
                {"role": "assistant", "content": answer},
            ]
        ]
    else:
        messages = [
            [   
                {"role": "system", "content":  f"{proposal_prompt}"},
                {"role": "user", "content":  f"{question}"},
                {"role": "assistant", "content": answer},
            ]
        ]

    last_turn_tokens, last_turn_token_logprobs, last_turn_sequence_logprobs, last_turn_avg_sequence_logprobs = vllm_object.fetch_logprobs(model, messages, temperature=temperature)
    
    return last_turn_sequence_logprobs, last_turn_avg_sequence_logprobs


class LogProbVLLM:
    def __init__(self, strong_model_name, strong_port, weak_model_name, weak_port):
        self.strong_model_name = strong_model_name
        self.strong_port = strong_port
        self.weak_model_name = weak_model_name
        self.weak_port = weak_port
        self.strong_tokenizer = AutoTokenizer.from_pretrained(strong_model_name)
        self.weak_tokenizer = AutoTokenizer.from_pretrained(weak_model_name)

    def fetch_logprobs(self, model, messages, temperature, start_output_token_idx=None):
       
        if model == "strong": 
            modelname = self.strong_model_name
            port = self.strong_port
        elif model == "weak": 
            modelname = self.weak_model_name
            port = self.weak_port
    
        if model == "strong": 
            tokenizer = self.strong_tokenizer
            
        elif model == "weak": 
            tokenizer = self.weak_tokenizer

        logprobs_pipeline = LogprobsVLLM(modelname, port, tokenizer)

        results = logprobs_pipeline.get_last_turn_logprobs(messages, temperature)

        # TODO: it is not okay to only take the first element of the results.
        last_turn_tokens, last_turn_token_logprobs, last_turn_sequence_logprobs = results[0]['last_turn_tokens'], results[0]['last_turn_token_logprobs'], results[0]['last_turn_sequence_logprobs']
        
        # TODO: only keep the tokens after the start_output_token_idx.
        if start_output_token_idx is not None:
            last_turn_tokens = last_turn_tokens[start_output_token_idx:]

            last_turn_token_logprobs = last_turn_token_logprobs[start_output_token_idx:]
            last_turn_sequence_logprobs = sum(last_turn_token_logprobs)
        last_turn_avg_sequence_logprobs = last_turn_sequence_logprobs / len(last_turn_tokens)
        return last_turn_tokens, last_turn_token_logprobs, last_turn_sequence_logprobs, last_turn_avg_sequence_logprobs


QWEN_MATH_PROMPT = "Please reason step by step, and put your final answer within \\boxed{{}}."
def sigmoid(x):
    """
    Compute the sigmoid function for a float input.
    
    Args:
        x (float): Input value
        
    Returns:
        float: Sigmoid of input, 1/(1 + e^(-x))
    """
    if not isinstance(x, (int, float)):
        raise TypeError("Input must be a number")
    return 1.0 / (1.0 + np.exp(-x))

import re
from collections import Counter

def repetition_checker(answer, word_threshold=20, repetition_threshold=2):
    """
    Check if the answer contains repetitive sequences that might indicate degeneration.
    
    Args:
        answer (str): The text to check for repetitions
        word_threshold (int): Minimum length of sequences to check for repetition
        repetition_threshold (int): Number of repetitions required to flag text
        
    Returns:
        bool: True if repetitive sequences of word_threshold words or more are found, False otherwise
    """

    # write an algorithm in O(n) time complexity to check for repetition
    # Clean the text by removing extra whitespace
    cleaned_text = re.sub(r'\s+', ' ', answer).strip()
    words = cleaned_text.split()
    
    # If there are too few words, no need to check for repetition
    if len(words) < word_threshold*2:  # Need at least 40 words to have repetition of 20-word sequences
        return False
    
    # Use rolling hash (Rabin-Karp algorithm) for O(n) time complexity
    n = len(words)
    
    # We'll use a single window size equal to the threshold
    window_size = word_threshold
    
    # If text is too short for a repetition of our minimum size
    if n < window_size * (repetition_threshold+1):
        return False
    
    # Use a set to track seen fingerprints
    seen_hashes = Counter()
    
    # Compute hash for the first window
    current_hash = hash(tuple(words[:window_size]))
    seen_hashes[current_hash] = 1
    
    # Slide the window through the text, one word at a time
    for i in range(1, n - window_size + 1):
        # Compute new hash by removing first word and adding next word
        # This gives us O(1) hash computation per window
        current_hash = hash(tuple(words[i:i+window_size]))
        
        # If we've seen this hash before, we found a potential repetition
        if current_hash in seen_hashes:
        #     # Verify the match (in case of hash collision)
        #     for j in range(i-window_size, -1, -1):
        #         if words[j:j+window_size] == words[i:i+window_size]:
            seen_hashes[current_hash] += 1
            if seen_hashes[current_hash] > repetition_threshold: # report when 3 times of repetition.
                return True
        seen_hashes[current_hash] += 1
    
    return False


class DrSow(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        self.model, self.tokenizer = self.load_model_and_tokenizer(search_config.strong_model_name, search_config.strong_port, search_config.weak_model_name, search_config.weak_port)
    def load_model_and_tokenizer(self, strong_model_name: str, strong_port: int, weak_model_name: str, weak_port: int) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        # Load PRM model
        vllm_object = LogProbVLLM(strong_model_name=strong_model_name, strong_port=strong_port, weak_model_name=weak_model_name, weak_port=weak_port)
        # Load tokenizer
        tokenizer = vllm_object.strong_tokenizer

        return vllm_object, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]], rm_batch_size=64, temperature=0.03, aggregate_method=None, system_prompt=None ,**kwargs
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question.
        '''
        

        def get_scores(question, answer, vllm_object, result_dict, key):
            strong_score, strong_avg_score = get_logprob_of_output_given_question(
                model="strong",
                question=question, 
                answer=answer,
                proposal_prompt=system_prompt,
                temperature=1.0,
                vllm_object=vllm_object
            )
            weak_score, weak_avg_score = get_logprob_of_output_given_question(
                model="weak",
                question=question, 
                answer=answer,
                proposal_prompt=system_prompt,
                temperature=1.0,
                vllm_object=vllm_object
            )
            result_dict[key] = {
                "strong": strong_avg_score,
                "weak": weak_avg_score,
                "strong_unnormalized": strong_score,
                "weak_unnormalized": weak_score,
            }

        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_batch_scores = []
            # Create a copy of answers to avoid modifying the original list
            answers_copy = answers.copy()
            for i in tqdm(range(0, len(answers_copy), rm_batch_size)):
                batch_answers = answers_copy[i:i+rm_batch_size]
                
                manager = Manager()
                results = manager.dict()
                processes = []
                for idx, answer in enumerate(batch_answers):
                    processes.append(
                        Process(target=get_scores, args=(question, answer, self.model, results, idx))
                    )
                for process in processes:
                    process.start()

                for process in processes:
                    process.join()

                avg_drsow = []
                word_threshold = 15
                repetition_indices = []
                for idx in range(len(batch_answers)):
                    avg_strong_score = results[idx]['strong']
                    avg_weak_score = results[idx]['weak']
                    # if avg_strong_score is very negative; and avg_weak_score is more negative, resulting in a strong positive score
                    # this would be a tricky case, leading to reward hacking. 
                    if avg_strong_score < -1.1 and avg_weak_score < avg_strong_score:
                        drsow_score = avg_strong_score
                    elif repetition_checker(batch_answers[idx], word_threshold=word_threshold, repetition_threshold=2):
                        drsow_score = -2 # artificially low score for repetitive answers
                        print(f"########## Repetitive answer detected at threshold {word_threshold}##########")
                        repetition_indices.append(idx)
                    else:
                        drsow_score = avg_strong_score - avg_weak_score
                    if aggregate_method == "unnormalized":
                        drsow_score = results[idx]["strong_unnormalized"] - results[idx]["weak_unnormalized"]
                    else:
                        drsow_score = sigmoid(drsow_score / temperature)

                    avg_drsow.append([drsow_score])
                
                if len(repetition_indices) > 0:
                    print(f"########## Random picked repetitive answer detected at indices: {repetition_indices[0]}##########")
                    print(f"########## Repetitive answer: {batch_answers[repetition_indices[0]]}##########")
                all_batch_scores.extend(avg_drsow)
            # normalize the scores to be between 0 and 1
            all_scores.append(all_batch_scores)
            
        return all_scores


class PRIME(PRM):
    def __init__(self, search_config: Config, **model_kwargs):
        # override original init, because we need to load two models and a tokenizer
        # super().__init__(search_config, **model_kwargs)
        self.model, self.ref_model, self.tokenizer = self.load_model_and_tokenizer()


    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedModel, PreTrainedTokenizer]:
        # Load PRM model
        model = AutoModelForCausalLM.from_pretrained(
            'PRIME-RL/EurusPRM-Stage2',
            device_map="auto",
            attn_implementation="flash_attention_2", 
            torch_dtype=torch.float16,
        ).eval()

        # Load reference model
        ref_model = AutoModelForCausalLM.from_pretrained(
            'Qwen/Qwen2.5-Math-7B-Instruct',
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained('PRIME-RL/EurusPRM-Stage2')

        return model, ref_model, tokenizer
    
    def score(
        self, questions: list[str], outputs: list[list[str]], **kwargs
    ) -> list[list[float]]:
        '''
        Score a batch of questions and their step-by-step outputs using PRIME scoring.
        questions: list of questions
        outputs: list of lists of N responses, where N answers correspond to 1 question. 
        '''



        # TODO: implement PRIME scoring
        # implement based on the commented example code above, and also the MathShepherd code
        # Prepare inputs by combining questions and outputs
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                # we assume here that the answers use "\n\n" to separate steps. 
                ans_list = ans.split("\n\n")
                # Prepare conversation for scoring
                conversation = [
                    {"role": "user", "content": question},
                    {"role": "assistant", "content": "\n\n".join(ans_list)},
                ]

                # Tokenize full conversation
                input_ids = self.tokenizer.apply_chat_template(
                    conversation,
                    tokenize=True,
                    add_generation_prompt=False,
                    return_tensors='pt'
                ).to(self.model.device)
                
                attention_mask = (input_ids != self.tokenizer.pad_token_id).to(self.model.device)

                # Get token positions for each step
                step_last_tokens = []
                for step_num in range(0, len(ans_list) + 1):
                    step_conv = [
                        {"role": "user", "content": question},
                        {"role": "assistant", "content": "\n\n".join(ans_list[:step_num])},
                    ]
                    conv_text = self.tokenizer.apply_chat_template(
                        step_conv,
                        tokenize=False,
                        add_generation_prompt=False
                    ).strip()
                    
                    if step_num != 0 and step_num != len(ans_list):
                        conv_text += '\n\n'
                        
                    curr_ids = self.tokenizer.encode(conv_text, add_special_tokens=False)
                    step_last_tokens.append(len(curr_ids) - 2)

                inputs = {
                    'input_ids': input_ids,
                    'attention_mask': attention_mask,
                    'labels': input_ids
                }

                label_mask = torch.zeros_like(input_ids)
                label_mask[0, step_last_tokens[0]:] = 1
                step_last_tokens = torch.tensor([step_last_tokens]).to(self.model.device)

                def get_logps(model,inputs):
                    logits = model(input_ids=inputs['input_ids'], attention_mask=inputs['attention_mask']).logits
                    labels = inputs['labels'][:, 1:].clone().long()
                    logits = logits[:, :-1, :]
                    labels[labels == -100] = 0
                    per_token_logps = torch.gather(logits.log_softmax(-1), dim=2, index=labels.unsqueeze(2)).squeeze(2)
                    return per_token_logps

                # Get log probabilities from both models
                with torch.no_grad():
                    # Main model logprobs
                    per_token_logps = get_logps(self.model, inputs)
                    ref_per_token_logps = get_logps(self.ref_model, inputs)


                # Calculate rewards
                raw_reward = per_token_logps - ref_per_token_logps
                beta_reward = 0.001 * raw_reward * label_mask[:, 1:]  # Using 0.001 as default coefficient
                beta_reward = beta_reward.cumsum(-1)
                step_rewards = beta_reward.gather(dim=-1, index=step_last_tokens[:, 1:]).tolist()[0]
                
                all_step_scores.append(step_rewards)
            
            all_scores.append(all_step_scores)

        return all_scores






class MathShepherd(PRM):
    def load_model_and_tokenizer(self) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        model_id = "peiyi9979/math-shepherd-mistral-7b-prm"
        tokenizer = AutoTokenizer.from_pretrained(model_id)
        # For batched inference
        tokenizer.pad_token = tokenizer.eos_token
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            attn_implementation="flash_attention_2",
            torch_dtype=torch.float16,
        ).eval()
        return model, tokenizer

    def score(
        self, questions: list[str], outputs: list[list[str]], **kwargs
    ) -> list[list[float]]:
        inputs_for_prm = []
        lengths = []
        for question, output in zip(questions, outputs):
            prompt = self.search_config.system_prompt + "\n" + question + "\n"
            special_outputs = [o.replace("\n\n", " ки\n\n") for o in output]
            special_outputs = [
                o + " ки" if o[-2:] != "\n\n" else o for o in special_outputs
            ]
            inputs_for_prm.extend([f"{prompt} {o}" for o in special_outputs])
            lengths.append(len(output))

        # TODO: tokenize each batch independently so there is less padding and faster inference
        output_scores = batched_math_shepherd_inference(
            self.model,
            self.tokenizer,
            inputs_for_prm,
            self.search_config.prm_batch_size,
        )
        cumulative_lengths = list(accumulate(lengths))
        # reshape the output scores to match the input
        output_scores = [
            output_scores[i:j]
            for i, j in zip([0] + cumulative_lengths[:-1], cumulative_lengths)
        ]

        # stripped_output_scores = [] TODO: strip out the reward for previous steps
        for output_score, output in zip(output_scores, outputs):
            assert len(output_score) == len(
                output
            ), f"{len(output_score)} != {len(output)}"

        return output_scores


class RLHFFlow(PRM):
    def load_model_and_tokenizer(
        self, **model_kwargs
    ) -> tuple[PreTrainedModel, PreTrainedTokenizer]:
        tokenizer = AutoTokenizer.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data"
        )
        model = AutoModelForCausalLM.from_pretrained(
            "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data",
            device_map="auto",
            torch_dtype=torch.bfloat16,
            **model_kwargs,
        ).eval()
        tokenizer.padding_side = "right"
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id

        plus_tag_id = tokenizer.encode("+")[-1]
        minus_tag_id = tokenizer.encode("-")[-1]
        self.candidate_tokens = [plus_tag_id, minus_tag_id]

        return model, tokenizer

    def score(
        self,
        questions: list[str],
        outputs: list[list[str]],
        batched: bool = True,
        batch_size=8,
        **kwargs,
    ) -> list[list[float]]:
        if batched is True:
            return self._score_batched(questions, outputs, batch_size=batch_size)
        else:
            return self._score_single(questions, outputs)

    def _score_single(self, questions: list[str], outputs: list[list[str]]):
        # reference code: https://github.com/RLHFlow/RLHF-Reward-Modeling/blob/main/math-rm/prm_evaluate.py
        all_scores = []
        for question, answers in zip(questions, outputs, strict=True):
            all_step_scores = []
            for ans in answers:
                single_step_score = []
                conversation = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        # TODO: add the system prompt like we did for math shepard?
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})
                    input_ids = self.tokenizer.apply_chat_template(
                        conversation, return_tensors="pt"
                    ).to(self.model.device)
                    with torch.no_grad():
                        logits = self.model(input_ids).logits[
                            :, -3, self.candidate_tokens
                        ]  # simple version, the +/- is predicted by the '-3' position
                        step_scores = logits.softmax(dim=-1)[
                            :, 0
                        ]  # 0 means the prob of + (1 mean -)
                        # print(scores)
                        single_step_score.append(
                            step_scores[0]
                            .detach()
                            .to("cpu", dtype=torch.float32)
                            .item()
                        )

                all_step_scores.append(single_step_score)
            all_scores.append(all_step_scores)
        return all_scores

    def _score_batched(
        self, questions: list[str], outputs: list[list[str]], batch_size: int = 2, **kwargs
    ):
        # The RLHFlow models are trained to predict the "+" or "-" tokens in a dialogue, but since these are not unique
        # we need to introduce a dummy special token here for masking.

        special_tok_id = self.tokenizer("ки", return_tensors="pt").input_ids[0, 1]
        # We construct two parallel dialogues, one with a "+" token per assistant turn, the other with the dummy token "ки" for masking
        conversations = []
        conversations2 = []
        for question, answers in zip(questions, outputs, strict=True):
            for ans in answers:
                conversation = []
                conversation2 = []
                ans_list = ans.split("\n\n")
                for k in range(len(ans_list)):
                    if k == 0:
                        text = question + " " + ans_list[0]
                    else:
                        text = ans_list[k]
                    conversation.append({"content": text, "role": "user"})
                    conversation.append({"content": "+", "role": "assistant"})

                    # we track to location of the special token with ки in order to extract the scores
                    conversation2.append({"content": text, "role": "user"})
                    conversation2.append({"content": "ки", "role": "assistant"})

                conversations.append(conversation)
                conversations2.append(conversation2)

        output_scores = []
        for i in range(0, len(conversations), batch_size):
            convs_batch = conversations[i : i + batch_size]
            convs2_batch = conversations2[i : i + batch_size]
            inputs_batch = self.tokenizer.apply_chat_template(
                convs_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            inputs2_batch = self.tokenizer.apply_chat_template(
                convs2_batch, padding=True, return_tensors="pt"
            ).to(self.model.device)
            assert inputs_batch.shape == inputs2_batch.shape
            with torch.no_grad():
                logits = self.model(inputs_batch).logits[:, :, self.candidate_tokens]
                scores = logits.softmax(dim=-1)[
                    :, :, 0
                ]  # 0 means the prob of + (1 mean -)

                for i in range(len(convs_batch)):
                    # We slice on the N-1 token since the model is trained to predict the Nth one ("+" in this case)
                    step_scores_flat = scores[i, :-1][
                        inputs2_batch[i, 1:] == special_tok_id
                    ].tolist()
                    output_scores.append(step_scores_flat)

        # reshape the output scores to match the input
        reshaped_output_scores = []
        counter = 0
        for question, answers in zip(questions, outputs):
            scores = []
            for answer in answers:
                scores.append(output_scores[counter])
                counter += 1
            reshaped_output_scores.append(scores)

        return reshaped_output_scores


def load_prm(config: Config) -> PRM:
    if config.prm_path == "peiyi9979/math-shepherd-mistral-7b-prm":
        return MathShepherd(config)

    if config.prm_path == "RLHFlow/Llama3.1-8B-PRM-Deepseek-Data":
        return RLHFFlow(config)
    if config.prm_path == "PRIME-RL/EurusPRM-Stage2":
        return PRIME(config)

    if config.prm_path == "Qwen/Qwen2.5-Math-PRM-7B" or config.prm_path == "Qwen/Qwen2.5-Math-PRM-72B":
        return QWEN_PRM(config)
    
    if config.prm_path == "internlm/internlm2-7b-reward":
        return INTERMLM_ORM(config)
    
    if config.prm_path == "Qwen/Qwen2.5-Math-RM-72B":
        return QWEN_ORM(config)

    if config.prm_path == "unnormalized_drsow":
        # config.strong_model_name = "Qwen/Qwen2.5-32B-instruct"
        # config.weak_model_name = "Qwen/Qwen2.5-32B"
        return DrSow(config)
    
    if "skywork" in config.prm_path.lower():
        return Skywork_ORM(config)

    if config.prm_path.lower() == "drsow":
        return DrSow(config)

    raise NotImplementedError(f"PRM {config.prm_path} not implemented")

if __name__ == "__main__":
    # write a test for the INTERMLM_ORM model
    # config = Config(prm_path="internlm/internlm2-7b-reward")
    # config = Config(prm_path="drsow")
    config = Config(prm_path="Skywork/Skywork-Reward-Llama-3.1-8B-v0.2")
    prm = load_prm(config)
    questions = ["Jane has 12 apples. She gives 4 apples to her friend Mark, then buys 1 more apple, and finally splits all her apples equally among herself and her 2 siblings. How many apples does each person get?"]
    outputs = [["1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among herself and her 2 siblings (3 people in total). 9 ÷ 3 = 3 apples each. Each person gets 3 apples.",
                "1. Jane starts with 12 apples and gives 4 to Mark. 12 - 4 = 8. Jane now has 8 apples.\n2. Jane buys 1 more apple. 8 + 1 = 9. Jane now has 9 apples.\n3. Jane splits the 9 apples equally among her 2 siblings (2 people in total). 9 ÷ 2 = 4.5 apples each. Each person gets 4 apples."]*8,
               ]
    # [0.76171875, 0.96484375, 0.99609375]
    # [0.765625]
    scores = prm.score(questions, outputs)
    print(scores)
    breakpoint()




