
from .vllm_server import VLLM
from multiprocessing import Process, Manager
import torch 
from transformers import AutoTokenizer
import numpy as np


class LogprobsVLLM:
    def __init__(self, modelname, port, tokenizer):
        self.modelname = modelname
        self.port =port
        self.engine = VLLM(modelname, port)
        self.tokenizer = tokenizer

    def get_last_turn_logprobs(self, messages_list, temperature):
        """_summary_

        Args:
            messages_list (List): [
                [
                    {
                        "role": ..,
                        "content": ..,
                    }
                ],
                ...
            ]

        Return:
        last_turn_tokens, last_turn_token_logprobs, last_turn_sequence_logprobs 
        """
        conversation_contexts = [messages[:-1] for messages in messages_list]
        full_conversations = messages_list

        if self.modelname != "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B":
            context_batch = [
                self.tokenizer.apply_chat_template(message, add_generation_prompt=True, tokenize=False)
                for message in conversation_contexts
            ]
        else:
            context_batch = [
                self.tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False)
                for message in conversation_contexts
            ]

        full_batch = [
            self.tokenizer.apply_chat_template(message, add_generation_prompt=False, tokenize=False)
            for message in full_conversations
        ]

        tokenized_context_batch = [self.tokenizer.encode(ex) for ex in context_batch]

        # for each item in the tokenized batch; find the index of last non-pad token
        generation_first_token_indices = [
            len(ex) for ex in tokenized_context_batch
        ]

        def fetch_logprobs(batch, model_name, port, result_dict, key, temperature):
            tokens, tokens_logprobs = self.engine.vllm_request_logprobs(batch, model_name=model_name, port=port, temperature=temperature)
            result_dict[key] = {
                "tokens": tokens,
                "tokens_logprobs": tokens_logprobs
            }

        manager = Manager()
        results = manager.dict()

        processes = [
            Process(target=fetch_logprobs, args=(full_batch, self.modelname, self.port, results, 'model_outputs', temperature)),
        ]

        for process in processes:
            process.start()

        for process in processes:
            process.join()

        model_tokens = results['model_outputs']["tokens"]

        model_logprobs = results['model_outputs']["tokens_logprobs"]
        final_reward_dicts = []

        for idx in range(len(model_logprobs)):

            response_start_idx = generation_first_token_indices[idx]

            model_unmask_indices = [
                i for i, token in enumerate(model_tokens[idx]) if i >= response_start_idx
            ]

            generated_tokens_pref =  np.array(model_tokens[idx])[model_unmask_indices]
            generated_logprobs_pref =  np.array(model_logprobs[idx])[model_unmask_indices]

            # add single instances into the batch
            final_reward_dicts.append({
                "last_turn_tokens": generated_tokens_pref,
                "last_turn_token_logprobs": generated_logprobs_pref,
                "last_turn_sequence_logprobs": sum(generated_logprobs_pref)
            })
        # the final rewards output needs to be an object that contains
        # generated tokens and their respective log-probs
        # for both the pref_model and the ref_model

        return final_reward_dicts


if __name__ == "__main__":
    modelname="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
    port=8001
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(modelname)
    logprobs_pipeline = LogprobsVLLM(modelname, port, tokenizer)
    messages = [
        [
            {
                "role": "user",
                "content": "hello"
            },
            {
                "role": "assistant",
                "content": "hello"
            }
        ]
    ]
    results = logprobs_pipeline.get_last_turn_logprobs(messages, temperature=0.1)

    print(results)
    breakpoint()