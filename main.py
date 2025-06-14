from datasets import load_dataset
from vllm import LLM, SamplingParams
from finlab.search.particle_gibbs_batch import particle_gibbs_kernel
from finlab.models.reward_models import load_prm, Config
import os
from tqdm import tqdm
from finlab.models.vllm_server import VLLM
from transformers import AutoTokenizer
import numpy as np
import argparse
import logging
import json
import uuid
import concurrent.futures
import concurrent
from openai import OpenAI
import torch


# Disable logging from the sal.search module
logging.getLogger('sal.search').setLevel(logging.ERROR)
os.environ["TOKENIZERS_PARALLELISM"] = "False"

THOUGHT_PROMPT = ("Your role as an assistant is to thoroughly analyze and answer questions strictly based on the given context and your internal knowledge, following a structured, step-by-step reasoning process to arrive at the correct solution. " "Use this step-by-step format:\n\n" "## Step 1: [Evidence from the context] {List all information relevant to the question that you can find from the context}\n\n" "## Step 2: [Reasoning] {Use the above evidence and knowledge to reason step by step to get to the final answer}\n\n" "## Step 3: [Final answer] put your final answer within \\boxed{{}}\n\n" "Now, using the provided context, solve the following question step by step in the above format:")

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        return False


def load_finance_bench():
    """Load and prepare the FinanceBench dataset"""
    finance_bench = load_dataset("gx-ai-architect/financebench-numerical-fixed")
    
    all_examples = []
    numerical_examples = []

    for ds in finance_bench['train']:
        question = ds['question']
        answer = ds['answer']
        documents = [evidence_instance['evidence_text'] for evidence_instance in ds['evidence']]
        
        example = {
            "question": question,
            "answer": answer,
            "documents": documents
        }
        
        all_examples.append(example)
        
        # Filter for numerical answers
        if len(answer.split()) < 3 and ("$" in answer[0] or is_number(answer[0]) or "%" in answer[0]):
            numerical_examples.append(example)
    
    return all_examples, numerical_examples


def load_nvidia_bench():
    with open("data/test/nvidia_easy_questions.jsonl", "r") as f:
        """Load the NVIDIA benchmark questions from a JSONL file"""
        nvidia_examples = []
        
        for line in f:
            example = json.loads(line)
            nvidia_examples.append(example)
        
    return nvidia_examples


def format_prompt(question, documents, tokenizer, use_rag_thought_prompt=False, thinking=None):
    """Format the prompt for the model"""
    if use_rag_thought_prompt:
        prompt = "DOCUMENT: " + "\n\n".join(documents) + "\n\n" + f"DOCUMENT END\n\n Given the above document,\n<thought-prompt>\n{question}.".replace("<thought-prompt>", THOUGHT_PROMPT)
    else:
        prompt = "DOCUMENT: " + "\n\n".join(documents) + "\n\n" + f"DOCUMENT END\n\n Given the above document, {question} Please reason step by step, and put your final answer within \\boxed{{}}."

    if thinking is not None:
        formatted_prompt = tokenizer.apply_chat_template(
            [
                {"role": "system", "content": f"detailed thinking {thinking}"},
                {"role": "user", "content": prompt}
            ],
            tokenize=False,
            add_generation_prompt=True
        )
        formatted_prompt = formatted_prompt + "\n\n<think>"
    else:
        formatted_prompt = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=False,
            add_generation_prompt=True
        )
    
    return formatted_prompt, prompt


def get_response(formatted_prompt, model_name, vllm_server, port):
    """Get response from the VLLM server"""
    response = vllm_server.make_vllm_request(
        [formatted_prompt],
        port=str(port),
        model_name=model_name,
        decoding_method="greedy",
        max_new_tokens=20000,
        temperature=0.0,
        top_k=50,
        top_p=0.95,
        stop_sequences=None,
        num_workers=40,
        logprobs=0,
        echo=False,
        repetition_penalty=1.0
    )
    return response[0]["generated_text"]


def extract_answer(text):
    """
    Extract the answer from text that contains a LaTeX boxed expression.
    Properly handles nested braces and extracts the content inside \boxed{}.
    """
    if not text:
        return "ERROR"

    # Convert to lowercase for case-insensitive matching
    text_lower = text.lower()

    # Look for \boxed pattern
    if "\\boxed" not in text_lower:
        if "## step 3:" in text_lower:
            return text_lower.split("## step 3:")[-1].strip()
        elif "step 3:" in text_lower:
            return text_lower.split("step 3:")[-1].strip()
        elif "\n\n" in text:
            return text.split("\n\n")[-1].strip()
        else:
            return text

    # Find all instances of \boxed
    boxed_indices = [i for i, _ in enumerate(text_lower) if text_lower[i:i+6] == "\\boxed"][-1:]

    for start_idx in boxed_indices:
        # Find the opening brace after \boxed
        open_brace_idx = text.find("{", start_idx)
        if open_brace_idx == -1:
            continue

        # Track nested braces to find the matching closing brace
        brace_count = 1
        for i in range(open_brace_idx + 1, len(text)):
            if text[i] == "{":
                brace_count += 1
            elif text[i] == "}":
                brace_count -= 1
                
            if brace_count == 0:
                # Extract content between braces
                content = text[open_brace_idx + 1:i].strip()
                if content:
                    return content.replace("\%", "%")
    
    return "ERROR"


def is_correct(judge_model_name, question, extracted_answer, real_answer, tokenizer, openai_client=None):
    """Check if the extracted answer is correct using model evaluation"""
    for attempt in range(3):
        try:
            model_evaluation, text = model_evaluate_answer(
                judge_model_name, question, real_answer, extracted_answer, tokenizer, openai_client
            )
            break
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            print(f"Attempt {attempt+1} failed: {e}. Retrying...")
    
    return model_evaluation, text


def model_evaluate_answer(judge_model_name, question, real_answer, generated_answer, tokenizer, openai_client=None):
    """Use the model to evaluate if an answer is correct"""
    if len(real_answer.split()) < 3 and ("$" in real_answer or is_number(real_answer)):
        input_text = (
            f"Question: {question}\nCorrect Answer: {real_answer}\nProposed Solution: {generated_answer}\n\n"
            "Given the above correct answer and proposed solution, follow the below steps to evaluate the proposed solution:\n"
        f"[Step 1] First analyze the Proposed Solution using the correct answer as a reference.\nFollow the question regarding the precision/decimal requirement of the answer;\nif not specified, you can ignore decimal differences. You should output Correct if the Proposed Solution is exactly the same as the Correct Answer, or when the question does not specify decimal precision, integer digits are the same. If decimal precision is required, you should output Partial if the Proposed Solution is only decimals difference from the Correct Answer. Otherwise, you should output Incorrect."
            f"[Step 2] Output your final judgement in the following format: \\boxed{{Correct}} or \\boxed{{Incorrect}} or \\boxed{{Partial}}.\n"
        )
    else:
        input_text = (
            f"Question: {question}\nCorrect Answer: {real_answer}\nProposed Solution: {generated_answer}\n\n"
            "Given the above correct answer and proposed solution, follow the below steps to evaluate the proposed solution:\n"
            f"[Step 1] First analyze the Proposed Solution using the correct answer as a reference.\n"
            f"Check if the final conclusion in the Proposed Solution aligns with the correct answer.\n"
            f"You should output Correct if the Proposed Solution reaches the same conclusion.\n"
            f"Output Partial if the Proposed Solution contains contradictory information against the correct answer, but still gives a correct conclusion.\n"
            f"Output Incorrect if the Proposed Solution does not reach the same conclusion.\n"
            f"[Step 2] Output your final judgement in the following format: \\boxed{{Correct}} or \\boxed{{Incorrect}} or \\boxed{{Partial}}.\n"
        )



    response = openai_client.chat.completions.create(
        model=judge_model_name,
        messages=[{"role": "user", "content": input_text}],  # Send one message at a time
        temperature=0.0,
        max_tokens=4000
    )
    response = response.choices[0].message.content
    extracted_text = extract_answer(response).lower()

    if "partial" in extracted_text:
        return 0.5, response
    elif "incorrect" in extracted_text:
        return 0, response
    elif "correct" in extracted_text:
        return 1, response
    else:
        return 0, response

def particle_filtering(llm, prompt, prm, num_particles=32):
    """Run particle filtering to get the best response"""
    particles, _, _ = particle_gibbs_kernel(
        prompt, llm, prm, None, num_particles, 1.0, True, 
        llm_sampling_temp=0.8, get_full_response=False, 
        max_steps_for_each_particle=40, random_seed=96
    )
    rewards = [p.rewards[-1] for p in particles]
    max_reward_idx = np.argmax(rewards)
    return "\n\n".join(particles[max_reward_idx].trajectory)


def evaluate_on_benchmark(test_set, model_name, llm=None, prm=None,
                       use_rag_thought_prompt=False, judge_model_name="gpt4-o", 
                       test_time_compute_budget=32, thinking=None, sampling_method="greedy", args=None, judge_openai_client=None, temperature=0.1):
    """
    Evaluates the model on a development set of financial questions.
    
    Args:
        vllm_server: VLLM server for the model
        test_set: A list of dictionaries, each containing 'question', 'answer', and 'documents' keys
        model_name: Name of the model
        tokenizer: Tokenizer for the model
        llm: The language model to use for evaluation
        prm: Preference reward model for particle filtering
        use_rag_thought_prompt: Whether to use RAG thought prompt
        judge_model_name: Name of the judge model
        num_particles: Number of particles for particle filtering
        thinking: Detailed thinking to include in the prompt
        
    Returns:
        dict: Results including accuracy and individual question results
    """
    results = []
    correct_count = 0
    
    tokenizer = llm.get_tokenizer()
    if "phi-4-mini-instruct" in model_name.lower():
        tokenizer.eos_token = "<|end|>"
        tokenizer.eos_token_id = tokenizer.encode("<|end|>")[0]

    judge_tokenizer = None
    if sampling_method == "greedy" or sampling_method == "best-of-n" or sampling_method == "majority-vote":

        formatted_prompts = [
            format_prompt(example['question'], example['documents'], tokenizer, 
                         use_rag_thought_prompt=use_rag_thought_prompt, thinking=thinking)[0] 
            for example in test_set
        ]
        unformatted_prompts = [
            format_prompt(example['question'], example['documents'], tokenizer, 
                         use_rag_thought_prompt=use_rag_thought_prompt, thinking=thinking)[1] 
            for example in test_set
        ]
        all_scores = []
        if sampling_method == "greedy":
            # Greedy decoding approach
            print("Using greedy decoding...")
            sampling_params = SamplingParams(max_tokens=4000, temperature=0.0, top_p=0.95)
            raw_outputs = llm.generate(formatted_prompts, sampling_params)
            responses = [res.outputs[0].text for res in raw_outputs]
            all_responses = responses
        elif sampling_method == "majority-vote":
            # best-of-n decoding approach
            print(f"Using majority-vote-of-{test_time_compute_budget} decoding...")
            sampling_params = SamplingParams(max_tokens=4000, temperature=temperature, top_p=0.95, n=test_time_compute_budget)
            raw_outputs = llm.generate(formatted_prompts, sampling_params)
            all_responses = [ [candidate.text for candidate in raw_output.outputs] for raw_output in raw_outputs]
            # Process all responses for each question using the reward model
            responses = []
            for i, example in enumerate(test_set):
                question = unformatted_prompts[i]
                candidates = all_responses[i]
                # extract the short response;
                extracted_answers = [extract_answer(candidate) for candidate in candidates]
                # majority vote
                # Count occurrences of each answer and select the most common one
                most_common_answer = max(set(extracted_answers), key=extracted_answers.count)
                responses.append(most_common_answer)
                print(f"Question {i+1}: Selected candidate with majority vote")
            

        else:
            # best-of-n decoding approach
            print(f"Using best-of-{test_time_compute_budget} decoding...")
            sampling_params = SamplingParams(max_tokens=4000, temperature=temperature, top_p=0.95, n=test_time_compute_budget)
            raw_outputs = llm.generate(formatted_prompts, sampling_params)
            all_responses = [ [candidate.text for candidate in raw_output.outputs] for raw_output in raw_outputs]

            # Process all responses for each question using the reward model
            responses = []
            for i, example in enumerate(test_set):
                question = unformatted_prompts[i]
                candidates = all_responses[i]
                # Score all candidates using the reward model
                if "unnormalized" in args.prm_path:
                    candidate_scores = prm.score([question], [candidates], aggregate_method="unnormalized")[0]
                else:
                    candidate_scores = prm.score([question], [candidates])[0]
                # n_question x n_candidates x num_steps
                
                # Flatten the scores bc they're nested
                candidate_scores = [score[-1] for score in candidate_scores]
                all_scores.append(candidate_scores)
                # Find the candidate with the highest score
                best_candidate_idx = np.argmax(candidate_scores)
                best_candidate = candidates[best_candidate_idx]
                
                # Add the best candidate to the responses
                responses.append(best_candidate)
                
                print(f"Question {i+1}: Selected candidate {best_candidate_idx+1}/{len(candidates)} with score {candidate_scores[best_candidate_idx]:.4f}")


        gpt4o_judge, gpt4o_judge_text = [], []
        with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
            futures = []
            for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
                output = executor.submit(is_correct, judge_model_name, example['question'], extract_answer(responses[i]), example['answer'], judge_openai_client, judge_openai_client)
                futures.append(output)
            for future in tqdm(concurrent.futures.as_completed(futures), total=len(futures)):
                score, judge_text = future.result()
                gpt4o_judge.append(score)
                gpt4o_judge_text.append(judge_text)
            
        for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
            correct_count += gpt4o_judge[i]
                
            results.append({
                'question_id': i,
                'question': example['question'],
                'correct_answer': example['answer'],
                'model_response': responses[i],
                "all_responses": all_responses[i],
                "all_responses_scores": all_scores[i] if len(all_scores) > 0 else None,
                'extracted_answer': extract_answer(responses[i]),
                'is_correct': gpt4o_judge[i],
                'evaluation_judgement': gpt4o_judge_text[i]
            })
            
            print(f"Question {i+1}/{len(test_set)}: {'✓' if gpt4o_judge[i]==1 else '1/2' if gpt4o_judge[i]==0.5 else '✗'} (Accuracy so far: {correct_count/(i+1):.2%})")
            print(f"Model Answer: {extract_answer(responses[i])}")
            print(f"Correct Answer: {example['answer']}")

    elif sampling_method == "pf":
        # Particle filtering approach
        print(f"Using particle filtering with {test_time_compute_budget} particles...")
        for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
            question = example['question']
            correct_answer = example['answer']
            documents = example['documents']
            
            _, user_prompt = format_prompt(
                question, documents, tokenizer, 
                use_rag_thought_prompt=use_rag_thought_prompt, thinking=thinking
            )

            model_response = particle_filtering(llm, user_prompt, prm, test_time_compute_budget)
            extracted_answer = extract_answer(model_response)
            
            scoring, judgement = is_correct(
                judge_model_name, question, extracted_answer, correct_answer, judge_tokenizer, judge_openai_client
            )

            correct_count += scoring
                
            results.append({
                'question_id': i,
                'question': question,
                'correct_answer': correct_answer,
                'model_response': model_response,
                'extracted_answer': extracted_answer,
                'is_correct': scoring,
                'evaluation_judgement': judgement
            })
            
            print(f"Question {i+1}/{len(test_set)}: {'✓' if scoring==1 else '1/2' if scoring==0.5 else '✗'} (Accuracy so far: {correct_count/(i+1):.2%})")
            print(f"Model Answer: {extracted_answer}")
            print(f"Correct Answer: {correct_answer}")
            
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}. Choose either 'greedy', 'pf', or 'best-of-n', or 'majority-vote'.")

    # Calculate accuracy
    accuracy = correct_count / len(test_set) if len(test_set) > 0 else 0
    print(f"Accuracy: {accuracy:.2%}")
    
    return {
        'accuracy': accuracy,
        'correct_count': correct_count,
        'total_count': len(test_set),
        'results': results
    }



if __name__ == "__main__":

    import argparse

    # Set up argument parser
    parser = argparse.ArgumentParser(description='Evaluate models on financial benchmarks')
    parser.add_argument('--model-name', type=str, default="microsoft/phi-4", help='Model name to evaluate')
    parser.add_argument('--judge-model-name', type=str, default="gpt-4o", help='Model name for judging; only support openai models for now.')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.9, help='GPU memory utilization')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--prm-path', type=str, default="drsow", help='PRM path; prm used for particle filtering; or best-of-n or majority-vote')
    parser.add_argument('--output-dir', type=str, default="finbench_eval_full", help='Output directory')
    parser.add_argument('--use-rag-thought-prompt', type=bool, default=True, help='Whether to use RAG thought prompt')
    parser.add_argument('--sampling-method', type=str, default="greedy", help='Sampling method; greedy or pf or best-of-n')
    parser.add_argument('--test-time-compute-budget', type=int, default=32, help='Test time compute budget for particle filtering or best-of-n')
    parser.add_argument('--thinking', type=str, default=None, help='Detailed thinking; use "on" or "off" for nemotron models')
    parser.add_argument('--bench-name', type=str, default="finbench", help='Benchmark to evaluate on: finbench or nvidia-bench')
    args = parser.parse_args()

    # Use the parsed arguments
    model_name = args.model_name
    test_time_compute_budget = args.test_time_compute_budget
    thinking = args.thinking
    sampling_method = args.sampling_method
    prm = load_prm(Config(
        prm_path=args.prm_path,
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        weak_model_name="Qwen/Qwen2.5-32B",
        strong_port = 8305,
        weak_port = 8306
    ))

    prm.score(["What is the revenue of NVIDIA in 2024 Q4?"], [["$39,331 million"]])


    # Check if we're using OpenAI API
    if model_name.startswith("openai/"):
        from openai import OpenAI
        import os
        
        # Extract the actual model name from the prefix
        openai_model_name = model_name.replace("openai/", "")
        
        # Check if OpenAI API key is set
        if "OPENAI_API_KEY" not in os.environ:
            raise ValueError("OPENAI_API_KEY environment variable must be set to use OpenAI models")
        
        # Create OpenAI client
        openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
        
        # Set llm to None as we'll use the OpenAI client directly
        llm = None
        
        print(f"Using OpenAI API with model: {openai_model_name}")
    else:
        # For non-OpenAI models, continue with VLLM setup

        # get number of gpus available
        num_gpus = torch.cuda.device_count()
        tensor_parallel_size = min(num_gpus, 2)
        openai_client = None
        openai_model_name = None
        llm = LLM(
            model=model_name,
            gpu_memory_utilization=args.gpu_memory_utilization,
            tensor_parallel_size=tensor_parallel_size,
            max_model_len=8192
        )
    
    judge_openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    # add a uuid to the output file name
    import uuid
    uuid = str(uuid.uuid4())
    
    mode = sampling_method

    if thinking is not None:
        mode += f"_thinking_{thinking}"

    output_file = os.path.join(args.output_dir,args.bench_name, f"{model_name.replace('/', '_')}_{mode}_prm_{args.prm_path.replace('/', '_')}_budget_{test_time_compute_budget}_{uuid}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if args.bench_name == "finbench":
        testset = load_finance_bench()[0]
    elif args.bench_name == "nvidia-bench":
        testset = load_nvidia_bench()
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}. Choose either 'finbench' or 'nvidia-bench'.")

    eval_result = evaluate_on_benchmark(testset, model_name, 
                                    llm=llm, use_rag_thought_prompt=args.use_rag_thought_prompt, prm=prm,
                                    judge_model_name=args.judge_model_name, test_time_compute_budget=test_time_compute_budget, thinking=thinking, sampling_method=sampling_method, args=args, judge_openai_client=judge_openai_client)

    # save the eval result to a json file
    with open(output_file, "w") as f:
        for result in eval_result['results']:
            f.write(json.dumps(result) + "\n")
