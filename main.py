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
        if len(answer.split()) < 3 and ("$" in answer[0] or is_number(answer[0])):
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


def is_correct(vllm_server, question, extracted_answer, real_answer, tokenizer):
    """Check if the extracted answer is correct using model evaluation"""
    for attempt in range(3):
        try:
            model_evaluation, text = model_evaluate_answer(
                vllm_server, question, real_answer, extracted_answer, tokenizer
            )
            break
        except Exception as e:
            if attempt == 2:  # Last attempt
                raise
            print(f"Attempt {attempt+1} failed: {e}. Retrying...")
    
    return model_evaluation, text


def model_evaluate_answer(vllm_server, question, real_answer, generated_answer, tokenizer):
    """Use the model to evaluate if an answer is correct"""
    numerical = is_number(real_answer)
    if numerical:
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
            f"For factual statements, check if the key facts and conclusions in the Proposed Solution match those in the Correct Answer.\n"
            f"The Proposed Solution must contain the correct information or equivalent statements that convey the same meaning.\n"
            f"You should output Correct if the Proposed Solution contains all the essential information from the Correct Answer.\n"
            f"Output Partial if the Proposed Solution contains some but not all of the correct information or has minor inaccuracies.\n"
            f"Output Incorrect if the Proposed Solution contains significant factual errors or misses critical information.\n"
            f"[Step 2] Output your final judgement in the following format: \\boxed{{Correct}} or \\boxed{{Incorrect}} or \\boxed{{Partial}}.\n"
        )

    prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": input_text}],
        tokenize=False,
        add_generation_prompt=True
    )

    response = vllm_server.make_vllm_request(
        [prompt],
        port=vllm_server.port,
        model_name=vllm_server.model_name,
        decoding_method="greedy",
        max_new_tokens=20000,
        temperature=0.0,
        top_k=50,
        top_p=0.85,
        stop_sequences=None,
        num_workers=40,
        logprobs=0,
        echo=False,
        repetition_penalty=1.0
    )
    
    extracted_text = extract_answer(response[0]["generated_text"]).lower()
    if "partial" in extracted_text:
        return 0.5, response[0]["generated_text"]
    elif "incorrect" in extracted_text:
        return 0, response[0]["generated_text"]
    elif "correct" in extracted_text:
        return 1, response[0]["generated_text"]
    else:
        raise ValueError("Model evaluation failed: couldn't determine correctness")


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
                       use_rag_thought_prompt=False, judge_vllm_server=None, 
                       test_time_compute_budget=32, thinking=None, sampling_method="greedy", args=None):
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
        judge_vllm_server: VLLM server for the judge model
        num_particles: Number of particles for particle filtering
        thinking: Detailed thinking to include in the prompt
        
    Returns:
        dict: Results including accuracy and individual question results
    """
    results = []
    correct_count = 0
    
    # Use the same server for judging if not provided
    if judge_vllm_server is None:
        raise ValueError("Judge VLLM Server not provided")
    tokenizer = llm.get_tokenizer()
    if "phi-4-mini-instruct" in model_name.lower():
        tokenizer.eos_token = "<|end|>"
        tokenizer.eos_token_id = tokenizer.encode("<|end|>")[0]

    judge_tokenizer = AutoTokenizer.from_pretrained(judge_vllm_server.model_name)

    if sampling_method == "greedy" or sampling_method == "best-of-n":

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
        if sampling_method == "greedy":
            # Greedy decoding approach
            print("Using greedy decoding...")
            sampling_params = SamplingParams(max_tokens=4000, temperature=0.0, top_p=0.95)
            raw_outputs = llm.generate(formatted_prompts, sampling_params)
            responses = [res.outputs[0].text for res in raw_outputs]
        else:
            # best-of-n decoding approach
            print(f"Using best-of-{test_time_compute_budget} decoding...")
            sampling_params = SamplingParams(max_tokens=4000, temperature=0.8, top_p=0.95, n=test_time_compute_budget)
            raw_outputs = llm.generate(formatted_prompts, sampling_params)

            # Process all responses for each question using the reward model
            responses = []
            for i, example in enumerate(test_set):
                question = unformatted_prompts[i]
                candidates = [output.text for output in raw_outputs[i].outputs]
                
                # drsow_system_prompt = "You are a Finance Analyst with extreme attention to detail. You are given finance documents and data, and you need to answer questions based on the content you have. You must follow the user instruction carefully, including the decimal precision requirement of the answer. You are also an excellent mathematician, and you can perform complex calculations with ease."
                # drsow_system_prompt = "You are a Finance Analyst with extreme attention to detail. You are given finance documents and data, and you need to answer questions based on the content you have. You must follow the user instruction carefully, including the decimal precision requirement of the answer. You can not make any mistake in the reasoning step! If you make a single mistake, PEOPLE WILL DIE!!!!"
                # Score all candidates using the reward model
                if "unnormalized" in args.prm_path:
                    candidate_scores = prm.score([question], [candidates], aggregate_method="unnormalized")[0]
                else:
                    candidate_scores = prm.score([question], [candidates])[0]
                # n_question x n_candidates x num_steps
                
                # Flatten the scores bc they're nested
                candidate_scores = [score[-1] for score in candidate_scores]
                
                # Find the candidate with the highest score
                best_candidate_idx = np.argmax(candidate_scores)
                best_candidate = candidates[best_candidate_idx]
                
                # Add the best candidate to the responses
                responses.append(best_candidate)
                
                print(f"Question {i+1}: Selected candidate {best_candidate_idx+1}/{len(candidates)} with score {candidate_scores[best_candidate_idx]:.4f}")

        for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
            question = example['question']
            correct_answer = example['answer']
            extracted_answer = extract_answer(responses[i])
            
            scoring, judgement = is_correct(
                judge_vllm_server, question, extracted_answer, correct_answer, judge_tokenizer
            )
            

            correct_count += scoring
                
            results.append({
                'question_id': i,
                'question': question,
                'correct_answer': correct_answer,
                'model_response': responses[i],
                'extracted_answer': extracted_answer,
                'is_correct': scoring,
                'evaluation_judgement': judgement
            })
            
            print(f"Question {i+1}/{len(test_set)}: {'✓' if scoring==1 else '1/2' if scoring==0.5 else '✗'} (Accuracy so far: {correct_count/(i+1):.2%})")
            print(f"Model Answer: {extracted_answer}")
            print(f"Correct Answer: {correct_answer}")

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
                judge_vllm_server, question, extracted_answer, correct_answer, judge_tokenizer
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
    # elif sampling_method == "best-of-n":
    #     # best-of-n sampling
    #     for i, example in enumerate(tqdm(test_set, desc="Evaluating")):
    #         question = example['question']
    #         correct_answer = example['answer']
    #         documents = example['documents']
    #         formatted_prompt =format_prompt(example['question'], example['documents'], tokenizer, use_rag_thought_prompt=use_rag_thought_prompt, thinking=thinking)[0]
        
    #         sampling_params = SamplingParams(max_tokens=8000, temperature=0.8, top_p=0.95, n=test_time_compute_budget)
    #         raw_outputs = llm.generate(formatted_prompt, sampling_params)
    #         responses = [single_res.text for single_res in raw_outputs[0].outputs]

    #         rewards = prm.score([question], [responses])[0] # zero index; because we only have one question
            
    #         # Select the best response based on rewards
    #         best_response_idx = np.argmax(rewards)
    #         model_response = responses[best_response_idx]
            
    #         # Extract answer and judge correctness
    #         extracted_answer = extract_answer(model_response)
            
    #         is_answer_correct, judgement = is_correct(
    #             judge_vllm_server, question, extracted_answer, correct_answer, judge_tokenizer
    #         )
            
    #         if is_answer_correct:
    #             correct_count += 1
                
    #         results.append({
    #             'question_id': i,
    #             'question': question,
    #             'correct_answer': correct_answer,
    #             'model_response': model_response,
    #             'extracted_answer': extracted_answer,
    #             'is_correct': is_answer_correct,
    #             'evaluation_judgement': judgement
    #         })
            
    #         print(f"Question {i+1}/{len(test_set)}: {'✓' if is_answer_correct else '✗'}")
    #         print(f"Model Answer: {extracted_answer}")
    #         print(f"Correct Answer: {correct_answer}")
            
    else:
        raise ValueError(f"Invalid sampling method: {sampling_method}. Choose either 'greedy', 'pf', or 'best-of-n'.")

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
    parser.add_argument('--judge-model-name', type=str, default="microsoft/phi-4", help='Model name for judging')
    parser.add_argument('--judge-port', type=int, default=8000, help='Port for the judge VLLM server')
    parser.add_argument('--gpu-memory-utilization', type=float, default=0.8, help='GPU memory utilization')
    parser.add_argument('--device', type=int, default=0, help='GPU device ID')
    parser.add_argument('--prm-path', type=str, default="drsow", help='PRM path; prm used for particle filtering; or best-of-n')
    parser.add_argument('--output-dir', type=str, default="finbench_eval_full", help='Output directory')
    parser.add_argument('--use-rag-thought-prompt', type=bool, default=True, help='Whether to use RAG thought prompt')
    parser.add_argument('--sampling-method', type=str, default="greedy", help='Sampling method; greedy or pf or best-of-n')
    parser.add_argument('--test-time-compute-budget', type=int, default=32, help='Test time compute budget for particle filtering or best-of-n')
    parser.add_argument('--thinking', type=str, default=None, help='Detailed thinking')
    parser.add_argument('--bench-name', type=str, default="finbench", help='Benchmark to evaluate on: finbench or nvidia-bench')
    args = parser.parse_args()

    # Use the parsed arguments
    model_name = args.model_name
    test_time_compute_budget = args.test_time_compute_budget
    thinking = args.thinking
    sampling_method = args.sampling_method
    prm = load_prm(Config(
        prm_path=args.prm_path,
        # strong_model_name="Qwen/QwQ-32B",
        strong_model_name="Qwen/Qwen2.5-32B-instruct",
        weak_model_name="Qwen/Qwen2.5-32B",
        strong_port = 8305,
        weak_port = 8306
    ))

    prm.score(["What is the revenue of NVIDIA in 2024 Q4?"], [["$39,331 million"]])
    # print(prm.model.strong_port)
    # print(prm.model.weak_port)
    # print(prm.model.strong_model_name)
    # print(prm.model.weak_model_name)
    # test1 = prm.model.fetch_logprobs("strong", [[{"role": "user", "content": "What is the revenue of NVIDIA in 2024 Q4?"}, {"role": "assistant", "content": "$39,331 million and more!!!   !!!"}]], 1.0)
    # test2 = prm.model.fetch_logprobs("weak", [[{"role": "user", "content": "What is the revenue of NVIDIA in 2024 Q4?"}, {"role": "assistant", "content": "$39,331 million and more!!!   !!!"}]], 1.0)
    llm = LLM(
        model=model_name,
        gpu_memory_utilization=args.gpu_memory_utilization,
        swap_space=16,
        # enable_prefix_caching=False,
        # device=args.device,
        # tensor_parallel_size=1,
    )


    os.environ["TOKENIZERS_PARALLELISM"] = "False"

    judge_vllm_server = VLLM(model_name=args.judge_model_name, port=args.judge_port)

    # add a uuid to the output file name
    import uuid
    uuid = str(uuid.uuid4())
    
    mode = sampling_method

    if thinking is not None:
        mode += f"_thinking_{thinking}"

    output_file = os.path.join(args.output_dir,args.bench_name, f"{model_name.replace('/', '_')}_{mode}_prm_{args.prm_path.replace('/', '_')}_budget_{test_time_compute_budget}_numerical_eval_{uuid}.jsonl")
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    if args.bench_name == "finbench":
        testset = load_finance_bench()[0]
    elif args.bench_name == "nvidia-bench":
        testset = load_nvidia_bench()
    else:
        raise ValueError(f"Invalid benchmark name: {args.bench_name}. Choose either 'finbench' or 'nvidia-bench'.")

    eval_result = evaluate_on_benchmark(testset, model_name, 
                                     llm=llm, use_rag_thought_prompt=args.use_rag_thought_prompt, prm=prm,
                                     judge_vllm_server=judge_vllm_server, test_time_compute_budget=test_time_compute_budget, thinking=thinking, sampling_method=sampling_method, args=args)

    # save the eval result to a json file
    with open(output_file, "w") as f:
        for result in eval_result['results']:
            f.write(json.dumps(result) + "\n")
