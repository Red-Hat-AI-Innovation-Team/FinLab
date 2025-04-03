from vllm import LLM, SamplingParams
import torch
from finlab.config import Config
from finlab.models.reward_models import load_prm
import os
from finlab.utils.data import get_dataset, save_dataset
from datasets import load_dataset
import numpy as np

os.environ["HF_TOKEN"] = "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"

from glob import glob

from datasets import load_dataset
from finlab.utils.math import *
from finlab.utils.grader import *

from finlab.utils.qwen_math_parser import *
from collections import defaultdict
import json
import numpy as np
import random

import pickle
import os
import logging


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def softmax(x):
    """
    Compute softmax values for a vector x.

    Args:
        x (numpy.ndarray): Input array of shape (n,)

    Returns:
        numpy.ndarray: Softmax probabilities of shape (n,)
    """
    # Subtract max for numerical stability
    # This prevents overflow when computing exp
    exp_x = np.exp(x - np.max(x))
    return exp_x / exp_x.sum()


def inverse_sigmoid(x):
    """
    Calculate the inverse sigmoid (logit) of a value x.

    Args:
        x (float): Input value between 0 and 1 (exclusive)

    Returns:
        float: The inverse sigmoid value

    Raises:
        ValueError: If x is not between 0 and 1
    """
    if x > 1 or x < 0:
        raise ValueError("x must be between 0 and 1")
    # Add small epsilon to prevent log(0)
    eps = np.finfo(float).eps
    x = np.clip(x, eps, 1 - eps)
    
    return np.log(x) - np.log(1 - x)  # More stable than np.log(x/(1-x))


def take_a_step(question, llm, config, steps_so_far=[], first=False, temperature=0.8, get_full_response=False):
    """
    Generates a response for a single step with a given temperature.

    Args:
        question (str): The input question/prompt.
        llm: The language model instance.
        config: Configuration containing the system prompt.
        steps_so_far (list): Previous steps in the trajectory.
        first (bool): If True, this is the first step (affects prompt construction).
        temperature (float): The sampling temperature for this step.
        get_full_response (bool): If True, get the full response.
    Returns:
        tuple: (response_text, stop_reason)
    """
    tokenizer = llm.get_tokenizer()
    system = [
        {
            "role": "system",
            "content": config.system_prompt,
        }
    ]
    sampling_params = SamplingParams(
        temperature=temperature,  # Dynamic temperature
        max_tokens=2048,
        top_p=1.0,
        stop=["\n\n", "<|eot_id|>"] if not get_full_response else ["<|eot_id|>"],
    )

    if first:
        prompt = tokenizer.apply_chat_template(
            system + [{"role": "user", "content": question}],
            tokenize=False,
            add_generation_prompt=True,
        )
    else:
        prompt = tokenizer.apply_chat_template(
            system + [{"role": "user", "content": question}], tokenize=False
        )
        prompt = prompt + "\n\n".join(steps_so_far) + "\n\n"

    res = llm.generate(prompt, sampling_params)
    response_text = res[0].outputs[0].text
    response_tokens = res[0].outputs[0].token_ids

    if tokenizer.eos_token_id in response_tokens:
        stop_reason = "EOS"
    else:
        stop_reason = "END OF STEP"

    return response_text, stop_reason


def take_a_step_for_batch(
    question,
    llm,
    system_prompt,
    particles_steps_so_far=[[]],
    first=False,
    temperature=0.8,
    n_particles=1,
    get_full_response=False,
):
    # you throw a list of questions into the llm.generate function call
    tokenizer = llm.get_tokenizer()
    sampling_params = SamplingParams(
        temperature=temperature,  # Dynamic temperature
        max_tokens=2048,
        top_p=1.0,
        stop=["\n\n", "<|eot_id|>"] if not get_full_response else ["<|eot_id|>"],
    )

    if system_prompt is not None:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question},
        ]
    else:
        messages = [
            {"role": "user", "content": question},
        ]

    if particles_steps_so_far == [[]]:
        if first:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
            prompt = prompt + "\n\n".join(particles_steps_so_far[0]) + "\n\n"

        particles_prompts = [prompt] * n_particles

    else:
        # we need to integrate the previous particle steps so far into the prompt
        particles_prompts = []
        for steps_so_far in particles_steps_so_far:
            if first:
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
            else:
                assert "\n\n" not in steps_so_far[-1], "individual steps should not contain step separators"
                prompt = tokenizer.apply_chat_template(
                    messages, tokenize=False, add_generation_prompt=True
                )
                prompt = prompt + "\n\n".join(steps_so_far) + "\n\n"
            particles_prompts.append(prompt)

    # print("Particles prompts",particles_prompts[0])
    particles_res_lists = llm.generate(particles_prompts, sampling_params)
    particles_responses = [res.outputs[0].text for res in particles_res_lists]
    particles_response_tokens = [
        res.outputs[0].token_ids for res in particles_res_lists
    ]

    def stop_reason(response_tokens):
        if tokenizer.eos_token_id in response_tokens:
            return "EOS"
        else:
            return "END OF STEP"

    particles_stops = [
        stop_reason(response_tokens) for response_tokens in particles_response_tokens
    ]
    return particles_responses, particles_stops


class Particle:
    def __init__(self, temperature=0.8):
        """
        Initializes a particle with a given temperature.

        Args:
            temperature (float): The initial temperature of the particle.
        """
        self.trajectory = []  # Tracks the sequence of responses
        self.rewards = []  # Tracks rewards for each step
        self.steps = 0  # Steps taken by the particle
        self.active = True  # Indicates if the particle is still evolving
        self.preferred = False  # Indicates if the particle is preferred
        self.temperature = temperature  # Dynamic temperature of the particle

    def add_step(self, response, reward, stop, max_steps_for_each_particle):
        """Adds a step to the particle's trajectory."""
        self.trajectory.append(response)
        self.rewards.append(reward)
        self.steps += 1
        if stop == "EOS" or """\\boxed""" in response:
            self.active = False
        if self.steps >= max_steps_for_each_particle:
            self.active = False

    def get_last_reward(self):
        """Returns the last recorded reward."""
        return self.rewards[-1]

    def is_active(self):
        """Checks if the particle is active."""
        return self.active

    def get_trajectory(self):
        """Returns the full trajectory as a single string."""
        return "\n\n".join(self.trajectory)

    def set_temperature(self, new_temperature):
        """Sets a new temperature for the particle."""
        self.temperature = new_temperature

    def deepcopy(self, numSteps=None):
        """Returns a deep copy of the particle."""
        new_particle = Particle(temperature=self.temperature)

        # if numSteps is not None:
        if numSteps is not None and numSteps > len(
                self.trajectory
            ):  # capping it so it doesnt go out of bounds
                assert self.active == False, "self.active should be False when resampling inactive particles."
                numSteps = len(self.trajectory)

        if numSteps is not None:
            # ensure numSteps is not greater than the length of the trajectory
            new_particle.trajectory = self.trajectory[:numSteps]
            new_particle.rewards = self.rewards[:numSteps]
            new_particle.steps = numSteps
            if numSteps == len(self.trajectory):
                # sampling every step; no eos before this current step; 
                new_particle.active = self.active
            elif numSteps < len(self.trajectory):
                raise ValueError("current Step is less than the length of the trajectory, should never happen")
            else:
                raise ValueError("numSteps is greater than the length of the trajectory; which should never happen.")
        else:
            new_particle.trajectory = self.trajectory.copy()
            new_particle.rewards = self.rewards.copy()
            new_particle.steps = self.steps
            new_particle.active = self.active

        new_particle.preferred = self.preferred
        return new_particle


def temperature_linear_annealing(starting_temp, ending_temp, total_steps, current_step):
    """
    Computes the temperature at a given step using linear annealing.

    Args:
        starting_temp (float): Initial temperature.
        ending_temp (float): Final temperature.
        total_steps (int): Total number of annealing steps.
        current_step (int): Current step number (0-indexed).

    Returns:
        float: Temperature at the current step.
    """
    if current_step < 0:
        raise ValueError("current_step must be >= 0.")

    if current_step >= total_steps:
        # Return constant ending temperature after the total steps.
        return ending_temp

    if total_steps <= 1:
        # Return ending temperature directly if there's only 1 or no step.
        return ending_temp

    temp_range = starting_temp - ending_temp
    step_fraction = current_step / (total_steps - 1)  # Adjust for 0-indexing
    return starting_temp - (temp_range * step_fraction)


def particle_gibbs_kernel(
    question,
    llm,
    prm,
    system_prompt,
    n_particles,
    softmax_temp,
    resample_inactive,
    reference_particle=None,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
    aggregate_method="product",
    get_full_response=False,
    max_steps_for_each_particle=40,
    random_seed=96,
):
    """
    Implements particle Gibbs sampling for response generation.

    Args:
        question: The input question/prompt
        llm: Language model instance
        prm: Parameter object containing reward model
        config: Configuration for LLM
        n_particles: Number of particles to maintain
        resample_inactive: Whether to resample inactive particles
        get_full_response: Whether to get the full response
    Returns:
        List of trajectories and their scores
    """
    logger.info("Starting Particle Gibbs sampling...")
    logger.info(f"Particles: {n_particles}")
    logger.info(f"Resample inactive: {resample_inactive}")
    logger.info(f"LLM sampling temperature: {llm_sampling_temp}")
    logger.info(f"Softmax temperature: {softmax_temp}")
    logger.info(f"Temperature annealing: {temperature_annealing}")
    logger.info(f"Aggregate method: {aggregate_method}")
    logger.info(f"Random seed: {random_seed}")

    # Sample new particles based on weights
    # Set random seed for reproducibility if needed
    rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
    stepwise_particle_tracker_before = []
    stepwise_particle_tracker_after = []

    # Initialize particles
    if reference_particle is None:
        particles = [
            Particle(temperature=llm_sampling_temp) for _ in range(n_particles)
        ]
    else:
        particles = [
            Particle(temperature=llm_sampling_temp) for _ in range(n_particles - 1)
        ]

    responses, stops = take_a_step_for_batch(question, llm, system_prompt, first=True, temperature=llm_sampling_temp, n_particles=len(particles))

    # batched the reward scoring
    rewards = prm.score([question], [responses], aggregate_method=aggregate_method)
    rewards = [r[-1] for r in rewards[-1]] # rewards[-1] because question is a list of length 1. r[-1] because we want the last step reward.


    for idx, particle in enumerate(particles):
        particle.add_step(responses[idx], rewards[idx], stops[idx], max_steps_for_each_particle)

    stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    step = 1

    while any(particle.is_active() for particle in particles):
        if resample_inactive:
            rewards = [particle.get_last_reward() for particle in particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)


            
            if reference_particle is None:
                sampled_particles = rng.choice(
                    particles,
                    size=len(particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )
            else:
                sampled_particles = rng.choice(
                    particles + [reference_particle],
                    size=len(particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )

            # TODO: change this to allow inactive particles to be sampled (bc perhaps the score of an incomplete still-evolving particle is higher than that of a completed particle)
            # print(
            #    f"Sampled particle indices: {[particles.index(p) for p in sampled_particles+[reference_particle]]}"
            # )
            # current num_of_iteration (num)
            
            particles = [
                particle.deepcopy(numSteps=step) for particle in sampled_particles
            ]
            # if any([not particle.active for particle in sampled_particles]):
            #     print("###########################")
            #     print([not particle.active for particle in particles])

        else:
            # Check active particles
            active_particles = [
                particle for particle in particles if particle.is_active()
            ]
            # print(f"Active particles: {len(active_particles)} / {n_particles}")

            # Calculate rewards and weights
            rewards = [particle.get_last_reward() for particle in active_particles]

            if reference_particle is not None:
                if step >= len(reference_particle.rewards):
                    rewards.append(reference_particle.rewards[-1])
                else:
                    rewards.append(reference_particle.rewards[step])

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=step,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)

            # print(f"Rewards of active particles: {rewards}")
            # print(f"Logits (inverse sigmoid): {logits}")
            # print(f"Weights (softmax): {weights}")

            # Sample new particles based on weights
            if reference_particle is None:
                print("len(particles)", len(particles))
                print("len(weights)", len(weights))
                sampled_particles = rng.choice(
                    active_particles,
                    size=len(active_particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )
            else:
                sampled_particles = rng.choice(
                    active_particles + [reference_particle],
                    size=len(active_particles),
                    p=weights,
                    replace=True,  # before particles was active_particles
                )

            particles = [
                particle.deepcopy()
                for particle in particles
                if not particle.is_active()
            ]

            for i in range(len(sampled_particles)):
                particles.append(sampled_particles[i].deepcopy(numSteps=step))
            # if any([not particle.active for particle in particles]):
            #     print([not particle.active for particle in particles])
            #     breakpoint()
        stepwise_particle_tracker_after.append([p.deepcopy() for p in particles])

        # use the take_a_step_for_batch function to take a step for each particle
        active_particles = [particle for particle in particles if particle.is_active()]
        inactive_particles = [particle for particle in particles if not particle.is_active()]
        responses, stops = take_a_step_for_batch(question, llm, system_prompt, first=False, particles_steps_so_far=[particle.trajectory for particle in active_particles], n_particles=len(active_particles))
        responses_to_pass_for_score = ["\n\n".join(particle.trajectory) + "\n\n" + response for response, particle in zip(responses, active_particles)]
        # rewards = [prm.score([question], [[response]], aggregate_method=aggregate_method)[-1][-1][-1] for response in responses_to_pass_for_score]

        rewards = prm.score([question], [responses_to_pass_for_score], aggregate_method=aggregate_method)
        rewards = [r[-1] for r in rewards[-1]] # rewards[-1] because question is a list of length 1. r[-1] because we want the last step reward.

        for idx, particle in enumerate(active_particles):
            if not particle.is_active():
                continue
            particle.add_step(responses[idx], rewards[idx], stops[idx], max_steps_for_each_particle)

        particles = inactive_particles + active_particles
        step = step + 1
        stepwise_particle_tracker_before.append([p.deepcopy() for p in particles])

    if reference_particle is None:
        return (
            particles,
            stepwise_particle_tracker_before,
            stepwise_particle_tracker_after,
        )
    else:
        return (
            particles + [reference_particle],
            stepwise_particle_tracker_before,
            stepwise_particle_tracker_after,
        )


import copy


def particle_gibbs_batch(
    x,
    config,
    llm,
    prm,
    total_timesteps=1,
    n_particles=4,
    resample_inactive=True,
    softmax_temp=1.0,
    temperature_annealing=(),
    llm_sampling_temp=0.8,
    aggregate_method="product",
    get_full_response=False,
    random_seed=96,
):
    particle_intermediate_storage = []
    if isinstance(x["unique_id"], int):
        question_id = x["unique_id"]
    else:
        question_id = x["unique_id"].replace("/", "_").strip(".json")
    logger.info(f"Processing question: {question_id}")
    particles_tracker = []
    current_timestep = 1
#     thought_prompt = ("Your role as an assistant is to thoroughly analyze and answer questions strictly based on the given context and your internal knowledge, following a structured, step-by-step reasoning process to arrive at the correct solution. "
# "Present your answer in a structured step-by-step format as follows:\n\n"
# "Step 1: {List all the relevant information you have about the question from the context}\n\n"
# "Step 2: {Recall all the relevant information you have about the question from your internal knowledge if applicable}\n\n"
# "Step 3: {Next step in the reasoning process, building upon the previous one, ensuring accuracy and logical flow}\n\n"
# "Step 4: {Continue this pattern, ensuring each step follows naturally and is supported by the provided context}\n\n"
# "... (Include additional steps as necessary to reach the final conclusion logically and precisely)\n\n"
# "...{Clearly state the final answer supported by the reasoning in previous steps}\n\n"
# "Now, using the provided context, solve the following question step by step in the above format:")
    thought_prompt = ("Your role as an assistant is to thoroughly analyze and answer questions strictly based on the given context and your internal knowledge, following a structured, step-by-step reasoning process to arrive at the correct solution. "
"Use this step-by-step format:\n\n"
"## Step 1: [Evidence from the context] {List all information relevant to the question that you can find from the context}\n\n"
"## Step 2: [Reasoning] {Use the above evidence and knowledge to reason step by step to get to the final answer}\n\n"
"## Step 3: [Final answer] {Clearly state the complete final answer supported by the reasoning in previous steps}\n\n"
"Now, using the provided context, solve the following question step by step in the above format:")

    current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                             x['problem'].replace("<thought_prompt>", thought_prompt), 
                            llm, 
                            prm, 
                            config.system_prompt, 
                            n_particles, 
                            resample_inactive=resample_inactive,
                            softmax_temp=softmax_temp,
                            temperature_annealing=temperature_annealing,
                            llm_sampling_temp=llm_sampling_temp,
                            aggregate_method=aggregate_method,
                            random_seed=random_seed,
    )
    particles_tracker.append(current_particles)
    particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # compute_budget_used = sum([len(p.trajectory) for p in current_particles])
    if total_timesteps > 1:
        while current_timestep < total_timesteps:
            # preferred_particle = current_particles[
            #     np.argmax([p.rewards[-1] for p in current_particles])
            # ]
            rewards = [particle.get_last_reward() for particle in current_particles]

            logits = [inverse_sigmoid(r) for r in rewards]
            logits = np.array(logits)

            if temperature_annealing:
                softmax_temp = temperature_linear_annealing(
                    starting_temp=temperature_annealing[0],
                    ending_temp=temperature_annealing[1],
                    total_steps=temperature_annealing[2],
                    current_step=current_timestep,
                )

            logger.info(f"Softmax temperature: {softmax_temp}")
            weights = softmax(logits / softmax_temp)
            rng = np.random.RandomState(random_seed) if random_seed is not None else np.random
            preferred_particle = rng.choice(
                current_particles,
                size=1,
                p=weights,
                replace=True,  # before particles was active_particles
            )[0]

            preferred_particle.preferred = True

            current_particles, tracker_before, tracker_after = particle_gibbs_kernel(
                x["problem"],
                llm,
                prm,
                config.system_prompt,
                n_particles,
                softmax_temp,
                resample_inactive=resample_inactive,
                reference_particle=preferred_particle,
                temperature_annealing=temperature_annealing,
                get_full_response=get_full_response,
                random_seed=random_seed,
            )
            particles_tracker.append(current_particles)
            current_timestep += 1
            particle_intermediate_storage.append([copy.deepcopy(current_particles)])

    # # Create output directory if it doesn't exist
    os.makedirs(config.output_dir, exist_ok=True)

    save_path = os.path.join(config.output_dir, f"{question_id}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(particles_tracker, f)

    intermediate_dir = os.path.join(config.output_dir, "intermediate")
    os.makedirs(intermediate_dir, exist_ok=True)
    save_path_intermediate = os.path.join(
        intermediate_dir, f"{question_id}_intermediate.pkl"
    )
    with open(save_path_intermediate, "wb") as f:
        pickle.dump(particle_intermediate_storage, f)

    # with open(save_path.replace(".pkl", "_before.pkl"), "wb") as f:
    #    pickle.dump(tracker_before, f)

    # with open(save_path.replace(".pkl", "_after.pkl"), "wb") as f:
    #    pickle.dump(tracker_after, f)

    logger.info(f"Saved particles to: {save_path}")

    return x
