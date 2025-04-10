
## Overview

FinLab provides the code to run inference time scaling on FinanceBench with automatic evaluation.

It also includes the official implementation of Dr. SoW reward function, which shows good the best inference time scaling performance on the Benchmark.

## Usage

### Inference Time Scaling with Dr.Sow Reward

1. **Initialize the VLLM servers:**

   ```bash
   bash launch_judge.sh
   bash launch_drsow.sh
   ```

2. **Launch inference time scaling:**

   ```bash
   python main.py \
     --model-name meta-llama/Llama-3.1-8B-Instruct \
     --prm-path drsow \
     --sampling-method best-of-n \
     --test-time-compute-budget 128
   ```
![inference_scaling_Llama-3 1-8B-Instruct](https://github.com/user-attachments/assets/546658cf-2cb0-451c-84ee-28c5cb27caa3)

### Inference Time Scaling with Classifier Rewards

1. **Initialize the judge server:**

   ```bash
   bash launch_judge.sh
   ```

2. **Launch inference with classifier rewards:**

   ```bash
   python main.py \
     --model-name meta-llama/Llama-3.1-8B-Instruct \
     --prm-path Qwen/Qwen2.5-Math-PRM-7B \
     --sampling-method best-of-n \
     --test-time-compute-budget 128
   ```

## Parameters

- `--model-name`: Base language model to use
- `--prm-path`: Path to the reward model or classifier
- `--sampling-method`: Method for sampling (e.g., best-of-n)
- `--test-time-compute-budget`: Computational budget for inference

## Research Foundation

### Dr. SoW: Density Ratio of Strong-over-Weak LLMs

This project implements the Dr. SoW methodology as described in the paper:

[Dr. SoW: Density Ratio of Strong-over-weak LLMs for Reducing the Cost of Human Annotation in Preference Tuning](https://arxiv.org/pdf/2411.02481)

#### Citation

```
@inproceedings{Xu2024DrSD,
  title={Dr. SoW: Density Ratio of Strong-over-weak LLMs for Reducing the Cost of Human Annotation in Preference Tuning},
  author={Guangxuan Xu and Kai Xu and Shivchander Sudalairaj and Hao Wang and Akash Srivastava},
  year={2024},
  url={https://api.semanticscholar.org/CorpusID:273821919}
}
```
