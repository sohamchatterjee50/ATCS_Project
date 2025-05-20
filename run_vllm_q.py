import argparse
import json
import torch
from huggingface_hub import login
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


SYSTEM_PROMPT = (
    "You are Phi, a language model trained by Microsoft to help users. "
    "Your role as an assistant involves thoroughly exploring questions through a systematic thinking process before providing the final precise and accurate solutions. "
    "This requires engaging in a comprehensive cycle of analysis, summarizing, exploration, reassessment, reflection, backtracing, and iteration to develop well-considered thinking process. "
    "Please structure your response into two main sections: Thought and Solution using the specified format: <think> {Thought section} </think> {Solution section}. "
    "In the Thought section, detail your reasoning process in steps. Each step should include detailed considerations such as analysing questions, summarizing relevant findings, brainstorming new ideas, verifying the accuracy of the current steps, refining any errors, and revisiting previous steps. "
    "In the Solution section, based on various attempts, explorations, and reflections from the Thought section, systematically present the final solution that you deem correct. "
    "The Solution section should be logical, accurate, and concise and detail necessary steps needed to reach the conclusion. "
    "Now, try to solve the following question through the above guidelines:"
)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Run batch inference for BBQ disambiguation-context samples with vLLM and Microsoft Phi-4 ChatML formatting"
    )
    parser.add_argument(
        "--hf_token", type=str,
        default="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",
        help="Hugging Face API token for authentication"
    )
    parser.add_argument(
        "--model_name_or_path", type=str,
        default="microsoft/phi-4-reasoning-plus",
        help="Pretrained chat model name or path"
    )
    parser.add_argument(
        "--data_file", type=str,
        default="/home/scur1431/ATCS_Project/BBQ/Religion.jsonl",
        help="Path to the input JSONL dataset file"
    )
    parser.add_argument(
        "--log_file", type=str,
        default="/home/scur1431/ATCS_Project/LLM_Results/phi4-r.log",
        help="Path to the output log file"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Number of samples per batch"
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="Index of the first sample to process"
    )
    parser.add_argument(
        "--device", type=str, default="cuda",
        choices=["cpu", "cuda"],
        help="Device to run the model on"
    )
    parser.add_argument(
        "--max_new_tokens", type=int, default=1024,
        help="Maximum number of tokens to generate per sample"
    )
    parser.add_argument(
        "--temperature", type=float, default=0.7,
        help="Temperature for sampling"
    )
    return parser.parse_args()


def main():
    args = parse_args()

    # Authenticate to Hugging Face Hub
    login(token=args.hf_token)

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name_or_path)

    # Initialize vLLM
    dtype = torch.bfloat16 if (args.device == "cuda" and torch.cuda.is_available()) else torch.float32
    llm = LLM(
        model=args.model_name_or_path,
        tensor_parallel_size=1,
        dtype=dtype,
        device=args.device
    )

    # Load and filter dataset: only samples with context_condition == 'disambiguation'
    all_samples = []
    with open(args.data_file, 'r') as f:
        for line in f:
            all_samples.append(json.loads(line))
    filtered_samples = all_samples
 # filtered_samples = [s for s in all_samples if s.get('context_condition') == 'disambig']
    print(len(filtered_samples))
    # Batch processing on filtered data
    for batch_start in range(args.start_index, len(filtered_samples), args.batch_size):
        batch_end = min(batch_start + args.batch_size, len(filtered_samples))
        batch = filtered_samples[batch_start:batch_end]

        # Prepare ChatML-formatted prompts
        requests = []
        for sample in batch:
            question = sample.get('question', '').strip()
            ans0 = sample.get('ans0', '').strip()
            ans1 = sample.get('ans1', '').strip()
            ans2 = sample.get('ans2', '').strip()
            context = sample.get('context', '').strip()

            # Build user message
            user_msg = f"{question}"
            if context:
                user_msg = f"\n\nContext: {context}"

            # Assemble ChatML template
            prompt = (
                #f"<|im_start|>system<|im_sep|>{SYSTEM_PROMPT}<|im_end|>"
                f"\n<|im_start|>user<|im_sep|>{user_msg}<|im_end|>"
                f"\n<|im_start|>assistant<|im_sep|>"
            )

            requests.append({"prompt": prompt})

        # Setup sampling parameters
        sampling_params = SamplingParams(
            temperature=args.temperature,
            max_tokens=args.max_new_tokens
        )

        # Generate with vLLM
        results = llm.generate(requests, sampling_params)

        # Decode outputs
        outputs = [res.outputs[0].text for res in results]

                # Logging results in JSONL for downstream analysis
        with open(args.log_file, 'a') as logf:
            for i, out in enumerate(outputs):
                idx = batch_start + i
                sample = batch[i]
                record = {
                    "index": idx,
                    "question": sample.get('question', '').strip(),
                    "ans0": sample.get('ans0', '').strip(),
                    "ans1": sample.get('ans1', '').strip(),
                    "ans2": sample.get('ans2', '').strip(),
                    "context": sample.get('context', '').strip(),
                    "response": out.strip()
                }
                logf.write(json.dumps(record) + "")

        print(f"Processed batch {batch_start}-{batch_end - 1}, appended to {args.log_file}")


if __name__ == '__main__':
    main()