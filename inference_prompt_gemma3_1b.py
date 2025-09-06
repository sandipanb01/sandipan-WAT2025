#!/usr/bin/env python      #INFERENCE MODE FOR ALL LANGUAGE PAIR#
# inference_vllm.py
import argparse
import json
import sys
import os
from tqdm import tqdm
from pathlib import Path
import torch

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_file", required=True)
    p.add_argument("--output_file", required=True)
    p.add_argument("--model", default="google/gemma-3-1b-pt",  # Changed to Gemma
                   help="HF repo path or local dir; using google/gemma-3-1b-pt")
    p.add_argument("--max_new_tokens", type=int, default=4096)  # Reduced max_new_tokens
    p.add_argument("--sampling", action="store_true")
    p.add_argument("--temperature", type=float, default=0.7)
    p.add_argument("--top_p", type=float, default=0.9)
    p.add_argument("--batch_size", type=int, default=4)

    # For Colab environment, use default values if no args provided
    if 'google.colab' in sys.modules:
        # Parse with empty args first
        args = p.parse_args([])
        # Then set the required arguments with default values for Colab
        args.input_file = "./pralekha_data/test/eng_hin/doc.eng_2_hin.5.jsonl" # Example input file
        args.output_file = "./pralekha_data/translations/eng_hin_test_predictions.jsonl" # Example output file
        return args
    else:
        return p.parse_args()

def load_prompts(path):
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line: continue
        obj = json.loads(line) if line[0] in "{[" else {"prompt": line}
        if isinstance(obj, list): yield obj[0]
        else: yield obj["prompt"]

def build_generator(args):
    # Import vLLM here to control error handling
    from vllm import LLM, SamplingParams

    # Work around the UUID issue by patching vLLM's device_id_to_physical_device_id function
    import vllm.platforms.cuda as cuda_platform

    # Store the original function
    original_device_id_fn = cuda_platform.device_id_to_physical_device_id

    # Define a patched function that safely handles UUID device IDs
    def patched_device_id_fn(device_id):
        try:
            return original_device_id_fn(device_id)
        except ValueError:
            # If UUID-like device ID is encountered, return a safe index based on current context
            print(f"Warning: Encountered non-integer device ID: {device_id}, using index 0 instead")
            return 0

    # Apply the patch
    cuda_platform.device_id_to_physical_device_id = patched_device_id_fn

    # Initialize vLLM engine with safe settings for problematic environments
    print(f"Initializing vLLM for model: {args.model}")
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,  # Use single GPU tensor parallelism for stability
        dtype="float16",  # Changed to float16 for better compatibility
        trust_remote_code=True,
        gpu_memory_utilization=0.85,  # Be conservative with memory
        max_model_len=2048,  # Reduced max_model_len
        token=os.environ.get("HF_TOKEN") # Use HF_TOKEN from environment variable
    )

    # Configure sampling parameters for Gemma
    sampling_params = SamplingParams(
        max_tokens=args.max_new_tokens,
        temperature=args.temperature if args.sampling else 0.0,
        top_p=args.top_p if args.sampling else 1.0,
        stop=["<end_of_turn>", "\n\n"],  # Added Gemma stop tokens
    )

    # Define a generator function that uses vLLM for batched inference
    def generate(prompts):
        outputs = llm.generate(prompts, sampling_params)
        return [
            {"generated_text": prompt + output.outputs[0].text}
            for prompt, output in zip(prompts, outputs)
        ]

    return generate

# ---------------------------------------------------------------------------
# 4. BATCH TRANSLATION
# ---------------------------------------------------------------------------
def translate(
    llm: LLM,
    prompts: List[str],
    temperature: float = 0.0,
    max_tokens: int = 4096,
) -> List[str]:
    params = SamplingParams(
        temperature=temperature,
        max_tokens=max_tokens,
        stop=["<end_of_turn>", "\n\n"]
    )

    # Filter out prompts that are too long
    tokenizer = llm.get_tokenizer()
    valid_prompts = []
    skipped_prompts_count = 0
    for prompt in prompts:
        if len(tokenizer.encode(prompt)) <= llm.max_model_len:
            valid_prompts.append(prompt)
        else:
            skipped_prompts_count += 1

    if skipped_prompts_count > 0:
        print(f"Skipped {skipped_prompts_count} prompts that were too long for the model.")

    if not valid_prompts:
        print("No valid prompts to process.")
        return []

    outputs = llm.generate(valid_prompts, params)
    # Extract and clean the translations
    translations = []
    for out in outputs:
        text = out.outputs[0].text.strip()
        # Clean up any remaining special tokens
        text = text.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
        translations.append(text)
    return translations


# ---------------------------------------------------------------------------
# 5. END-TO-END EXECUTION FOR ALL LANGUAGE PAIRS
# ---------------------------------------------------------------------------
def main():
    args = parse_args()
    prompts = list(load_prompts(args.input_file))
    print(f"Loaded {len(prompts)} prompts.")

    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)

    # Initialize vLLM generator
    gen_fn = build_generator(args)
    print(f"Starting batched inference with vLLM (batch size: {args.batch_size})...")

    with open(args.output_file, "w", encoding="utf-8") as fout:
        for i in tqdm(range(0, len(prompts), args.batch_size), desc="Generating"):
            batch_prompts = prompts[i:i + args.batch_size]
            batch_outputs = gen_fn(batch_prompts)

            for output, prompt in zip(batch_outputs, batch_prompts):
                generation = output["generated_text"].replace(prompt, "").strip()
                # Clean up any remaining special tokens
                generation = generation.replace("<start_of_turn>", "").replace("<end_of_turn>", "").strip()
                fout.write(json.dumps([generation], ensure_ascii=False) + "\n")
            fout.flush()

            # Report memory usage for monitoring
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                used_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
                print(f"Max GPU memory used: {used_memory:.2f} MB")

    print("vLLM inference completed successfully.")

if __name__ == "__main__":
    main()
