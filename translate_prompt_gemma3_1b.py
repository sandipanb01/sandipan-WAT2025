#!/usr/bin/env python   #TRANSLATION PROMPT FOR ALL LANGUAGE PAIRS#
"""
Quick-start: translate a slice of the Pralekha dev-set with vLLM + Gemma-3-1B-PT.
"""
import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
from pathlib import Path
from typing import List

from datasets import load_dataset
from vllm import LLM, SamplingParams

import os, re, subprocess, sys

def _fix_cuda_visible_devices():
    cvd = os.environ.get("CUDA_VISIBLE_DEVICES", "")
    if not cvd.startswith("GPU-"):
        # Already numeric -> nothing to do
        return

    # 1) Build a map {UUID -> index} from `nvidia-smi -L`
    try:
        smi = subprocess.check_output(["nvidia-smi", "-L"], text=True)
    except FileNotFoundError:
        sys.exit("!! nvidia-smi not found—cannot map GPU UUIDs to indices")

    uuid2idx = {}
    for line in smi.splitlines():
        # Example: "GPU 2: NVIDIA H100 (UUID: GPU-b1b8e0d1-5d40-8a62-d5da-393bca3fd881)"
        m = re.match(r"GPU\s+(\d+):.*\(UUID:\s+(GPU-[0-9a-f\-]+)\)", line)
        if m:
            idx, uuid = m.groups()
            uuid2idx[uuid] = idx

    # 2) Translate the job-allocated UUID list to indices
    try:
        new_ids = ",".join(uuid2idx[uuid] for uuid in cvd.split(","))
    except KeyError as e:
        missing = str(e).strip("'")
        sys.exit(f"!! UUID {missing} not found in `nvidia-smi -L` output.\n"
                 f"   CUDA_VISIBLE_DEVICES was: {cvd}")

    os.environ["CUDA_VISIBLE_DEVICES"] = new_ids
    print(f"[fix-gpu] CUDA_VISIBLE_DEVICES  {cvd}  →  {new_ids}")

_fix_cuda_visible_devices()

# ---------------------------------------------------------------------------
# 1. DATA LOADING
# ---------------------------------------------------------------------------
def load_pralekha_split(
    src_lang: str = "eng",
    tgt_lang: str = "hin",
    subset: str = "dev",
    max_rows: int | None = None,
):
    ds = load_dataset("ai4bharat/Pralekha", f"{subset}", split=f"{src_lang}_{tgt_lang}")
    if max_rows:
        ds = ds.select(range(min(max_rows, len(ds))))
    return ds

# ---------------------------------------------------------------------------
# 2. PROMPT TEMPLATING (Gemma format for document translation)
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional document translator. Translate the entire document precisely "
    "from {src} to {tgt}, preserving meaning, tone, formatting, and document structure."
)

def make_prompts(sentences: List[str], src: str, tgt: str) -> List[str]:
    sys = _SYSTEM.format(src=src, tgt=tgt)
    return [
        (
            f"<start_of_turn>user\n"
            f"{sys}\n"
            f"Please translate the following document from {src} to {tgt}:\n"
            f"{s}\n"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )
        for s in sentences
    ]


# ---------------------------------------------------------------------------
# 3. MODEL INSTANTIATION (vLLM)
# ---------------------------------------------------------------------------
def init_gemma(checkpoint: str = "google/gemma-2b") -> LLM:
    """
    Loads the Gemma 3 1B PT checkpoint under vLLM.
    """
    # Set the HF_TOKEN environment variable for vLLM to use
    os.environ["HF_TOKEN"] = HF_TOKEN
    return LLM(
        model=checkpoint,
        dtype="float16",
        tokenizer=checkpoint,
        max_model_len=4096, # Reduced max_model_len to 4096
        trust_remote_code=True,
        tokenizer_mode="auto",
    )


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
    # All language pairs to process
    LANGUAGE_PAIRS = [
        "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
        "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    ]

    SUBSET = "dev"  # or "test"

    for pair in LANGUAGE_PAIRS:
        src_lang, tgt_lang = pair.split("_")
        print(f"\nProcessing {src_lang} to {tgt_lang}...")

        # Load data
        ds = load_pralekha_split(src_lang, tgt_lang, SUBSET)
        print(f"Loaded {len(ds):,} rows from {SUBSET}/{src_lang}_{tgt_lang}")

        # Initialize model
        llm = init_gemma()

        # Create prompts
        prompts = make_prompts(ds["src_txt"], src_lang, tgt_lang)

        # Translate
        translations = translate(llm, prompts)

        # Add predictions & persist
        # Ensure the number of translations matches the number of initially loaded documents
        # by adding empty strings for skipped prompts
        full_translations = [""] * len(ds)
        valid_prompt_index = 0
        tokenizer = llm.get_tokenizer()
        for i, prompt in enumerate(prompts):
            if len(tokenizer.encode(prompt)) <= llm.max_model_len:
                full_translations[i] = translations[valid_prompt_index]
                valid_prompt_index += 1


        ds = ds.add_column("pred_txt", full_translations)
        out_file = Path(f"translations_{src_lang}_{tgt_lang}_{SUBSET}.csv")
        ds.to_pandas().to_csv(out_file, index=False)
        print(f"✓ Saved translations to {out_file.resolve()}")

if __name__ == "__main__":
    main()
