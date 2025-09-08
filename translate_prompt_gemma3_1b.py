#!/usr/bin/env python
"""
Doc-level translation for IndicDoc WAT2025 using Gemma 3-1B PT.

- Loads Pralekha dev/test splits
- Builds doc-level prompts
- Runs translation with vLLM + Gemma
- Saves outputs as CSV
"""

import os
from pathlib import Path
from typing import List
from datasets import load_dataset
from vllm import LLM, SamplingParams
import pandas as pd

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
]

MAX_NEW_TOKENS = 4096

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
def load_pralekha_split(src_lang: str, tgt_lang: str, subset: str = "dev"):
    ds = load_dataset("ai4bharat/Pralekha", subset, split=f"{src_lang}_{tgt_lang}")
    return ds

# ---------------------------------------------------------------------------
# PROMPT BUILDER
# ---------------------------------------------------------------------------
_SYSTEM = (
    "You are a professional translator. Translate the following document "
    "from {src} to {tgt}, preserving meaning, tone, and formatting."
)

def make_prompts(sentences: List[str], src: str, tgt: str) -> List[str]:
    sys = _SYSTEM.format(src=src, tgt=tgt)
    return [
        (
            "<|begin_of_text|><|system|>\n"
            f"{sys}\n"
            "<|user|>\n"
            f"{s}\n"
            "<|assistant|>"
        )
        for s in sentences
    ]

# ---------------------------------------------------------------------------
# MODEL
# ---------------------------------------------------------------------------
def init_gemma(checkpoint: str = "google/gemma-3-1b-it") -> LLM:
    return LLM(
        model=checkpoint,
        dtype="bfloat16",
        tokenizer=checkpoint,
    )

def translate(llm: LLM, prompts: List[str]) -> List[str]:
    params = SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS)
    outputs = llm.generate(prompts, params)
    return [out.outputs[0].text.strip() for out in outputs]

# ---------------------------------------------------------------------------
# MAIN LOOP
# ---------------------------------------------------------------------------
def main():
    out_dir = Path("./translations")
    out_dir.mkdir(parents=True, exist_ok=True)

    llm = init_gemma()

    for pair in PAIRS:
        src, tgt = pair.split("_")
        subset = "dev"
        print(f"\n>>> Translating {pair} ({subset})")

        ds = load_pralekha_split(src, tgt, subset)
        prompts = make_prompts(ds["src_txt"], src, tgt)
        translations = translate(llm, prompts)

        ds = ds.add_column("pred_txt", translations)
        out_file = out_dir / f"translations_{pair}_{subset}.csv"
        ds.to_pandas().to_csv(out_file, index=False)

        print(f"✓ Saved {len(ds)} translations to {out_file}")

if __name__ == "__main__":
    main()
