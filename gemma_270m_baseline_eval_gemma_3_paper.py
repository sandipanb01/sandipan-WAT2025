#IT'S HIGHLY UNLIKELY THAT THIS CODE WITH THE MODEL GEMMA 270M IT WILL PRODUCE TRANSLATTIONS AT ALL.
#BUT WILL GIVE BLOATED SCORES DUE TO SOME SEMANTIC OVERLAPS.
#PLEASE LOOK INTO THIS PAPER https://arxiv.org/pdf/2402.17193 FOR FURTHER DETAILS.

# ============================================================
# Gemma-3-270M-IT ZERO-SHOT MT BASELINE (STRICT PAPER-ALIGNED)
# Dataset: AI4Bharat Pralekha (TEST)
# Directions: ENG ↔ HIN
# Decoding: Deterministic (Greedy)
# Metrics: BLEU, ChrF
# Prompting: Official Gemma-3 turn format (MINIMAL)
# ============================================================

from pathlib import Path
from typing import List
import json

import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu.metrics import BLEU, CHRF

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
DATASET_CONFIG = "test"

DIRECTIONS = ["eng_hin"]          # reverse handled explicitly
MAX_SAMPLES = 10                 # None = full test set
MAX_NEW_TOKENS = 4096             # conservative, paper-safe
BATCH_SIZE = 8                   # chat models require 1

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.float16 if DEVICE == "cuda" else torch.float32

OUT_DIR = Path("./gemma3_strict_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# OFFICIAL GEMMA-3 PROMPT (MINIMAL, PAPER-ALIGNED)
# ============================================================

def make_prompt(src_text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Minimal instruction.
    No task optimization.
    """
    return (
        "<start_of_turn>user\n"
        f"Translate from {src_lang.upper()} to {tgt_lang.upper()}:\n"
        f"{src_text}\n"
        "<end_of_turn>\n"
        "<start_of_turn>model\n"
    )

# ============================================================
# MODEL INIT
# ============================================================

def init_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=DTYPE,
        device_map="auto"
    )

    model.eval()

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return tokenizer, model

# ============================================================
# TRANSLATION
# ============================================================

@torch.no_grad()
def translate(
    tokenizer,
    model,
    prompts: List[str],
) -> List[str]:

    outputs = []

    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt")
        enc = {k: v.to(DEVICE) for k, v in enc.items()}

        generated = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            temperature=0.0,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

        prompt_len = enc["input_ids"].shape[-1]
        gen_tokens = generated[0][prompt_len:]

        text = tokenizer.decode(
            gen_tokens,
            skip_special_tokens=True
        ).strip()

        outputs.append(text)

    return outputs

# ============================================================
# METRICS
# ============================================================

def compute_metrics(hyps: List[str], refs: List[str]):
    bleu = BLEU().corpus_score(hyps, [refs]).score
    chrf = CHRF().corpus_score(hyps, [refs]).score
    return bleu, chrf

# ============================================================
# EVALUATION
# ============================================================

def evaluate_direction(
    tokenizer,
    model,
    src_lang,
    tgt_lang,
    src_texts,
    ref_texts,
    tag,
):

    prompts = [make_prompt(s, src_lang, tgt_lang) for s in src_texts]

    predictions = []
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        predictions.extend(
            translate(tokenizer, model, prompts[i:i + BATCH_SIZE])
        )

    # ---------- Save CSV ----------
    df = pd.DataFrame({
        "src": src_texts,
        "ref": ref_texts,
        "pred": predictions,
    })
    csv_path = OUT_DIR / f"{tag}.csv"
    df.to_csv(csv_path, index=False)

    # ---------- Save JSONL ----------
    jsonl_path = OUT_DIR / f"{tag}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s, r, p in zip(src_texts, ref_texts, predictions):
            f.write(json.dumps(
                {"src": s, "ref": r, "pred": p},
                ensure_ascii=False
            ) + "\n")

    bleu, chrf = compute_metrics(predictions, ref_texts)

    return bleu, chrf

# ============================================================
# MAIN
# ============================================================

def main():
    tokenizer, model = init_model()
    results = []

    for direction in DIRECTIONS:
        src, tgt = direction.split("_")

        ds = load_dataset(
            DATASET_NAME,
            DATASET_CONFIG,
            split=direction
        )

        if MAX_SAMPLES is not None:
            ds = ds.select(range(min(len(ds), MAX_SAMPLES)))

        # SRC → TGT
        bleu, chrf = evaluate_direction(
            tokenizer, model,
            src, tgt,
            ds["src_txt"], ds["tgt_txt"],
            f"{src}_{tgt}_test"
        )

        results.append({
            "direction": f"{src}_{tgt}",
            "BLEU": bleu,
            "ChrF": chrf
        })

        # TGT → SRC
        bleu_r, chrf_r = evaluate_direction(
            tokenizer, model,
            tgt, src,
            ds["tgt_txt"], ds["src_txt"],
            f"{tgt}_{src}_test"
        )

        results.append({
            "direction": f"{tgt}_{src}",
            "BLEU": bleu_r,
            "ChrF": chrf_r
        })

    pd.DataFrame(results).to_csv(
        OUT_DIR / "summary_metrics.csv", index=False
    )

    print("✅ STRICT Gemma-3 baseline complete")

# ============================================================
if __name__ == "__main__":
    main()
