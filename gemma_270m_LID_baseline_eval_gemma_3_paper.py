#IT'S HIGHLY UNLIKELY THAT THIS CODE WITH THE MODEL GEMMA 270M IT WILL PRODUCE TRANSLATTIONS AT ALL.
#BUT WILL GIVE BLOATED SCORES DUE TO SOME SEMANTIC OVERLAPS.
#PLEASE LOOK INTO THIS PAPER https://arxiv.org/pdf/2402.17193 FOR FURTHER DETAILS.

import os

# ============================================================
# Gemma-3-270M-IT ZERO-SHOT MT BASELINE (STRICT PAPER-ALIGNED)
# Includes LangDetect to identify "Bloated" scores from source copying.
# ============================================================

# Install langdetect if not already installed
!pip install langdetect

from pathlib import Path
from typing import List
import json

import torch
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm
from langdetect import detect

from transformers import AutoTokenizer, AutoModelForCausalLM
from sacrebleu.metrics import BLEU, CHRF

# ============================================================
# CONFIG
# ============================================================

MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
DATASET_CONFIG = "test"

DIRECTIONS = ["eng_hin"]          # Reverse handled explicitly
MAX_SAMPLES = 10                  # None = full test set
MAX_NEW_TOKENS = 512
BATCH_SIZE = 1                    # Set to 1 for strict instruction following

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32 # Gemma 3 native dtype

OUT_DIR = Path("./gemma3_strict_baseline")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Map dataset codes to langdetect ISO codes
LANG_MAP = {
    "hin": "hi",
    "eng": "en"
}

# ============================================================
# UTILS
# ============================================================

def check_translation_validity(pred_text: str, target_lang_code: str) -> int:
    """
    Detects if the predicted text matches the intended target language.
    Prevents 'bloated' scores caused by the model simply repeating source text.
    """
    if not pred_text.strip():
        return 0
    try:
        detected = detect(pred_text)
        return 1 if detected == target_lang_code else 0
    except:
        return 0

def make_prompt(src_text: str, src_lang: str, tgt_lang: str) -> str:
    """
    Official Gemma-3 turn format (Minimal/Strict).
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
def translate(tokenizer, model, prompts: List[str]) -> List[str]:
    outputs = []
    for prompt in prompts:
        enc = tokenizer(prompt, return_tensors="pt").to(DEVICE)

        generated = model.generate(
            **enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,        # Deterministic
            temperature=None,       # Required when do_sample=False
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
# EVALUATION
# ============================================================

def evaluate_direction(tokenizer, model, src_lang, tgt_lang, src_texts, ref_texts, tag):
    prompts = [make_prompt(s, src_lang, tgt_lang) for s in src_texts]

    predictions = []
    print(f"--- Running Evaluation: {tag} ---")
    for i in tqdm(range(0, len(prompts), BATCH_SIZE)):
        predictions.extend(
            translate(tokenizer, model, prompts[i:i + BATCH_SIZE])
        )

    # --- VALIDITY CHECK (LID) ---
    iso_code = LANG_MAP.get(tgt_lang, tgt_lang)
    valid_hits = [check_translation_validity(p, iso_code) for p in predictions]
    valid_lang_ratio = sum(valid_hits) / len(predictions) if predictions else 0

    # --- METRICS ---
    bleu = BLEU().corpus_score(predictions, [ref_texts]).score
    chrf = CHRF().corpus_score(predictions, [ref_texts]).score

    # --- SAVE RESULTS ---
    df = pd.DataFrame({
        "src": src_texts,
        "ref": ref_texts,
        "pred": predictions,
        "lang_correct": valid_hits
    })
    df.to_csv(OUT_DIR / f"{tag}.csv", index=False)

    return bleu, chrf, valid_lang_ratio

# ============================================================
# MAIN
# ============================================================

def main():
    tokenizer, model = init_model()
    results = []

    for direction in DIRECTIONS:
        src, tgt = direction.split("_")
        ds = load_dataset(DATASET_NAME, DATASET_CONFIG, split=direction)

        if MAX_SAMPLES is not None:
            ds = ds.select(range(min(len(ds), MAX_SAMPLES)))

        # SRC -> TGT
        bleu, chrf, l_ratio = evaluate_direction(
            tokenizer, model, src, tgt,
            ds["src_txt"], ds["tgt_txt"], f"{src}_{tgt}_test"
        )
        results.append({"direction": f"{src}_{tgt}", "BLEU": bleu, "ChrF": chrf, "LID_Accuracy": l_ratio})

        # TGT -> SRC
        bleu_r, chrf_r, l_ratio_r = evaluate_direction(
            tokenizer, model, tgt, src,
            ds["tgt_txt"], ds["src_txt"], f"{tgt}_{src}_test"
        )
        results.append({"direction": f"{tgt}_{src}", "BLEU": bleu_r, "ChrF": chrf_r, "LID_Accuracy": l_ratio_r})

    summary_df = pd.DataFrame(results)
    summary_df.to_csv(OUT_DIR / "summary_metrics.csv", index=False)

    print("\n" + "="*30)
    print(summary_df)
    print("="*30)
    print(f"\u2705 STRICT Gemma-3 baseline complete. Results saved to {OUT_DIR}")

if __name__ == "__main__":
    main()
