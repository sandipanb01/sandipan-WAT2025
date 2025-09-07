#!/usr/bin/env python
"""
Iterative Back-Translation (IBT) with Pralekha
- Runs document-level IBT on all language pairs (bidirectional)
- Generates synthetic data + merges with gold Pralekha
- Outputs ready-to-train Hugging Face DatasetDict
"""

import os
import json
import torch
from pathlib import Path
from datasets import load_dataset, Dataset, DatasetDict, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM

# -------------------------
# Config
# -------------------------
MODEL_PATH = "./gemma3-1b-pt-indicdoc"  # your fine-tuned model
OUTPUT_DIR = Path("./ibt_augmented")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal",
    "eng_mar", "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "hin_eng", "tam_eng", "urd_eng"
]

MAX_TOKENS = 4096
MAX_DOCS = 200     # limit per pair for speed/debug
N_ITER = 2         # number of back-translation cycles
SPLIT = "train"

# -------------------------
# Helpers
# -------------------------
def load_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH)
    model = AutoModelForCausalLM.from_pretrained(MODEL_PATH, device_map="auto")
    return tokenizer, model

def translate_doc(tokenizer, model, text, src, tgt, max_tokens=MAX_TOKENS):
    """Document-level translation"""
    prompt = f"""<start_of_turn>user
Translate this {src} document to {tgt}:
{text}<end_of_turn>
<start_of_turn>model"""
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_tokens).to(model.device)
    outputs = model.generate(**inputs, max_new_tokens=512, do_sample=False)
    decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

    if "<start_of_turn>model" in decoded:
        decoded = decoded.split("<start_of_turn>model")[-1].strip()
    if "<end_of_turn>" in decoded:
        decoded = decoded.split("<end_of_turn>")[0].strip()
    return decoded

def backtranslate_corpus(tokenizer, model, src_lang, tgt_lang, split=SPLIT, max_docs=MAX_DOCS):
    """Run forward+back translation for one language pair"""
    print(f"[IBT] {src_lang} → {tgt_lang} → {src_lang}")
    try:
        ds = load_dataset("ai4bharat/Pralekha", split=f"{src_lang}_{tgt_lang}", name=split)
    except Exception as e:
        print(f"  [WARN] Could not load {src_lang}_{tgt_lang}: {e}")
        return []

    ds = ds.select(range(min(max_docs, len(ds))))

    synthetic_pairs = []
    for row in ds:
        src_text = row.get("src_txt") or row.get("src_text", "")
        tgt_text = row.get("tgt_txt") or row.get("tgt_text", "")
        if not src_text or not tgt_text:
            continue

        # Forward translate source → target
        fwd = translate_doc(tokenizer, model, src_text, src_lang, tgt_lang)

        # Back translate target → source
        back = translate_doc(tokenizer, model, fwd, tgt_lang, src_lang)

        synthetic_pairs.append({
            "src_txt": back,
            "tgt_txt": fwd,
            "original_src": src_text,
            "original_tgt": tgt_text,
            "lang_pair": f"{src_lang}_{tgt_lang}"
        })

    return synthetic_pairs

# -------------------------
# Main IBT loop + dataset merge
# -------------------------
def main():
    tokenizer, model = load_model()
    augmented_data = []

    for it in range(N_ITER):
        print(f"\n========== Iteration {it+1}/{N_ITER} ==========")
        for pair in LANGUAGE_PAIRS:
            src, tgt = pair.split("_")
            synthetic = backtranslate_corpus(tokenizer, model, src, tgt, split=SPLIT)
            if synthetic:
                augmented_data.extend(synthetic)
                out_file = OUTPUT_DIR / f"{pair}_ibt_iter{it+1}.jsonl"
                with open(out_file, "w", encoding="utf-8") as f:
                    for item in synthetic:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")
                print(f"  Saved {len(synthetic)} synthetic pairs → {out_file}")

    # -------------------------
    # Merge with gold Pralekha
    # -------------------------
    print("\n[INFO] Merging synthetic + gold Pralekha...")
    gold_datasets = []
    for pair in LANGUAGE_PAIRS:
        try:
            ds_gold = load_dataset("ai4bharat/Pralekha", split=f"{pair}", name=SPLIT)
            gold_datasets.append(ds_gold)
        except Exception as e:
            print(f"  [WARN] Could not load gold {pair}: {e}")

    # Convert synthetic list → Dataset
    synthetic_dataset = Dataset.from_list(augmented_data) if augmented_data else None

    if synthetic_dataset and gold_datasets:
        merged = concatenate_datasets(gold_datasets + [synthetic_dataset])
    elif synthetic_dataset:
        merged = synthetic_dataset
    else:
        merged = concatenate_datasets(gold_datasets)

    dataset_dict = DatasetDict({"train": merged})
    out_path = OUTPUT_DIR / f"ibt_augmented_dataset_iter{N_ITER}"
    dataset_dict.save_to_disk(str(out_path))

    print(f"[INFO] Finished IBT. Total synthetic pairs: {len(augmented_data)}")
    print(f"[INFO] Merged dataset saved to {out_path}")

if __name__ == "__main__":
    main()
