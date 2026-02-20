# ============================================================
# IMPORTS
# ============================================================
import os
import torch
import json
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
import sacrebleu
import unicodedata
import sys
import shutil

# ============================================================
# 0. SEED & ENV
# ============================================================
set_seed(42)
torch.set_grad_enabled(False)

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_CONFIG = "test"

EVAL_SAMPLES = None
MAX_SRC_LEN = 3500
MAX_TGT_LEN = 3500
BATCH_SIZE = 8
MAX_TOKENS = 3500

OUTPUT_DIR = "baseline_eval_outputs"
Path(OUTPUT_DIR).mkdir(exist_ok=True)

# ============================================================
# 2. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"

# ============================================================
# 3. FILTERS (STRICT, SAME AS TRAINING)
# ============================================================
def strict_filter(example):
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], add_special_tokens=True)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], add_special_tokens=True)["input_ids"])
    return (src_len <= MAX_SRC_LEN) and (tgt_len <= MAX_TGT_LEN)

# ============================================================
# 4. LOAD OFFICIAL TEST DATA
# ============================================================
eval_dataset = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")
eval_dataset = eval_dataset.filter(strict_filter)
eval_dataset = eval_dataset.filter(length_filter)

if EVAL_SAMPLES is not None:
    eval_dataset = eval_dataset.shuffle(seed=42).select(range(EVAL_SAMPLES))

print(f"✅ Evaluation samples: {len(eval_dataset)}")

# ============================================================
# 5. LOAD BASE MODEL (NO TRAINING)
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    attn_implementation="sdpa"
)
model.eval()

# ============================================================
# 6. EVALUATION LOOP (BATCHED, STRICT)
# ============================================================
results = []
metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []}
}

def is_devanagari(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

lid_correct = []
buffer = []

for sample in tqdm(eval_dataset, desc="Evaluating"):
    buffer.extend([
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
    ])

    if len(buffer) < BATCH_SIZE:
        continue

    prompts, meta = [], []
    for mode, instr, src, ref in buffer:
        prompts.append(
            "<start_of_turn>user\n"
            f"{instr}\n{src}"
            "<end_of_turn>\n"
            "<start_of_turn>model\n"
        )
        meta.append((mode, src, ref))

    inputs = tokenizer(prompt, truncation=True, padding=False, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True, 
            temperature=0.1,
            repetition_penalty=1.1
        )

    for i, (mode, src, ref) in enumerate(meta):
        prompt_len = inputs.attention_mask[i].sum().item()
        pred_tokens = outputs[i][prompt_len:]
        pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["preds"].append(pred)
        metrics[mode]["refs"].append(ref)

        if mode == "ENG_to_HIN":
            lid_correct.append(is_devanagari(pred))
        else:
            lid_correct.append(not is_devanagari(pred))

    buffer = []

# ============================================================
# 7. METRICS
# ============================================================
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

e2h_bleu, e2h_chrf = calc_metrics(
    metrics["ENG_to_HIN"]["preds"],
    metrics["ENG_to_HIN"]["refs"]
)
h2e_bleu, h2e_chrf = calc_metrics(
    metrics["HIN_to_ENG"]["preds"],
    metrics["HIN_to_ENG"]["refs"]
)

true_lid_acc = np.mean(lid_correct)

print("\n" + "=" * 50)
print("STRICT ZERO-SHOT BASELINE (BATCHED)")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print(f"Strict Script Accuracy: {true_lid_acc:.2%}")
print("=" * 50)

# ============================================================
# 7B. SAVE METRICS CSV
# ============================================================
metrics_df = pd.DataFrame([
    {"direction": "ENG_to_HIN", "BLEU": e2h_bleu, "chrF": e2h_chrf},
    {"direction": "HIN_to_ENG", "BLEU": h2e_bleu, "chrF": h2e_chrf},
])
metrics_df.to_csv(f"{OUTPUT_DIR}/directional_metrics.csv", index=False)
print(f"📊 Metrics CSV saved → {OUTPUT_DIR}/directional_metrics.csv")

# ============================================================
# 8. SAVE JSON + JSONL
# ============================================================
with open(f"{OUTPUT_DIR}/baseline_eval.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

df = pd.DataFrame(results)

out_dir = Path(f"{OUTPUT_DIR}/jsonl")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin.jsonl"
hin_path = out_dir / "hin_to_eng.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, \
     open(hin_path, "w", encoding="utf-8") as fh:
    for r in results:
        line = json.dumps(
            {"src": r["source"], "ref": r["reference"], "pred": r["prediction"]},
            ensure_ascii=False
        )
        if r["mode"] == "ENG_to_HIN":
            fe.write(line + "\n")
        else:
            fh.write(line + "\n")

print(f"📁 Outputs saved to: {OUTPUT_DIR}")

# ============================================================
# 9. QUALITATIVE CHECK
# ============================================================
print("\n🧪 TOP-10 SAMPLES\n")
for i in range(min(10, len(df))):
    r = df.iloc[i]
    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("-" * 80)

# ============================================================
# ZIP EXPORT
# ============================================================
shutil.make_archive("translation_outputs", "zip", OUTPUT_DIR)
print("\n📦 translation_outputs.zip created")
