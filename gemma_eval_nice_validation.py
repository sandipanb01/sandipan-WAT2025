import os
import torch
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sacrebleu

from tqdm import tqdm
from pathlib import Path
from difflib import unified_diff
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# ----------------------------
# Config
# ----------------------------
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EVAL_DIR   = OUTPUT_DIR / "checkpoint_eval"
DIFF_DIR   = EVAL_DIR / "diffs"

for d in [EVAL_DIR, DIFF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_TGT_LEN = 2400
BLEU_REGRESSION_DROP = 1.0

# ----------------------------
# Tokenizer
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ----------------------------
# Dataset (held-out split)
# ----------------------------
raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
split = raw.train_test_split(test_size=0.1, seed=42)
test_set = split["test"]

# ----------------------------
# Helpers
# ----------------------------
def is_devanagari(text):
    return any("DEVANAGARI" in unicodedata.name(c, "") for c in text)

def calc_metrics(preds, refs):
    return (
        sacrebleu.corpus_bleu(preds, [refs]).score,
        sacrebleu.corpus_chrf(preds, [refs]).score
    )

# ----------------------------
# Evaluation
# ----------------------------
all_stats = {}
all_outputs = {}

ckpts = sorted(os.listdir(CKPT_DIR))

for ckpt in ckpts:
    print(f"\n🔍 Evaluating {ckpt}")
    model = AutoModelForCausalLM.from_pretrained(
        CKPT_DIR / ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    ).eval()

    metrics = {"E2H": {"p": [], "r": []}, "H2E": {"p": [], "r": []}}
    lid = []
    outputs = []

    for s in tqdm(test_set):
        pairs = [
            ("E2H", "Translate to HINDI DEVANAGARI:", s["src_txt"], s["tgt_txt"]),
            ("H2E", "Translate to ENGLISH:", s["tgt_txt"], s["src_txt"]),
        ]

        for mode, instr, src, ref in pairs:
            prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
            inp = tokenizer(prompt, return_tensors="pt").to(model.device)

            with torch.no_grad():
                out = model.generate(
                    **inp,
                    max_new_tokens=MAX_TGT_LEN,
                    do_sample=False,
                    temperature=0.1,
                    repetition_penalty=1.1
                )

            pred = tokenizer.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True).strip()

            metrics[mode]["p"].append(pred)
            metrics[mode]["r"].append(ref)
            outputs.append((mode, src, ref, pred))
            lid.append(is_devanagari(pred) if mode == "E2H" else not is_devanagari(pred))

    e2h_bleu, e2h_chrf = calc_metrics(metrics["E2H"]["p"], metrics["E2H"]["r"])
    h2e_bleu, h2e_chrf = calc_metrics(metrics["H2E"]["p"], metrics["H2E"]["r"])

    all_stats[ckpt] = {
        "ENG→HIN BLEU": round(e2h_bleu, 2),
        "ENG→HIN chrF2": round(e2h_chrf, 2),
        "HIN→ENG BLEU": round(h2e_bleu, 2),
        "HIN→ENG chrF2": round(h2e_chrf, 2),
        "Script Acc (%)": round(np.mean(lid) * 100, 2)
    }

    all_outputs[ckpt] = outputs

# ----------------------------
# Metrics + regression
# ----------------------------
df = pd.DataFrame.from_dict(all_stats, orient="index")
df.to_csv(EVAL_DIR / "checkpoint_metrics.csv")

prev = None
for ckpt, row in df.iterrows():
    if prev and prev - row["ENG→HIN BLEU"] >= BLEU_REGRESSION_DROP:
        print(f"⚠ BLEU regression at {ckpt}: {prev} → {row['ENG→HIN BLEU']}")
    prev = row["ENG→HIN BLEU"]

# ----------------------------
# Plots
# ----------------------------
for metric in df.columns:
    plt.figure()
    plt.plot(df.index, df[metric])
    plt.xticks(rotation=45)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{metric.replace(' ', '_')}.png")
    plt.close()

# ----------------------------
# Side-by-side diffs
# ----------------------------
for i in range(1, len(ckpts)):
    c1, c2 = ckpts[i-1], ckpts[i]
    with open(DIFF_DIR / f"{c1}_vs_{c2}.txt", "w", encoding="utf-8") as f:
        for r1, r2 in zip(all_outputs[c1], all_outputs[c2]):
            if r1[3] != r2[3]:
                diff = unified_diff(
                    r1[3].split(),
                    r2[3].split(),
                    fromfile=c1,
                    tofile=c2,
                    lineterm=""
                )
                f.write("\n".join(diff) + "\n\n")

print("✅ Evaluation complete")
