# ============================================================
# eval.py — STRICT MULTI-CHECKPOINT EVALUATION
# ============================================================

import os
import json
import torch
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from difflib import unified_diff
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu

# ============================================================
# CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EVAL_DIR   = OUTPUT_DIR / "checkpoint_eval"
DIFF_DIR   = EVAL_DIR / "diffs"
PRED_DIR   = EVAL_DIR / "predictions"

for d in [EVAL_DIR, DIFF_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_TGT_LEN = 2400
BLEU_REGRESSION_DROP = 1.0

# ============================================================
# TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)

# ============================================================
# HELD-OUT DATA (SAME SPLIT LOGIC)
# ============================================================
raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
split = raw.train_test_split(test_size=0.1, seed=42)
val_set = split["test"]

# ============================================================
# HELPERS
# ============================================================
def devanagari_ratio(text):
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    return sum("DEVANAGARI" in unicodedata.name(c, "") for c in chars) / len(chars)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

# ============================================================
# CHECKPOINT DISCOVERY
# ============================================================
ckpts = sorted(
    [c for c in os.listdir(CKPT_DIR) if c.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)

all_stats = {}
all_outputs = {}

# ============================================================
# EVALUATION LOOP
# ============================================================
for ckpt in ckpts:
    print(f"\n🔍 Evaluating {ckpt}")

    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model = PeftModel.from_pretrained(base_model, CKPT_DIR / ckpt).eval()

    ckpt_pred_dir = PRED_DIR / ckpt
    ckpt_pred_dir.mkdir(exist_ok=True)

    files = {
        "E2H": open(ckpt_pred_dir / "E2H.jsonl", "w", encoding="utf-8"),
        "H2E": open(ckpt_pred_dir / "H2E.jsonl", "w", encoding="utf-8")
    }

    lid_scores = []

    for idx, s in enumerate(tqdm(val_set)):
        pairs = [
            ("E2H", "Translate to HINDI DEVANAGARI:", s["src_txt"], s["tgt_txt"]),
            ("H2E", "Translate to ENGLISH:", s["tgt_txt"], s["src_txt"])
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

            pred = tokenizer.decode(
                out[0][inp.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            files[mode].write(json.dumps({
                "sample_id": idx,
                "src": src,
                "ref": ref,
                "pred": pred
            }, ensure_ascii=False) + "\n")

            lid_scores.append(
                devanagari_ratio(pred) > 0.6 if mode == "E2H"
                else devanagari_ratio(pred) < 0.4
            )

    for f in files.values():
        f.close()

    e2h = load_jsonl(ckpt_pred_dir / "E2H.jsonl")
    h2e = load_jsonl(ckpt_pred_dir / "H2E.jsonl")

    all_stats[ckpt] = {
        "ENG→HIN BLEU": sacrebleu.corpus_bleu(
            [x["pred"] for x in e2h], [[x["ref"] for x in e2h]]
        ).score,
        "ENG→HIN chrF2": sacrebleu.corpus_chrf(
            [x["pred"] for x in e2h], [[x["ref"] for x in e2h]], beta=2
        ).score,
        "HIN→ENG BLEU": sacrebleu.corpus_bleu(
            [x["pred"] for x in h2e], [[x["ref"] for x in h2e]]
        ).score,
        "HIN→ENG chrF2": sacrebleu.corpus_chrf(
            [x["pred"] for x in h2e], [[x["ref"] for x in h2e]], beta=2
        ).score,
        "Script Acc (%)": np.mean(lid_scores) * 100
    }

    all_outputs[ckpt] = {"E2H": e2h, "H2E": h2e}

# ============================================================
# METRICS + REGRESSION
# ============================================================
df = pd.DataFrame.from_dict(all_stats, orient="index")
df.to_csv(EVAL_DIR / "checkpoint_metrics.csv")

prev = None
for ckpt in ckpts:
    bleu = df.loc[ckpt, "ENG→HIN BLEU"]
    if prev is not None and prev - bleu >= BLEU_REGRESSION_DROP:
        print(f"⚠ BLEU REGRESSION at {ckpt}: {prev:.2f} → {bleu:.2f}")
    prev = bleu

# ============================================================
# METRIC PLOTS
# ============================================================
steps = [int(c.split("-")[-1]) for c in ckpts]

for metric in df.columns:
    plt.figure()
    plt.plot(steps, df[metric].values, marker="o")
    plt.xlabel("Training Steps")
    plt.ylabel(metric)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{metric.replace(' ', '_')}.png")
    plt.close()

# ============================================================
# SIDE-BY-SIDE DIFFS
# ============================================================
for i in range(1, len(ckpts)):
    c1, c2 = ckpts[i - 1], ckpts[i]
    with open(DIFF_DIR / f"{c1}_vs_{c2}.txt", "w", encoding="utf-8") as f:
        for mode in ["E2H", "H2E"]:
            for r1, r2 in zip(all_outputs[c1][mode], all_outputs[c2][mode]):
                if r1["pred"] != r2["pred"]:
                    diff = unified_diff(
                        list(r1["pred"]),
                        list(r2["pred"]),
                        fromfile=f"{c1}-{mode}",
                        tofile=f"{c2}-{mode}",
                        lineterm=""
                    )
                    f.write("".join(diff) + "\n\n")

print("\n✅ EVALUATION COMPLETE")
print(f"📁 Results saved to: {EVAL_DIR}")
