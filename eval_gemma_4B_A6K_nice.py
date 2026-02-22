# ============================================================
# EVAL SCRIPT — STRICT (BATCHED INFERENCE)
# ============================================================

import json
import torch
import unicodedata
import numpy as np
import pandas as pd

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
import sacrebleu

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_DIR = "./gemma3_outputs/final_merged"
DATASET_NAME = "ai4bharat/pralekha"

SRC_LANG = "eng"
TGT_LANG = "hin"

OUTPUT_DIR = Path("./gemma3_outputs")
BATCH_SIZE_INFER = 4
MAX_TOKENS = 3500

# ============================================================
# 2. TOKENIZER + MODEL
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "left"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16,
    device_map="auto"
).eval()

# ============================================================
# 3. LOAD TEST DATA (OFFICIAL)
# ============================================================
test_raw = load_dataset(
    DATASET_NAME,
    data_dir="test",
    split="train"
)

test_raw = test_raw.filter(
    lambda x: (
        x["src_lang"] == SRC_LANG and
        x["tgt_lang"] == TGT_LANG and
        x["src_txt"].strip() and
        x["tgt_txt"].strip()
    ),
    num_proc=4
)

print(f"Test samples: {len(test_raw)}")

# ============================================================
# 4. HELPERS
# ============================================================
def is_devanagari(txt):
    return any("DEVANAGARI" in unicodedata.name(c, "") for c in txt)

# ============================================================
# 5. BUILD INFERENCE PAIRS
# ============================================================
pairs = []
for s in test_raw:
    pairs.append(("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", s["src_txt"], s["tgt_txt"]))
    pairs.append(("HIN_to_ENG", "Translate to ENGLISH:", s["tgt_txt"], s["src_txt"]))

results = []
metrics = {"ENG_to_HIN": {"p": [], "r": []}, "HIN_to_ENG": {"p": [], "r": []}}

# ============================================================
# 6. BATCHED INFERENCE
# ============================================================
for i in tqdm(range(0, len(pairs), BATCH_SIZE_INFER)):
    batch = pairs[i:i+BATCH_SIZE_INFER]

    prompts = [
        f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        for _, instr, src, _ in batch
    ]

    enc = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)

    with torch.no_grad():
        outs = model.generate(
            **enc,
            max_new_tokens=MAX_TOKENS,
            do_sample=False,
            use_cache=True
        )

    for j, (mode, _, src, ref) in enumerate(batch):
        gen = outs[j][enc["input_ids"].shape[1]:]
        pred = tokenizer.decode(gen, skip_special_tokens=True)

        results.append({
            "mode": mode,
            "source": src,
            "reference": ref,
            "prediction": pred
        })

        metrics[mode]["p"].append(pred)
        metrics[mode]["r"].append(ref)

# ============================================================
# 7. METRICS + EXPORT
# ============================================================
def score(p, r):
    return sacrebleu.corpus_bleu(p, [r]).score, sacrebleu.corpus_chrf(p, [r]).score

summary = []
for k in metrics:
    bleu, chrf = score(metrics[k]["p"], metrics[k]["r"])
    lid = np.mean([
        is_devanagari(p) if k == "ENG_to_HIN" else not is_devanagari(p)
        for p in metrics[k]["p"]
    ])
    summary.append([k, round(bleu,2), round(chrf,2), round(lid*100,2)])

df = pd.DataFrame(summary, columns=["Direction","BLEU","chrF","ScriptAcc"])
df.to_excel(OUTPUT_DIR / "final_translation_report.xlsx", index=False)

with open(OUTPUT_DIR / "final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("✅ EVALUATION COMPLETE — METRICS EXPORTED")
