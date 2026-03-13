# ============================================================
# IMPORTS
# ============================================================

import torch
import json
import os
import unicodedata
import pandas as pd
import numpy as np
from tqdm import tqdm
from pathlib import Path
import sacrebleu

from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset


# ============================================================
# SETTINGS
# ============================================================

MODEL_DIR = "./gemma3_outputs/final_merged"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = "final_eval_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

MAX_TGT_LEN = 4096
BATCH_SIZE = 4

torch.manual_seed(42)
torch.cuda.manual_seed_all(42)


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading model...")

tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_DIR,
    torch_dtype=torch.bfloat16
).to("cuda")

model.eval()


# ============================================================
# PROMPT BUILDER (DATASET MAP)
# ============================================================

def build_prompt(example, tokenizer):

    pairs = [
        ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", example["src_txt"], example["tgt_txt"]),
        ("HIN_to_ENG", "Translate to ENGLISH:", example["tgt_txt"], example["src_txt"])
    ]

    rows = []

    for mode, instr, src, ref in pairs:

        prompt = (
            f"<start_of_turn>user\n{instr}\n"
            f"{src}<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        tokens = tokenizer(
            prompt,
            truncation=True,
            padding=False
        )

        rows.append({
            "input_ids": tokens["input_ids"],
            "attention_mask": tokens["attention_mask"],
            "reference": ref,
            "mode": mode,
            "source": src
        })

    return rows


# ============================================================
# LOAD DATASET
# ============================================================

print("Loading dataset...")

dataset = load_dataset(
    DATASET_NAME,
    "test",
    split="eng_hin"
)

dataset = dataset.map(
    build_prompt,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=32
)

dataset = dataset.flatten()


# ============================================================
# BATCHED INFERENCE (ADVISOR STYLE)
# ============================================================

def evaluate(model, tokenizer, dataset, batch_size=4):

    predictions = []
    references = []
    modes = []
    sources = []

    print("Running inference...")

    for i in tqdm(range(0, len(dataset), batch_size)):

        batch = dataset[i:i+batch_size]

        padded = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"]
            },
            padding=True,
            return_tensors="pt"
        )

        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)

        with torch.no_grad():

            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_TGT_LEN,
                do_sample=False,
                use_cache=True
            )

        new_tokens = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True
        )

        predictions.extend(decoded)

        references.extend(batch["reference"])
        modes.extend(batch["mode"])
        sources.extend(batch["source"])

    return predictions, references, modes, sources


# ============================================================
# RUN EVALUATION
# ============================================================

preds, refs, modes, sources = evaluate(
    model,
    tokenizer,
    dataset,
    batch_size=BATCH_SIZE
)


# ============================================================
# BUILD RESULTS TABLE
# ============================================================

results = []

metrics = {
    "ENG_to_HIN": {"preds": [], "refs": []},
    "HIN_to_ENG": {"preds": [], "refs": []}
}

for pred, ref, mode, src in zip(preds, refs, modes, sources):

    pred = pred.strip()

    results.append({
        "mode": mode,
        "source": src,
        "reference": ref,
        "prediction": pred
    })

    metrics[mode]["preds"].append(pred)
    metrics[mode]["refs"].append(ref)


# ============================================================
# SCRIPT DETECTION
# ============================================================

def is_devanagari(text):

    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True

    return False


lid_correct = []

for r in results:

    if r["mode"] == "ENG_to_HIN":
        lid_correct.append(is_devanagari(r["prediction"]))
    else:
        lid_correct.append(not is_devanagari(r["prediction"]))

true_lid_acc = np.mean(lid_correct)


# ============================================================
# METRICS
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


print("\n" + "="*60)
print("FINAL METRICS")
print(f"ENG → HIN | BLEU: {e2h_bleu} | chrF: {e2h_chrf}")
print(f"HIN → ENG | BLEU: {h2e_bleu} | chrF: {h2e_chrf}")
print("="*60)
print(f"Script Accuracy: {true_lid_acc:.2%}")
print("="*60)


# ============================================================
# SAVE JSON
# ============================================================

with open("final_eval_strict.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)


# ============================================================
# JSONL EXPORT
# ============================================================

out_dir = Path("exports_jsonl")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

with open(eng_path, "w", encoding="utf-8") as fe, \
     open(hin_path, "w", encoding="utf-8") as fh:

    for r in results:

        line = json.dumps(
            {
                "src": r["source"],
                "ref": r["reference"],
                "pred": r["prediction"]
            },
            ensure_ascii=False
        )

        if r["mode"] == "ENG_to_HIN":
            fe.write(line + "\n")
        else:
            fh.write(line + "\n")


# ============================================================
# EXCEL METRICS
# ============================================================

summary_rows = []

for direction in ["ENG_to_HIN", "HIN_to_ENG"]:

    subset = [r for r in results if r["mode"] == direction]

    preds = [r["prediction"] for r in subset]
    refs = [r["reference"] for r in subset]

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    lid_hits = []

    for p in preds:

        dev = is_devanagari(p)
        correct = dev if direction == "ENG_to_HIN" else not dev

        lid_hits.append(correct)

    script_acc = np.mean(lid_hits) * 100

    summary_rows.append({
        "Direction": direction.replace("_"," "),
        "BLEU": round(bleu,2),
        "chrF": round(chrf,2),
        "Script Accuracy (%)": round(script_acc,2),
        "Total Samples": len(subset)
    })


metrics_df = pd.DataFrame(summary_rows)

excel_path = os.path.join(
    OUTPUT_DIR,
    "final_translation_metrics.xlsx"
)

metrics_df.to_excel(excel_path, index=False)


print("\nMetrics saved to:")
print(excel_path)


# ============================================================
# SAMPLE OUTPUTS
# ============================================================

print("\nTop-10 examples\n")

for i in range(min(10, len(results))):

    r = results[i]

    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("-"*80)
