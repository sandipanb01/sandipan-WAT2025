import os
import json
import torch
import shutil
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher, unified_diff
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    set_seed
)
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ============================================================
# 0. REPRODUCIBILITY (LeCun-style strict determinism)
# ============================================================
set_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EVAL_DIR   = OUTPUT_DIR / "checkpoint_eval"
DIFF_DIR   = EVAL_DIR / "diffs"

for d in [OUTPUT_DIR, CKPT_DIR, EVAL_DIR, DIFF_DIR]:
    d.mkdir(parents=True, exist_ok=True)

MAX_TRAIN_SAMPLES = None
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN

BLEU_REGRESSION_DROP = 1.0   # strict threshold

# ============================================================
# 2. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3. STRICT DATA FILTERING (ANTI-CHEATING)
# ============================================================
def strict_filter(example):
    sim = SequenceMatcher(
        None,
        example["src_txt"].lower(),
        example["tgt_txt"].lower()
    ).ratio()
    return sim < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
raw = raw.filter(strict_filter)
raw = raw.filter(length_filter)

if MAX_TRAIN_SAMPLES is None:
    limit = len(raw)
else:
    limit = min(MAX_TRAIN_SAMPLES, len(raw))

raw = raw.shuffle(seed=42).select(range(limit))

split = raw.train_test_split(test_size=0.1, seed=42)
train_set = split["train"]
val_set   = split["test"]

print(f"Train: {len(train_set)} | Val: {len(val_set)}")

# ============================================================
# 4. BIDIRECTIONAL PROMPT FORMAT (STRICT PARITY)
# ============================================================
def format_fn(batch):
    prompts, completions = [], []
    for i in range(len(batch["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", batch["src_txt"][i], batch["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", batch["tgt_txt"][i], batch["src_txt"][i]

        prompts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        )
        completions.append(f"{tgt}<end_of_turn>")

    return {"prompt": prompts, "completion": completions}

train_ds = train_set.map(format_fn, batched=True, remove_columns=train_set.column_names)
val_ds   = val_set.map(format_fn, batched=True, remove_columns=val_set.column_names)

# ============================================================
# 5. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    target_modules=[
        "q_proj", "k_proj", "v_proj",
        "o_proj", "gate_proj", "up_proj", "down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 6. TRAINER (CHECKPOINTS + VALIDATION)
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=val_ds,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=10,

        eval_strategy="steps",
        eval_steps=100,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=10,

        max_length=MAX_SEQ_LEN,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        completion_only_loss=True,
        report_to="none"
    )
)

trainer.train()

# ============================================================
# 7. LOSS CURVES (TRAIN + VAL)
# ============================================================
logs = trainer.state.log_history
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
val_loss   = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

plt.figure()
plt.plot(*zip(*train_loss), label="Train Loss")
plt.plot(*zip(*val_loss), label="Val Loss")
plt.legend()
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.tight_layout()
plt.savefig(OUTPUT_DIR / "loss_curve.png")
plt.close()

# ============================================================
# 8. CHECKPOINT-WISE STRICT EVALUATION
# ============================================================
def is_devanagari(text):
    for ch in text:
        if "DEVANAGARI" in unicodedata.name(ch, ""):
            return True
    return False

def calc_metrics(preds, refs):
    return (
        sacrebleu.corpus_bleu(preds, [refs]).score,
        sacrebleu.corpus_chrf(preds, [refs]).score
    )

test_set = val_set  # advisor-safe: held-out split

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

    results = []
    metrics = {"E2H": {"p": [], "r": []}, "H2E": {"p": [], "r": []}}
    lid = []

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
            results.append((mode, src, ref, pred))
            metrics[mode]["p"].append(pred)
            metrics[mode]["r"].append(ref)

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

    all_outputs[ckpt] = results

# ============================================================
# 9. METRIC TABLE + REGRESSION ALERTS
# ============================================================
df = pd.DataFrame.from_dict(all_stats, orient="index")
df.to_csv(EVAL_DIR / "checkpoint_metrics.csv")

prev = None
for ckpt, row in df.iterrows():
    if prev and prev - row["ENG→HIN BLEU"] >= BLEU_REGRESSION_DROP:
        print(f" BLEU REGRESSION at {ckpt}: {prev} → {row['ENG→HIN BLEU']}")
    prev = row["ENG→HIN BLEU"]

# ============================================================
# 10. PLOTS
# ============================================================
for metric in ["ENG→HIN BLEU", "HIN→ENG BLEU", "ENG→HIN chrF2", "HIN→ENG chrF2"]:
    plt.figure()
    plt.plot(df.index, df[metric])
    plt.xticks(rotation=45)
    plt.title(metric)
    plt.tight_layout()
    plt.savefig(EVAL_DIR / f"{metric.replace(' ', '_')}.png")
    plt.close()

# ============================================================
# 11. SIDE-BY-SIDE CHECKPOINT DIFFS
# ============================================================
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

print("\n✅ END-TO-END PIPELINE COMPLETE")
print(f"📁 Outputs: {OUTPUT_DIR}")
