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
torch.use_deterministic_algorithms(True)

# ============================================================
# 1. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EVAL_DIR   = OUTPUT_DIR / "checkpoint_eval"
DIFF_DIR   = EVAL_DIR / "diffs"
PRED_DIR = EVAL_DIR / "predictions"
PRED_DIR.mkdir(parents=True, exist_ok=True)

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
# 8. FINAL MERGED MODEL SAVE
# ============================================================
merged_model = trainer.model.merge_and_unload()
merged_model = merged_model.to("cpu").eval()

FINAL_MODEL_DIR = OUTPUT_DIR / "final_merged"
FINAL_MODEL_DIR.mkdir(exist_ok=True)

merged_model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# ============================================================
# 9. CHECKPOINT EVALUATION
# ============================================================
def devanagari_ratio(text):
    chars = [c for c in text if c.isalpha()]
    if not chars:
        return 0.0
    return sum("DEVANAGARI" in unicodedata.name(c, "") for c in chars) / len(chars)

def load_jsonl(p):
    with open(p, encoding="utf-8") as f:
        return [json.loads(l) for l in f]

ckpts = sorted(
    [c for c in os.listdir(CKPT_DIR) if c.startswith("checkpoint-")],
    key=lambda x: int(x.split("-")[-1])
)

all_stats = {}
all_outputs = {}

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
# 10. METRICS + REGRESSION
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
# 11. METRIC PLOTS (STEP-ALIGNED)
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
# 12. SIDE-BY-SIDE CHARACTER DIFFS
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

print("\n✅ FULLY PATCHED PIPELINE COMPLETE")
print(f"📁 All outputs saved to: {OUTPUT_DIR}")
