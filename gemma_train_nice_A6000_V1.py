# ============================================================
# 0. GLOBAL UTF-8 SAFETY (CRITICAL)
# ============================================================

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="ignore")
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="ignore")

import os
import json
import torch
import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # UTF-safe, no GUI backend
import matplotlib.pyplot as plt
plt.rcParams["font.family"] = "DejaVu Sans"

from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ----------------------------
# Reproducibility
# ----------------------------
set_seed(42)
torch.manual_seed(42)

# ============================================================
# 1. CONFIGURATION
# ============================================================

MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-strict-bidirectional"

USE_FULL_DATA = True
VAL_RATIO = 0.1

MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_TOTAL_LEN = MAX_SRC_LEN + MAX_TGT_LEN

CHECKPOINT_STEPS = 500
SANITY_SUBSET_SIZE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
SANITY_LOG_DIR = Path(OUTPUT_DIR) / "sanity_check_logs"
SANITY_LOG_DIR.mkdir(exist_ok=True)

# ============================================================
# 2. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3. STRICT FILTERING
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

# ============================================================
# 4. LOAD + SPLIT
# ============================================================

raw_ds = load_dataset(DATASET_NAME, "train", split="eng_hin")

filtered_ds = raw_ds.filter(strict_filter, batched=False, load_from_cache_file=False)
filtered_ds = filtered_ds.filter(length_filter, batched=False, load_from_cache_file=False)

if MAX_TRAIN_SAMPLES is not None:
    filtered_ds = filtered_ds.select(range(min(MAX_TRAIN_SAMPLES, len(filtered_ds))))

if USE_FULL_DATA:
    split = filtered_ds.train_test_split(test_size=VAL_RATIO, seed=42, shuffle=True)
    train_ds = split["train"]
    eval_ds = split["test"]
else:
    train_ds = filtered_ds
    eval_ds = load_dataset(DATASET_NAME, "test", split="eng_hin") \
        .filter(length_filter, batched=False, load_from_cache_file=False)

if MAX_EVAL_SAMPLES is not None:
    eval_ds = eval_ds.select(range(min(MAX_EVAL_SAMPLES, len(eval_ds))))

# ============================================================
# 5. BIDIRECTIONAL PROMPTS
# ============================================================

def formatting_prompts_func(example):
    prompts, completions = [], []
    for i in range(len(example["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", example["src_txt"][i], example["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", example["tgt_txt"][i], example["src_txt"][i]

        prompts.append(
            f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        )
        completions.append(f"{tgt}<end_of_turn>")

    return {"prompt": prompts, "completion": completions}

train_ds = train_ds.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_ds.column_names,
    load_from_cache_file=False,
    writer_batch_size=50
)

eval_ds = eval_ds.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=eval_ds.column_names,
    load_from_cache_file=False,
    writer_batch_size=50
)

# ============================================================
# 6. MODEL + LoRA
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="flash_attention_2" if torch.cuda.is_available() else "eager"
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
# 7. SANITY CHECK (UTF SAFE)
# ============================================================

def safe_jsonl_write(path, records):
    with open(path, "w", encoding="utf-8", errors="ignore") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def run_sanity_subset(model, tokenizer, device, eval_subset, log_path=None):
    sanity_records = []
    for s in eval_subset:
        prompt = s["prompt"]
        ref = s["completion"].replace("<end_of_turn>", "").strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(device)
        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )

        pred = tokenizer.decode(
            output[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        sanity_records.append({
            "prompt": prompt.splitlines()[1],
            "predicted": pred,
            "reference": ref
        })

    if log_path:
        safe_jsonl_write(log_path, sanity_records)

class SanityCheckCallback(TrainerCallback):
    def __init__(self, eval_subset):
        self.eval_subset = eval_subset

    def on_evaluate(self, args, state, control, **kwargs):
        log_file = SANITY_LOG_DIR / f"sanity_step_{state.global_step}.jsonl"
        run_sanity_subset(
            kwargs["model"],
            kwargs["tokenizer"],
            kwargs["model"].device,
            self.eval_subset,
            log_path=log_file
        )

eval_subset = [eval_ds[i] for i in range(min(SANITY_SUBSET_SIZE, len(eval_ds)))]

# ============================================================
# 8. TRAINER
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    num_train_epochs=2,
    max_length=MAX_TOTAL_LEN,
    logging_steps=10,
    eval_steps=CHECKPOINT_STEPS,
    save_steps=CHECKPOINT_STEPS,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=500,
    completion_only_loss=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    args=training_args,
    callbacks=[SanityCheckCallback(eval_subset)]
)

trainer.train()

# ============================================================
# 9. LOSS LOGGING (UTF SAFE)
# ============================================================

log_history = trainer.state.log_history
train_steps, train_losses, eval_steps, eval_losses = [], [], [], []

for e in log_history:
    if "loss" in e and "eval_loss" not in e:
        train_steps.append(e["step"])
        train_losses.append(e["loss"])
    if "eval_loss" in e:
        eval_steps.append(e["step"])
        eval_losses.append(e["eval_loss"])

pd.DataFrame(
    {"step": train_steps, "train_loss": train_losses}
).to_csv(Path(OUTPUT_DIR) / "train_loss.csv", index=False, encoding="utf-8")

pd.DataFrame(
    {"step": eval_steps, "eval_loss": eval_losses}
).to_csv(Path(OUTPUT_DIR) / "eval_loss.csv", index=False, encoding="utf-8")

plt.figure()
plt.plot(train_steps, train_losses)
plt.grid(True)
plt.savefig(Path(OUTPUT_DIR) / "training_loss.png")
plt.close()

plt.figure()
plt.plot(eval_steps, eval_losses)
plt.grid(True)
plt.savefig(Path(OUTPUT_DIR) / "validation_loss.png")
plt.close()

# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================

final_dir = Path(OUTPUT_DIR) / "final_model"
final_dir.mkdir(exist_ok=True)

final_model = trainer.model.merge_and_unload()
final_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)

# ============================================================
# 11. CHECKPOINT EVAL + JSONL (UTF SAFE)
# ============================================================

def calc_metrics(preds, refs):
    refs = [r.replace("<end_of_turn>", "").strip() for r in refs]
    return (
        sacrebleu.corpus_bleu(preds, [refs]).score,
        sacrebleu.corpus_chrf(preds, [refs]).score
    )

checkpoints = sorted(
    Path(OUTPUT_DIR).glob("checkpoint-*"),
    key=lambda x: int(x.name.split("-")[-1])
)

history = []

for ckpt in checkpoints:
    model_ckpt = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto")
    model_ckpt.eval()

    eng_hin_preds, eng_hin_refs = [], []
    hin_eng_preds, hin_eng_refs = [], []
    records = []

    for s in tqdm(eval_ds, leave=False):
        prompt = s["prompt"]
        ref = s["completion"].replace("<end_of_turn>", "").strip()

        inputs = tokenizer(prompt, return_tensors="pt").to(model_ckpt.device)
        with torch.no_grad():
            out = model_ckpt.generate(
                **inputs,
                max_new_tokens=MAX_TGT_LEN,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )

        pred = tokenizer.decode(
            out[0][inputs.input_ids.shape[-1]:],
            skip_special_tokens=True
        ).strip()

        if "HINDI DEVANAGARI" in prompt:
            eng_hin_preds.append(pred)
            eng_hin_refs.append(ref)
            direction = "ENG_to_HIN"
        else:
            hin_eng_preds.append(pred)
            hin_eng_refs.append(ref)
            direction = "HIN_to_ENG"

        records.append({
            "direction": direction,
            "src": prompt,
            "ref": ref,
            "pred": pred
        })

    bleu_eh, chrf_eh = calc_metrics(eng_hin_preds, eng_hin_refs)
    bleu_he, chrf_he = calc_metrics(hin_eng_preds, hin_eng_refs)

    history.append({
        "checkpoint": ckpt.name,
        "bleu_eng_hin": bleu_eh,
        "chrf_eng_hin": chrf_eh,
        "bleu_hin_eng": bleu_he,
        "chrf_hin_eng": chrf_he
    })

    ckpt_jsonl = Path(OUTPUT_DIR) / "checkpoint_jsonl" / f"{ckpt.name}_translations.jsonl"
    ckpt_jsonl.parent.mkdir(exist_ok=True)
    safe_jsonl_write(ckpt_jsonl, records)

df_hist = pd.DataFrame(history)
df_hist.to_csv(
    Path(OUTPUT_DIR) / "checkpoint_translation_metrics.csv",
    index=False,
    encoding="utf-8"
)

plt.figure()
plt.plot(df_hist["bleu_eng_hin"], label="BLEU ENG→HIN")
plt.plot(df_hist["bleu_hin_eng"], label="BLEU HIN→ENG")
plt.plot(df_hist["chrf_eng_hin"], label="chrF ENG→HIN")
plt.plot(df_hist["chrf_hin_eng"], label="chrF HIN→ENG")
plt.legend()
plt.grid(True)
plt.savefig(Path(OUTPUT_DIR) / "bleu_chrf_vs_checkpoint.png")
plt.close()
