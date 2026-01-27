import os
import json
import math
import torch
import shutil
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
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

TRAIN_CONFIG = "train"
EVAL_CONFIG = "test"

MAX_TRAIN_SAMPLES = 100
EVAL_SAMPLES = 50

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_TOTAL_LEN = MAX_SRC_LEN + MAX_TGT_LEN

CHECKPOINT_STEPS = 100

os.makedirs(OUTPUT_DIR, exist_ok=True)

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
# 4. LOAD DATA
# ============================================================

raw_train = load_dataset(DATASET_NAME, TRAIN_CONFIG, split="eng_hin")
raw_eval  = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")

train_ds = raw_train.filter(strict_filter).filter(length_filter)
eval_ds  = raw_eval.filter(length_filter)

train_ds = train_ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))
eval_ds  = eval_ds.shuffle(seed=99).select(range(EVAL_SAMPLES))

# ============================================================
# 5. BIDIRECTIONAL PROMPT FORMAT
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

train_ds = train_ds.map(formatting_prompts_func, batched=True, remove_columns=train_ds.column_names)
eval_ds  = eval_ds.map(formatting_prompts_func, batched=True, remove_columns=eval_ds.column_names)

# ============================================================
# 6. MODEL + LoRA
# ============================================================

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.float32,
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
# 7. TRAINER (WITH EVAL + CHECKPOINTS)
# ============================================================

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=10,
    max_length=MAX_TOTAL_LEN,
    logging_steps=10,
    eval_steps=CHECKPOINT_STEPS,
    save_steps=CHECKPOINT_STEPS,
    evaluation_strategy="steps",
    save_strategy="steps",
    save_total_limit=10,
    completion_only_loss=True,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=eval_ds,
    peft_config=peft_config,
    args=training_args
)

trainer.train()

# ============================================================
# 7.5 EXTRACT + SAVE TRAIN / VALIDATION LOSS
# ============================================================

trainer_state_path = Path(OUTPUT_DIR) / "trainer_state.json"

with open(trainer_state_path, "r") as f:
    trainer_state = json.load(f)

log_history = trainer_state["log_history"]

train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in log_history:
    if "loss" in entry and "step" in entry:
        train_steps.append(entry["step"])
        train_losses.append(entry["loss"])
    if "eval_loss" in entry and "step" in entry:
        eval_steps.append(entry["step"])
        eval_losses.append(entry["eval_loss"])

loss_df = pd.DataFrame({
    "train_step": train_steps,
    "train_loss": train_losses
})

eval_df = pd.DataFrame({
    "eval_step": eval_steps,
    "eval_loss": eval_losses
})

loss_df.to_csv(Path(OUTPUT_DIR) / "train_loss.csv", index=False)
eval_df.to_csv(Path(OUTPUT_DIR) / "eval_loss.csv", index=False)

plt.figure()
plt.plot(train_steps, train_losses, label="Train Loss", alpha=0.8)
plt.plot(eval_steps, eval_losses, label="Validation Loss", marker="o")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training vs Validation Loss")
plt.legend()
plt.grid(True)
plt.savefig(Path(OUTPUT_DIR) / "train_vs_eval_loss.png")
plt.show()



# ============================================================
# 8. CHECKPOINT EVALUATION + JSONL DUMPS (BIDIRECTIONAL)
# ============================================================

def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return bleu, chrf

checkpoints = sorted(
    Path(OUTPUT_DIR).glob("checkpoint-*"),
    key=lambda x: int(x.name.split("-")[-1])
)

history = []
SANITY_DOC = "This is a simple test sentence to check if the model actually learns translation."

jsonl_root = Path("checkpoint_jsonl")
jsonl_root.mkdir(exist_ok=True)

for ckpt in checkpoints:
    print(f"\n🔍 Evaluating {ckpt.name}")
    model_ckpt = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto")
    model_ckpt.eval()

    eng_srcs, eng_refs, eng_preds = [], [], []
    hin_srcs, hin_refs, hin_preds = [], [], []

    for sample in tqdm(eval_ds, desc=f"Eval {ckpt.name}"):
        pairs = [
            ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["prompt"], sample["completion"]),
            ("HIN_to_ENG", "Translate to ENGLISH:", sample["completion"], sample["prompt"]),
        ]

        for mode, instr, src, ref in pairs:
            prompt = (
                f"<start_of_turn>user\n{instr}\n{src}"
                f"<end_of_turn>\n<start_of_turn>model\n"
            )

            inputs = tokenizer(prompt, return_tensors="pt").to(model_ckpt.device)

            with torch.no_grad():
                output = model_ckpt.generate(
                    **inputs,
                    max_new_tokens=MAX_TGT_LEN,
                    do_sample=False
                )

            pred = tokenizer.decode(
                output[0][inputs.input_ids.shape[-1]:],
                skip_special_tokens=True
            ).strip()

            if mode == "ENG_to_HIN":
                eng_srcs.append(src)
                eng_refs.append(ref)
                eng_preds.append(pred)
            else:
                hin_srcs.append(src)
                hin_refs.append(ref)
                hin_preds.append(pred)

    bleu_eh, chrf_eh = calc_metrics(eng_preds, eng_refs)
    bleu_he, chrf_he = calc_metrics(hin_preds, hin_refs)

    ckpt_dir = jsonl_root / ckpt.name
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    with open(ckpt_dir / "eng_to_hin_src_ref_pred.jsonl", "w", encoding="utf-8") as f:
        for s, r, p in zip(eng_srcs, eng_refs, eng_preds):
            f.write(json.dumps({"src": s, "ref": r, "pred": p}, ensure_ascii=False) + "\n")

    with open(ckpt_dir / "hin_to_eng_src_ref_pred.jsonl", "w", encoding="utf-8") as f:
        for s, r, p in zip(hin_srcs, hin_refs, hin_preds):
            f.write(json.dumps({"src": s, "ref": r, "pred": p}, ensure_ascii=False) + "\n")

    sanity_records = []
    for instr, tag in [
        ("Translate to HINDI DEVANAGARI:", "ENG_to_HIN"),
        ("Translate to ENGLISH:", "HIN_to_ENG"),
    ]:
        sp = f"<start_of_turn>user\n{instr}\n{SANITY_DOC}<end_of_turn>\n<start_of_turn>model\n"
        out = model_ckpt.generate(
            **tokenizer(sp, return_tensors="pt").to(model_ckpt.device),
            max_new_tokens=128
        )
        pred = tokenizer.decode(out[0], skip_special_tokens=True)
        sanity_records.append({"direction": tag, "prediction": pred})

    with open(ckpt_dir / "sanity_checks.jsonl", "w", encoding="utf-8") as f:
        for r in sanity_records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    history.append({
        "checkpoint": ckpt.name,
        "bleu_eng_hin": bleu_eh,
        "chrf_eng_hin": chrf_eh,
        "bleu_hin_eng": bleu_he,
        "chrf_hin_eng": chrf_he
    })

# ============================================================
# 9. PLOTS
# ============================================================
plt.figure()
plt.plot(df_hist["bleu_eng_hin"], label="BLEU ENG→HIN")
plt.plot(df_hist["bleu_hin_eng"], label="BLEU HIN→ENG")
plt.plot(df_hist["chrf_eng_hin"], label="chrF2 ENG→HIN")
plt.plot(df_hist["chrf_hin_eng"], label="chrF2 HIN→ENG")
plt.legend()
plt.title("Checkpoint-wise Translation Quality")
plt.xlabel("Checkpoint")
plt.ylabel("Score")
plt.grid(True)

plt.savefig(Path(OUTPUT_DIR) / "bleu_chrf_vs_checkpoint.png")
plt.show()

