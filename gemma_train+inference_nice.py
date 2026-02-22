# ============================================================
# 0. IMPORTS
# ============================================================
import os
import sys
import json
import torch
import shutil
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from difflib import SequenceMatcher
from datasets import load_dataset, Value
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ============================================================
# 1. REPRODUCIBILITY (STRICT)
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG (STRICTLY FROM SCRIPT B)
# ============================================================
MODEL_ID = "google/gemma-3-4b-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EXPORT_DIR = OUTPUT_DIR / "exports_jsonl"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)
EXPORT_DIR.mkdir(exist_ok=True)

MAX_TRAIN_SAMPLES = None
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_TOKENS = 3500
MAX_SEQ_LEN = MAX_SRC_LEN + MAX_TGT_LEN

# ============================================================
# 3. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
#tokenizer.padding_side = "right"

# ============================================================
# 4. STRICT FILTERS (ANTI-CHEATING)
# ============================================================
def strict_filter(example):
    s = example["src_txt"].lower()
    t = example["tgt_txt"].lower()
    return SequenceMatcher(None, s, t).ratio() < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

def clean_utf8(example):
    example["src_txt"] = example["src_txt"].decode("utf-8", "ignore")
    example["tgt_txt"] = example["tgt_txt"].decode("utf-8", "ignore")
    return example

# ============================================================
# 5. LOAD TRAIN + DEV + TEST (OFFICIAL)
# ============================================================
print("Loading Pralekha splits...")

train_raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
dev_raw   = load_dataset(DATASET_NAME, "dev",   split="eng_hin")
test_raw  = load_dataset(DATASET_NAME, "test",  split="eng_hin")

def preprocess(ds):
    ds = ds.cast_column("src_txt", Value("binary"))
    ds = ds.cast_column("tgt_txt", Value("binary"))
    ds = ds.map(clean_utf8, num_proc=32)
    ds = ds.cast_column("src_txt", Value("string"))
    ds = ds.cast_column("tgt_txt", Value("string"))
    ds = ds.filter(lambda x: x["src_txt"].strip() and x["tgt_txt"].strip())
    ds = ds.filter(strict_filter)
    ds = ds.filter(length_filter)
    return ds

train_raw = preprocess(train_raw)
dev_raw   = preprocess(dev_raw)
test_raw  = preprocess(test_raw)

if MAX_TRAIN_SAMPLES:
    train_raw = train_raw.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))

print(f"Train: {len(train_raw)} | Dev: {len(dev_raw)} | Test: {len(test_raw)}")

# ============================================================
# 6. BIDIRECTIONAL PROMPT FORMAT (STRICT PARITY)
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

train_ds = train_raw.map(format_fn, batched=True, remove_columns=train_raw.column_names)
dev_ds   = dev_raw.map(format_fn,   batched=True, remove_columns=dev_raw.column_names)

# ============================================================
# 7. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj",
        "o_proj","gate_proj","up_proj","down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 8. TRAINER (OFFICIAL DEV)
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        eval_strategy="steps",
        eval_steps=2000,
        save_steps=2000,
        save_total_limit=10,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LEN,
        completion_only_loss=True,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        packing=False,
        report_to="none"
    )
)

trainer.train()

# ============================================================
# 9. LOSS CURVE
# ============================================================
logs = trainer.state.log_history
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]
val_loss   = [(x["step"], x["eval_loss"]) for x in logs if "eval_loss" in x]

plt.plot(*zip(*train_loss), label="Train")
plt.plot(*zip(*val_loss), label="Dev")
plt.legend()
plt.savefig(OUTPUT_DIR / "loss_curve.png")
plt.close()

# ============================================================
# 10. MERGE FINAL MODEL
# ============================================================
model = trainer.model.merge_and_unload().eval()
FINAL_MODEL_DIR = OUTPUT_DIR / "final_merged"
FINAL_MODEL_DIR.mkdir(exist_ok=True)
model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)

# ============================================================
# 11. STRICT EVALUATION (TEST)
# ============================================================
def is_devanagari(txt):
    return any("DEVANAGARI" in unicodedata.name(c, "") for c in txt)

results, metrics = [], {"ENG_to_HIN": {"p": [], "r": []}, "HIN_to_ENG": {"p": [], "r": []}}

for s in tqdm(test_raw):
    pairs = [
        ("ENG_to_HIN","Translate to HINDI DEVANAGARI:", s["src_txt"], s["tgt_txt"]),
        ("HIN_to_ENG","Translate to ENGLISH:", s["tgt_txt"], s["src_txt"])
    ]
    for mode, instr, src, ref in pairs:
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
        inp = tokenizer(prompt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=MAX_TOKENS, use_cache=True, do_sample=False)
        pred = tokenizer.decode(out[0][inp.input_ids.shape[-1]:], skip_special_tokens=True)

        results.append({"mode":mode,"source":src,"reference":ref,"prediction":pred})
        metrics[mode]["p"].append(pred)
        metrics[mode]["r"].append(ref)

# ============================================================
# 12. METRICS + EXPORTS
# ============================================================
def score(p,r): 
    return sacrebleu.corpus_bleu(p,[r]).score, sacrebleu.corpus_chrf(p,[r]).score

summary = []
for k in metrics:
    b,c = score(metrics[k]["p"], metrics[k]["r"])
    lid = np.mean([(is_devanagari(p) if k=="ENG_to_HIN" else not is_devanagari(p)) for p in metrics[k]["p"]])
    summary.append([k,b,c,lid*100])

df = pd.DataFrame(summary, columns=["Direction","BLEU","chrF","ScriptAcc"])
df.to_excel(OUTPUT_DIR / "final_translation_report.xlsx", index=False)

with open(OUTPUT_DIR / "final_eval_strict.json","w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False,indent=2)

print("\n END-TO-END STRICT PIPELINE COMPLETE")
