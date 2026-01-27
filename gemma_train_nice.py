import os
import json
import torch
import random
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from difflib import SequenceMatcher
from itertools import islice, tee

from datasets import load_dataset, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ----------------------------
# 0. Reproducibility
# ----------------------------
set_seed(42)
torch.manual_seed(42)
random.seed(42)

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
CHECKPOINT_STEPS = 100
SANITY_SUBSET_SIZE = 5

os.makedirs(OUTPUT_DIR, exist_ok=True)
SANITY_LOG_DIR = Path(OUTPUT_DIR) / "sanity_check_logs"
SANITY_LOG_DIR.mkdir(exist_ok=True, parents=True)

# ============================================================
# 2. TOKENIZER
# ============================================================

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 3. FILTER FUNCTIONS
# ============================================================

def strict_filter(example):
    return SequenceMatcher(None, example["src_txt"].lower(), example["tgt_txt"].lower()).ratio() < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

# ============================================================
# 4. STREAMING DATASET WITH OPTIONAL TRAIN/VAL SPLIT
# ============================================================

def streaming_dataset(split_name, max_samples=None, val_ratio=0.0, use_full_data=True):
    ds_stream = load_dataset(DATASET_NAME, "train" if split_name=="train" else "test",
                              split="eng_hin", streaming=True)
    filtered = (x for x in ds_stream if strict_filter(x) and length_filter(x))
    if max_samples:
        filtered = islice(filtered, max_samples)

    if use_full_data and split_name=="train":
        # Split into train/val streams on-the-fly
        train_stream, val_stream = tee(filtered, 2)
        train_stream = (x for x in train_stream if random.random() >= val_ratio)
        val_stream = (x for x in val_stream if random.random() < val_ratio)
        return list(train_stream), list(val_stream)
    else:
        return list(filtered)

if USE_FULL_DATA:
    train_ds, eval_ds = streaming_dataset("train", max_samples=MAX_TRAIN_SAMPLES,
                                          val_ratio=VAL_RATIO, use_full_data=True)
else:
    train_ds = streaming_dataset("train", max_samples=MAX_TRAIN_SAMPLES, use_full_data=False)
    eval_ds = streaming_dataset("test", max_samples=MAX_EVAL_SAMPLES, use_full_data=False)

# ============================================================
# 5. BIDIRECTIONAL PROMPT FORMATTING
# ============================================================

def format_bidirectional(example):
    prompts, completions = [], []
    src_txt = example["src_txt"] if isinstance(example["src_txt"], list) else [example["src_txt"]]
    tgt_txt = example["tgt_txt"] if isinstance(example["tgt_txt"], list) else [example["tgt_txt"]]
    for i in range(len(src_txt)):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", src_txt[i], tgt_txt[i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", tgt_txt[i], src_txt[i]
        prompts.append(f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n")
        completions.append(f"{tgt}<end_of_turn>")
    return {"prompt": prompts, "completion": completions}

train_ds = [format_bidirectional(x) for x in train_ds]
eval_ds  = [format_bidirectional(x) for x in eval_ds]

# Flatten lists for HF Dataset
train_prompts = [p for ex in train_ds for p in ex["prompt"]]
train_completions = [c for ex in train_ds for c in ex["completion"]]
eval_prompts = [p for ex in eval_ds for p in ex["prompt"]]
eval_completions = [c for ex in eval_ds for c in ex["completion"]]

train_ds = Dataset.from_dict({"prompt": train_prompts, "completion": train_completions})
eval_ds  = Dataset.from_dict({"prompt": eval_prompts, "completion": eval_completions})

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
# 7. SANITY CHECK CALLBACK
# ============================================================

def run_sanity_subset(model, tokenizer, device, eval_subset, log_path=None):
    sanity_records = []
    print("\n=== SANITY CHECK ===")
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
        pred = tokenizer.decode(output[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        print(f"Prompt: {prompt.splitlines()[1]}")
        print(f"Predicted: {pred}")
        print(f"Reference: {ref}\n")
        sanity_records.append({"prompt": prompt.splitlines()[1], "predicted": pred, "reference": ref})
    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in sanity_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

class SanityCheckCallback(TrainerCallback):
    def __init__(self, eval_subset):
        self.eval_subset = eval_subset
    def on_evaluate(self, args, state, control, **kwargs):
        log_file = SANITY_LOG_DIR / f"sanity_step_{state.global_step}.jsonl"
        run_sanity_subset(kwargs["model"], kwargs["tokenizer"], kwargs["model"].device,
                           self.eval_subset, log_path=log_file)

eval_subset = eval_ds.select(range(min(SANITY_SUBSET_SIZE, len(eval_ds))))

# ============================================================
# 8. TRAINER CONFIG
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
    evaluation_strategy="epoch",
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
    args=training_args,
    callbacks=[SanityCheckCallback(eval_subset)]
)

trainer.train()

# ============================================================
# 9. TRAINING & VALIDATION LOSS ANALYSIS
# ============================================================

log_history = trainer.state.log_history
train_steps, train_losses, eval_steps, eval_losses = [], [], [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry.get("step"))
        train_losses.append(entry.get("loss"))
    if "eval_loss" in entry:
        eval_steps.append(entry.get("step"))
        eval_losses.append(entry.get("eval_loss"))

loss_dir = Path(OUTPUT_DIR)
pd.DataFrame({"step": train_steps, "train_loss": train_losses}).to_csv(loss_dir / "train_loss.csv", index=False)
pd.DataFrame({"step": eval_steps, "eval_loss": eval_losses}).to_csv(loss_dir / "eval_loss.csv", index=False)

plt.figure()
plt.plot(train_steps, train_losses, label="Training Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss vs Steps")
plt.grid(True)
plt.legend()
plt.savefig(loss_dir / "training_loss.png")
plt.close()

plt.figure()
plt.plot(eval_steps, eval_losses, label="Validation Loss")
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Validation Loss vs Steps")
plt.grid(True)
plt.legend()
plt.savefig(loss_dir / "validation_loss.png")
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
# 11. CHECKPOINT EVALUATION
# ============================================================

def calc_metrics(preds, refs):
    refs_clean = [r.replace("<end_of_turn>", "").strip() for r in refs]
    return sacrebleu.corpus_bleu(preds, [refs_clean]).score, sacrebleu.corpus_chrf(preds, [refs_clean]).score

checkpoints = sorted(Path(OUTPUT_DIR).glob("checkpoint-*"), key=lambda x: int(x.name.split("-")[-1]))
history = []

for ckpt in checkpoints:
    print(f"\n🔍 Evaluating {ckpt.name}")
    model_ckpt = AutoModelForCausalLM.from_pretrained(ckpt, device_map="auto")
    model_ckpt.eval()

    # Sanity check
    log_file = SANITY_LOG_DIR / f"{ckpt.name}_sanity.jsonl"
    run_sanity_subset(model_ckpt, tokenizer, model_ckpt.device, eval_subset, log_path=log_file)

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
        pred = tokenizer.decode(out[0][inputs.input_ids.shape[-1]:], skip_special_tokens=True).strip()
        src = prompt.split("\n")[-1].replace("<end_of_turn>", "").strip()

        if "HINDI DEVANAGARI" in prompt:
            eng_hin_preds.append(pred)
            eng_hin_refs.append(ref)
        else:
            hin_eng_preds.append(pred)
            hin_eng_refs.append(ref)

        records.append({
            "direction": "ENG_to_HIN" if "HINDI DEVANAGARI" in prompt else "HIN_to_ENG",
            "src": src, "ref": ref,
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
    with open(ckpt_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

df_hist = pd.DataFrame(history)
df_hist.to_csv(Path(OUTPUT_DIR) / "checkpoint_translation_metrics.csv", index=False)

plt.figure()
plt.plot(df_hist["bleu_eng_hin"], label="BLEU ENG→HIN")
plt.plot(df_hist["bleu_hin_eng"], label="BLEU HIN→ENG")
plt.plot(df_hist["chrf_eng_hin"], label="chrF2 ENG→HIN")
plt.plot(df_hist["chrf_hin_eng"], label="chrF2 HIN→ENG")
plt.legend()
plt.grid(True)
plt.savefig(Path(OUTPUT_DIR) / "bleu_chrf_vs_checkpoint.png")
plt.show() 
