import os
import torch
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset
from difflib import SequenceMatcher
from transformers import AutoModelForCausalLM, AutoTokenizer, set_seed, TrainerCallback
from peft import LoraConfig, PeftModel
from trl import SFTTrainer, SFTConfig
import sacrebleu

# --- Dependencies Guard ---
def install_and_import(package):
    import subprocess, sys
    try: __import__(package)
    except ImportError:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])

install_and_import("langdetect")
from langdetect import detect, DetectorFactory
DetectorFactory.seed = 42
set_seed(42)

# ============================================================
# 1. CONFIGURATION & STRICT FILTERING (Anti-Cheating)
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "./gemma3-strict-bidirectional"

TRAIN_CONFIG = "train"
EVAL_CONFIG  = "test"

# Set to None for full data
MAX_TRAIN_SAMPLES = None
MAX_EVAL_SAMPLES = None
MAX_TEST_SAMPLES = None  # NEW: separate test set

# NEW: Validation split ratio (10% of training data)
VAL_RATIO = 0.1

MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_TOTAL_LEN = MAX_SRC_LEN + MAX_TGT_LEN

# NEW: Checkpointing configuration
CHECKPOINT_STEPS = 500  # Save every 500 steps
SANITY_SUBSET_SIZE = 5  # Quick generation test samples

# Create output directories
os.makedirs(OUTPUT_DIR, exist_ok=True)
SANITY_LOG_DIR = Path(OUTPUT_DIR) / "sanity_check_logs"
SANITY_LOG_DIR.mkdir(exist_ok=True)

# ----------------------------
# Load tokenizer EARLY
# ----------------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 2. UTF-8 PRE-FILTER (CRITICAL FIX)
# ============================================================
# Some rows in ai4bharat/Pralekha contain bytes that are not valid UTF-8.
# PyArrow's .to_pylist() / .as_py() raises UnicodeDecodeError on those rows
# *before* any HF .filter() callback can see them.

def get_valid_indices(dataset, text_columns=("src_txt", "tgt_txt")):
    """Return list of row indices where all *text_columns* decode as valid UTF-8."""
    pa_table = dataset.data  # underlying pyarrow.Table
    valid = []
    for row_idx in range(len(pa_table)):
        ok = True
        for col in text_columns:
            scalar = pa_table.column(col)[row_idx]
            try:
                scalar.as_py()  # triggers the UTF-8 decode
            except (UnicodeDecodeError, Exception):
                ok = False
                break
        if ok:
            valid.append(row_idx)
    return valid

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
    src_len = len(tokenizer(example["src_txt"], add_special_tokens=True, truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], add_special_tokens=True, truncation=False)["input_ids"])
    return (src_len <= MAX_SRC_LEN) and (tgt_len <= MAX_TGT_LEN)

# ============================================================
# 4. LOAD + SPLIT DATA (TRAIN / VAL / TEST)
# ============================================================

print("Loading training data...")
raw_dataset = load_dataset(DATASET_NAME, TRAIN_CONFIG, split="eng_hin")

# --- Drop rows with invalid UTF-8 bytes BEFORE any .filter() ---
print("Scanning for invalid UTF-8 rows in training data...")
valid_idx = get_valid_indices(raw_dataset)
print(f"  Kept {len(valid_idx)} / {len(raw_dataset)} rows after UTF-8 pre-filter.")
raw_dataset = raw_dataset.select(valid_idx)

# Apply strict filtering
print("Applying strict similarity filter...")
filtered_dataset = raw_dataset.filter(strict_filter, batched=False, load_from_cache_file=False)
print(f"  Kept {len(filtered_dataset)} rows after similarity filter.")

print("Applying length filter...")
filtered_dataset = filtered_dataset.filter(length_filter, batched=False, load_from_cache_file=False)
print(f"  Kept {len(filtered_dataset)} rows after length filter.")

# Apply MAX_TRAIN_SAMPLES if set
if MAX_TRAIN_SAMPLES is not None:
    filtered_dataset = filtered_dataset.select(range(min(MAX_TRAIN_SAMPLES, len(filtered_dataset))))
    print(f"  Limited to {len(filtered_dataset)} training samples.")

# NEW: Split into train and validation
print(f"Splitting into train ({100*(1-VAL_RATIO):.0f}%) and validation ({100*VAL_RATIO:.0f}%)...")
split = filtered_dataset.train_test_split(test_size=VAL_RATIO, seed=42, shuffle=True)
train_set = split["train"]
val_set = split["test"]
print(f"  Train: {len(train_set)} samples")
print(f"  Validation: {len(val_set)} samples")

# ----------------------------
# SEPARATE TEST DATA
# ----------------------------
print("\nLoading test data...")
eval_dataset = load_dataset(DATASET_NAME, EVAL_CONFIG, split="eng_hin")

# Apply UTF-8 filter to test set
print("Scanning for invalid UTF-8 rows in test data...")
test_valid_idx = get_valid_indices(eval_dataset)
print(f"  Kept {len(test_valid_idx)} / {len(eval_dataset)} rows after UTF-8 pre-filter.")
eval_dataset = eval_dataset.select(test_valid_idx)

# Apply length filter to test set
eval_dataset = eval_dataset.filter(length_filter, batched=False, load_from_cache_file=False)
print(f"  Kept {len(eval_dataset)} rows after length filter.")

# Apply MAX_TEST_SAMPLES if set
if MAX_TEST_SAMPLES is not None:
    eval_dataset = eval_dataset.select(range(min(MAX_TEST_SAMPLES, len(eval_dataset))))
    print(f"  Limited to {len(eval_dataset)} test samples.")

test_set = eval_dataset
print(f"  Test: {len(test_set)} samples")

# ============================================================
# 5. MODEL & LoRA CONFIG
# ============================================================
print("\nLoading model...")
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
# 6. BIDIRECTIONAL FORMATTING (PROMPT/COMPLETION SPLIT)
# ============================================================
def formatting_prompts_func(example):
    prompts = []
    completions = []
    for i in range(len(example["src_txt"])):
        if i % 2 == 0:
            instr, src, tgt = "Translate to HINDI DEVANAGARI:", example["src_txt"][i], example["tgt_txt"][i]
        else:
            instr, src, tgt = "Translate to ENGLISH:", example["tgt_txt"][i], example["src_txt"][i]

        prompts.append(f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n")
        completions.append(f"{tgt}<end_of_turn>")
        
    return {"prompt": prompts, "completion": completions}

print("\nFormatting datasets...")
train_dataset = train_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=train_set.column_names
)

val_dataset = val_set.map(
    formatting_prompts_func,
    batched=True,
    remove_columns=val_set.column_names
)

# ============================================================
# 7. SANITY CHECK CALLBACK
# ============================================================
# This runs quick generation tests during training to catch issues early

def run_sanity_subset(model, tokenizer, device, eval_subset, log_path=None):
    """Generate translations on a small subset and log them."""
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
        
        # Show first line of prompt (the instruction)
        instruction = prompt.split("\n")[1] if "\n" in prompt else prompt[:50]
        print(f"Instruction: {instruction}")
        print(f"Predicted: {pred[:100]}...")  # First 100 chars
        print(f"Reference: {ref[:100]}...\n")

        sanity_records.append({
            "prompt": instruction,
            "predicted": pred,
            "reference": ref
        })

    if log_path:
        with open(log_path, "w", encoding="utf-8") as f:
            for rec in sanity_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    print("=" * 50)

class SanityCheckCallback(TrainerCallback):
    """Callback to run sanity checks during training."""
    def __init__(self, eval_subset):
        self.eval_subset = eval_subset

    def on_evaluate(self, args, state, control, **kwargs):
        """Run sanity check after each evaluation."""
        log_file = SANITY_LOG_DIR / f"sanity_step_{state.global_step}.jsonl"
        run_sanity_subset(
            kwargs["model"],
            kwargs["tokenizer"],
            kwargs["model"].device,
            self.eval_subset,
            log_path=log_file
        )

# Create sanity check subset from validation set
eval_subset = [val_dataset[i] for i in range(min(SANITY_SUBSET_SIZE, len(val_dataset)))]

# ============================================================
# 8. TRAINING WITH CHECKPOINTS
# ============================================================
print("\nStarting training...")

training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_length=MAX_TOTAL_LEN,
    learning_rate=2e-4,
    num_train_epochs=2,
    
    # NEW: Enable checkpointing and evaluation
    logging_steps=10,
    eval_steps=CHECKPOINT_STEPS,
    save_steps=CHECKPOINT_STEPS,
    do_eval=True,
    eval_strategy="steps",
    save_strategy="steps",
    save_total_limit=100,  # Keep only last 10 checkpoints
    
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    completion_only_loss=True,
    gradient_checkpointing=True,
    weight_decay=0.01,
    report_to="none"
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,  # NEW: validation dataset
    peft_config=peft_config,
    args=training_args,
    callbacks=[SanityCheckCallback(eval_subset)]
)

trainer.train()

# ============================================================
# 9. TRAINING & VALIDATION LOSS ANALYSIS
# ============================================================
print("\nAnalyzing training history...")

log_history = trainer.state.log_history
train_steps, train_losses = [], []
eval_steps, eval_losses = [], []

for entry in log_history:
    if "loss" in entry and "eval_loss" not in entry:
        train_steps.append(entry.get("step"))
        train_losses.append(entry.get("loss"))
    if "eval_loss" in entry:
        eval_steps.append(entry.get("step"))
        eval_losses.append(entry.get("eval_loss"))

# Save loss data to CSV
loss_dir = Path(OUTPUT_DIR)
pd.DataFrame({"step": train_steps, "train_loss": train_losses}).to_csv(
    loss_dir / "train_loss.csv", index=False
)
pd.DataFrame({"step": eval_steps, "eval_loss": eval_losses}).to_csv(
    loss_dir / "eval_loss.csv", index=False
)

# Plot training loss
plt.figure(figsize=(10, 6))
plt.plot(train_steps, train_losses, label="Training Loss", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training Loss vs Steps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(loss_dir / "training_loss.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot validation loss
plt.figure(figsize=(10, 6))
plt.plot(eval_steps, eval_losses, label="Validation Loss", linewidth=2, color='orange')
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Validation Loss vs Steps")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(loss_dir / "validation_loss.png", dpi=150, bbox_inches='tight')
plt.close()

# Plot both together
plt.figure(figsize=(12, 6))
plt.plot(train_steps, train_losses, label="Training Loss", linewidth=2)
plt.plot(eval_steps, eval_losses, label="Validation Loss", linewidth=2)
plt.xlabel("Training Step")
plt.ylabel("Loss")
plt.title("Training & Validation Loss")
plt.grid(True, alpha=0.3)
plt.legend()
plt.savefig(loss_dir / "combined_loss.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Loss plots saved to {loss_dir}")

# ============================================================
# 10. SAVE FINAL MODEL
# ============================================================
print("\nSaving final model...")
final_dir = Path(OUTPUT_DIR) / "final_model"
final_dir.mkdir(exist_ok=True)

final_model = trainer.model.merge_and_unload()
final_model.save_pretrained(final_dir)
tokenizer.save_pretrained(final_dir)
print(f"  Final model saved to {final_dir}")

# ============================================================
# 11. CHECKPOINT EVALUATION (BLEU + chrF for ALL checkpoints)
# ============================================================
print("\n" + "="*60)
print("EVALUATING ALL CHECKPOINTS ON TEST SET")
print("="*60)

def calc_metrics(preds, refs):
    """Calculate BLEU and chrF scores."""
    refs_clean = [r.replace("<end_of_turn>", "").strip() for r in refs]
    bleu = sacrebleu.corpus_bleu(preds, [refs_clean]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs_clean]).score
    return round(bleu, 2), round(chrf, 2)

# Find all checkpoints
checkpoints = sorted(
    Path(OUTPUT_DIR).glob("checkpoint-*"),
    key=lambda x: int(x.name.split("-")[-1])
)
print(f"Found {len(checkpoints)} checkpoints to evaluate.")

# Also evaluate the final merged model
checkpoints.append(final_dir)

checkpoint_history = []

# Create a smaller sanity subset for checkpoint evaluation
sanity_eval_subset = [val_dataset[i] for i in range(min(SANITY_SUBSET_SIZE, len(val_dataset)))]

for ckpt in checkpoints:
    print(f"\n{'='*60}")
    print(f"Evaluating: {ckpt.name}")
    print(f"{'='*60}")
    
    # Load checkpoint
    model_ckpt = AutoModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model_ckpt.eval()
    
    # --- SANITY CHECK LOG ---
    log_file = SANITY_LOG_DIR / f"{ckpt.name}_sanity.jsonl"
    run_sanity_subset(model_ckpt, tokenizer, model_ckpt.device, sanity_eval_subset, log_path=log_file)
    
    # Evaluate on test set
    eng_hin_preds, eng_hin_refs = [], []
    hin_eng_preds, hin_eng_refs = [], []
    records = []
    
    for sample in tqdm(test_set, desc=f"Testing {ckpt.name}", leave=False):
        pairs = [
            ("ENG_to_HIN", "Translate to HINDI DEVANAGARI:", sample["src_txt"], sample["tgt_txt"]),
            ("HIN_to_ENG", "Translate to ENGLISH:", sample["tgt_txt"], sample["src_txt"]),
        ]
        
        for mode, instr, src, ref in pairs:
            prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"
            
            inputs = tokenizer(prompt, return_tensors="pt").to(model_ckpt.device)
            with torch.no_grad():
                output = model_ckpt.generate(
                    **inputs,
                    max_new_tokens=MAX_TGT_LEN,
                    temperature=0.1,
                    do_sample=False,
                    repetition_penalty=1.1
                )
            
            pred_tokens = output[0][inputs.input_ids.shape[-1]:]
            pred = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()
            
            if mode == "ENG_to_HIN":
                eng_hin_preds.append(pred)
                eng_hin_refs.append(ref)
            else:
                hin_eng_preds.append(pred)
                hin_eng_refs.append(ref)
            
            records.append({
                "direction": mode,
                "src": src,
                "ref": ref,
                "pred": pred
            })
    
    # Calculate metrics
    bleu_eh, chrf_eh = calc_metrics(eng_hin_preds, eng_hin_refs)
    bleu_he, chrf_he = calc_metrics(hin_eng_preds, hin_eng_refs)
    
    checkpoint_history.append({
        "checkpoint": ckpt.name,
        "bleu_eng_hin": bleu_eh,
        "chrf_eng_hin": chrf_eh,
        "bleu_hin_eng": bleu_he,
        "chrf_hin_eng": chrf_he
    })
    
    print(f"\n  ENG → HIN | BLEU: {bleu_eh} | chrF: {chrf_eh}")
    print(f"  HIN → ENG | BLEU: {bleu_he} | chrF: {chrf_he}")
    
    # Save JSONL for this checkpoint
    ckpt_jsonl_dir = Path(OUTPUT_DIR) / "checkpoint_jsonl"
    ckpt_jsonl_dir.mkdir(exist_ok=True)
    ckpt_jsonl = ckpt_jsonl_dir / f"{ckpt.name}_translations.jsonl"
    
    with open(ckpt_jsonl, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    
    print(f"  Saved to: {ckpt_jsonl}")

# ============================================================
# 12. CHECKPOINT METRICS VISUALIZATION
# ============================================================
print("\nCreating checkpoint performance plots...")

df_hist = pd.DataFrame(checkpoint_history)
df_hist.to_csv(Path(OUTPUT_DIR) / "checkpoint_translation_metrics.csv", index=False)

# Plot BLEU and chrF across checkpoints
plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(df_hist.index, df_hist["bleu_eng_hin"], marker='o', label="BLEU ENG→HIN", linewidth=2)
plt.plot(df_hist.index, df_hist["bleu_hin_eng"], marker='s', label="BLEU HIN→ENG", linewidth=2)
plt.xlabel("Checkpoint Index")
plt.ylabel("BLEU Score")
plt.title("BLEU Scores Across Checkpoints")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df_hist.index, df_hist["checkpoint"], rotation=45, ha='right')

plt.subplot(2, 1, 2)
plt.plot(df_hist.index, df_hist["chrf_eng_hin"], marker='o', label="chrF ENG→HIN", linewidth=2)
plt.plot(df_hist.index, df_hist["chrf_hin_eng"], marker='s', label="chrF HIN→ENG", linewidth=2)
plt.xlabel("Checkpoint Index")
plt.ylabel("chrF Score")
plt.title("chrF Scores Across Checkpoints")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df_hist.index, df_hist["checkpoint"], rotation=45, ha='right')

plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "checkpoint_metrics.png", dpi=150, bbox_inches='tight')
plt.close()

# Combined plot (all 4 metrics)
plt.figure(figsize=(12, 6))
plt.plot(df_hist.index, df_hist["bleu_eng_hin"], marker='o', label="BLEU ENG→HIN", linewidth=2)
plt.plot(df_hist.index, df_hist["bleu_hin_eng"], marker='s', label="BLEU HIN→ENG", linewidth=2)
plt.plot(df_hist.index, df_hist["chrf_eng_hin"], marker='^', label="chrF ENG→HIN", linewidth=2)
plt.plot(df_hist.index, df_hist["chrf_hin_eng"], marker='v', label="chrF HIN→ENG", linewidth=2)
plt.xlabel("Checkpoint")
plt.ylabel("Score")
plt.title("Translation Quality Metrics Across Training")
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(df_hist.index, df_hist["checkpoint"], rotation=45, ha='right')
plt.tight_layout()
plt.savefig(Path(OUTPUT_DIR) / "all_metrics_combined.png", dpi=150, bbox_inches='tight')
plt.close()

print(f"  Plots saved to {OUTPUT_DIR}")

# ============================================================
# 13. FINAL EVALUATION SUMMARY
# ============================================================
print("\n" + "="*60)
print("FINAL SUMMARY")
print("="*60)

# Get final model metrics (last row)
final_metrics = checkpoint_history[-1]
print(f"\nFinal Model: {final_metrics['checkpoint']}")
print(f"  ENG → HIN | BLEU: {final_metrics['bleu_eng_hin']} | chrF: {final_metrics['chrf_eng_hin']}")
print(f"  HIN → ENG | BLEU: {final_metrics['bleu_hin_eng']} | chrF: {final_metrics['chrf_hin_eng']}")

# Find best checkpoint for each metric
best_bleu_eh_idx = df_hist["bleu_eng_hin"].idxmax()
best_bleu_he_idx = df_hist["bleu_hin_eng"].idxmax()

print(f"\nBest Checkpoints:")
print(f"  Best ENG→HIN BLEU: {df_hist.loc[best_bleu_eh_idx, 'checkpoint']} "
      f"(BLEU: {df_hist.loc[best_bleu_eh_idx, 'bleu_eng_hin']})")
print(f"  Best HIN→ENG BLEU: {df_hist.loc[best_bleu_he_idx, 'checkpoint']} "
      f"(BLEU: {df_hist.loc[best_bleu_he_idx, 'bleu_hin_eng']})")

print("\n" + "="*60)
print("All results saved to:", OUTPUT_DIR)
print("="*60)
print("\nGenerated files:")
print("  - train_loss.csv, eval_loss.csv")
print("  - training_loss.png, validation_loss.png, combined_loss.png")
print("  - checkpoint_translation_metrics.csv")
print("  - checkpoint_metrics.png, all_metrics_combined.png")
print("  - checkpoint_jsonl/*.jsonl (detailed predictions per checkpoint)")
print("  - sanity_check_logs/*.jsonl (generation samples during training)")
print("="*60)
