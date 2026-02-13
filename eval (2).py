import os
import json
import torch
import gc
import unicodedata
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import sacrebleu
from accelerate import Accelerator

# Force PyTorch to be more efficient with memory fragmentation
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# ============================================================
# 1. CONFIG
# ============================================================
accelerator = Accelerator()
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
EVAL_DIR   = OUTPUT_DIR / "checkpoint_eval"
PRED_DIR   = EVAL_DIR / "predictions"

if accelerator.is_main_process:
    for d in [EVAL_DIR, PRED_DIR]:
        d.mkdir(parents=True, exist_ok=True)

MAX_TGT_LEN = 1024
# Reduced Batch Size to 4 (Safer for 4800 seq len on A6000)
# Total effective batch size will be 8 (4 per GPU x 2 GPUs)
BATCH_SIZE = 16 

# ============================================================
# 2. TOKENIZER & DATA (Same Logic + Length Protection)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.padding_side = "left" 
tokenizer.pad_token = tokenizer.eos_token

raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
split = raw.train_test_split(test_size=0.1, seed=42)
val_set_raw = split["test"]

# IMPORTANT: Filter out "Memory Bombs" (sentences longer than training limits)
def length_filter(example):
    return len(example["src_txt"]) < 5000 and len(example["tgt_txt"]) < 5000

val_set = val_set_raw.filter(length_filter)

def devanagari_ratio(text):
    chars = [c for c in text if c.isalpha()]
    if not chars: return 0.0
    return sum("DEVANAGARI" in unicodedata.name(c, "") for c in chars) / len(chars)

# ============================================================
# 3. LOAD BASE MODEL ONCE (Speed Optimization)
# ============================================================
accelerator.print("🚀 Loading Base Model (Shared across all checkpoints)...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID, 
    torch_dtype=torch.bfloat16, 
    attn_implementation="flash_attention_2"
).to(accelerator.device)

# ============================================================
# 4. BATCHED EVALUATION FUNCTION
# ============================================================
def run_batched_eval(model, dataset, ckpt_name):
    ckpt_pred_dir = PRED_DIR / ckpt_name
    if accelerator.is_main_process:
        ckpt_pred_dir.mkdir(exist_ok=True)
    
    with accelerator.split_between_processes(list(range(len(dataset)))) as indices:
        local_results = {"E2H": [], "H2E": []}
        local_lid = []

        pbar = tqdm(range(0, len(indices), BATCH_SIZE), disable=not accelerator.is_main_process)
        for i in pbar:
            batch_indices = indices[i : i + BATCH_SIZE]
            batch = [dataset[int(idx)] for idx in batch_indices]
            
            for mode in ["E2H", "H2E"]:
                instr = "Translate to HINDI DEVANAGARI:" if mode == "E2H" else "Translate to ENGLISH:"
                sources = [b["src_txt"] if mode == "E2H" else b["tgt_txt"] for b in batch]
                refs = [b["tgt_txt"] if mode == "E2H" else b["src_txt"] for b in batch]
                
                prompts = [f"<start_of_turn>user\n{instr}\n{s}<end_of_turn>\n<start_of_turn>model\n" for s in sources]
                inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(accelerator.device)
                
                with torch.inference_mode():
                    outputs = model.generate(
                        **inputs,
                        max_new_tokens=MAX_TGT_LEN,
                        do_sample=False,
                        use_cache=True
                    )
                
                preds = tokenizer.batch_decode(outputs[:, inputs.input_ids.shape[-1]:], skip_special_tokens=True)
                
                for s, r, p in zip(sources, refs, preds):
                    p_clean = p.strip()
                    local_results[mode].append({"src": s, "ref": r, "pred": p_clean})
                    ratio = devanagari_ratio(p_clean)
                    local_lid.append(ratio > 0.6 if mode == "E2H" else ratio < 0.4)

    # Gather data from both GPUs
    gathered_e2h = accelerator.gather_for_metrics(local_results["E2H"])
    gathered_h2e = accelerator.gather_for_metrics(local_results["H2E"])
    gathered_lid = accelerator.gather_for_metrics(local_lid)

    if accelerator.is_main_process:
        # Save JSONL
        for mode, data in [("E2H", gathered_e2h), ("H2E", gathered_h2e)]:
            with open(ckpt_pred_dir / f"{mode}.jsonl", "w", encoding="utf-8") as f:
                for item in data: f.write(json.dumps(item, ensure_ascii=False) + "\n")
        
        # Stats Calculation
        return {
            "ENG→HIN BLEU": sacrebleu.corpus_bleu([x["pred"] for x in gathered_e2h], [[x["ref"] for x in gathered_e2h]]).score,
            "ENG→HIN chrF2": sacrebleu.corpus_chrf([x["pred"] for x in gathered_e2h], [[x["ref"] for x in gathered_e2h]], beta=2).score,
            "HIN→ENG BLEU": sacrebleu.corpus_bleu([x["pred"] for x in gathered_h2e], [[x["ref"] for x in gathered_h2e]]).score,
            "HIN→ENG chrF2": sacrebleu.corpus_chrf([x["pred"] for x in gathered_h2e], [[x["ref"] for x in gathered_h2e]], beta=2).score,
            "Script Acc (%)": np.mean(gathered_lid) * 100
        }
    return None

# ============================================================
# 5. MAIN CHECKPOINT LOOP
# ============================================================
ckpts = sorted([c for c in os.listdir(CKPT_DIR) if c.startswith("checkpoint-")], key=lambda x: int(x.split("-")[-1]))
all_stats = {}

for ckpt in ckpts:
    accelerator.print(f"\n🔍 Evaluating {ckpt}")
    
    # Load LoRA on top of existing base_model
    model = PeftModel.from_pretrained(base_model, CKPT_DIR / ckpt).eval()

    all_stats[ckpt] = run_batched_eval(model, val_set, ckpt)
    
    # CRITICAL: Thorough cleanup after each checkpoint
    del model
    gc.collect()
    torch.cuda.empty_cache()

# ============================================================
# 6. FINAL SAVE & PLOT
# ============================================================
if accelerator.is_main_process:
    df = pd.DataFrame.from_dict(all_stats, orient="index")
    df.to_csv(EVAL_DIR / "checkpoint_metrics.csv")
    print("\n✅ EVALUATION COMPLETE")