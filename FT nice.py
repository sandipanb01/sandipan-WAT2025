# ======================================================
# 🚀 PRODUCTION VERSION (VS CODE READY)
# ======================================================

import os, json, zipfile, warnings, gc
from pathlib import Path
import torch
from datasets import load_dataset, get_dataset_split_names, Dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"

# Change output directory for local environment
OUTPUT_DIR = Path("./universal_output_fixed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 384
BATCH_SIZE = 1
GRAD_ACCUM = 4

# 🔥 QUICK TEST = True for debugging
QUICK_TEST = True

if QUICK_TEST:
    TRAIN_SAMPLES = 50
    EVAL_SAMPLES = 10
    print("🧪 QUICK TEST MODE ENABLED")
else:
    TRAIN_SAMPLES = None
    EVAL_SAMPLES = None
    print("🚀 FULL TRAINING MODE")

# ------------------------------ FORMATTER
def format_example(src, tgt, direction, tokenizer):
    """Format example with explicit task prompt."""

    if direction == "eng_hin":
        instruction = """Task: Translate English to Hindi.
IMPORTANT: Output ONLY Hindi translation. Do NOT continue the English text.

English text:"""
        prompt = f"{instruction}\n{src}\n\nHindi translation:"
    else:
        instruction = """Task: Translate Hindi to English.
IMPORTANT: Output ONLY English translation. Do NOT continue the Hindi text.

Hindi text:"""
        prompt = f"{instruction}\n{src}\n\nEnglish translation:"
    
    prompt_ids = tokenizer(prompt, truncation=True,
        max_length=MAX_SEQ_LEN//2, add_special_tokens=True)["input_ids"]

    target_ids = tokenizer(
        f" {tgt}", truncation=True,
        max_length=MAX_SEQ_LEN//2, add_special_tokens=False)["input_ids"]

    full_ids = prompt_ids + target_ids + [tokenizer.eos_token_id]
    labels = [-100]*len(prompt_ids) + target_ids + [tokenizer.eos_token_id]

    return {
        "input_ids": full_ids,
        "attention_mask": [1]*len(full_ids),
        "labels": labels
    }
# ------------------------------ LOAD DATA
def load_translation_data(tokenizer, max_samples=None):
    """Load and format data."""
    data = {"train": [], "dev": []}
    
    print("\n" + "="*80)
    print("📚 LOADING TRAINING DATA")
    print("="*80)
    
    dataset_name = "ai4bharat/Pralekha"
    splits = get_dataset_split_names(dataset_name, "train")
    
    count = 0
    skipped = 0
    
    for split in tqdm(splits, desc="Loading train"):
        if max_samples and count >= max_samples:
            break
            
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in ["eng","hin"] or tl not in ["eng","hin"]: continue
        
        ds = load_dataset(dataset_name, split=split, streaming=True, name="train")
        
        for row in ds:
            if max_samples and count >= max_samples: break
            
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            
            # Quality checks
            if not s or not t or len(s.split()) < 5 or len(t.split()) < 5:
                skipped += 1
                continue
            
            if s[:100] == t[:100]:
                skipped += 1
                continue
            
            eng, hin = (s, t) if sl == "eng" else (t, s)
            
            # Add both directions
            data["train"].append({
                **format_example(eng, hin, "eng_hin", tokenizer), 
                "src": eng, 
                "tgt": hin, 
                "dirn": "eng_hin"
            })
            data["train"].append({
                **format_example(hin, eng, "hin_eng", tokenizer), 
                "src": hin, 
                "tgt": eng, 
                "dirn": "hin_eng"
            })
            
            count += 1
    
    print(f"✅ Loaded {len(data['train'])} training examples (skipped {skipped})")
    
    # Load eval data
    print("\n📊 LOADING EVAL DATA")
    dev_splits = get_dataset_split_names(dataset_name, "dev")
    eval_count = 0
    
    # Use EVAL_SAMPLES if set, otherwise default based on train samples
    if EVAL_SAMPLES:
        eval_max = EVAL_SAMPLES
    elif max_samples:
        eval_max = max_samples // 5
    else:
        eval_max = 100
    
    print(f"   Target: {eval_max} sentence pairs (= {eval_max * 2} total examples with both directions)")
    
    for split in tqdm(dev_splits, desc="Loading eval"):
        if eval_count >= eval_max:
            break
            
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in ["eng","hin"] or tl not in ["eng","hin"]: continue
        
        ds = load_dataset(dataset_name, split=split, streaming=True, name="dev")
        
        for row in ds:
            if eval_count >= eval_max: break
            
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t or len(s.split()) < 5 or len(t.split()) < 5: continue
            if s[:100] == t[:100]: continue
            
            eng, hin = (s, t) if sl == "eng" else (t, s)
            
            data["dev"].append({"src": eng, "tgt": hin, "dirn": "eng_hin"})
            data["dev"].append({"src": hin, "tgt": eng, "dirn": "hin_eng"})
            
            eval_count += 1
    
    print(f"✅ Loaded {len(data['dev'])} eval examples")
    return data

# ------------------------------ MODEL
def prepare_model():
    warnings.filterwarnings("ignore", message=".*label_names.*")
    
    print("\n🔧 Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: 
        tok.pad_token = tok.eos_token
    tok.padding_side = 'right'
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        torch_dtype=torch.float32,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True,
    )
    
    # Find LoRA targets
    target_modules = []
    for n, m in model.named_modules():
        if any(x in n.lower() for x in ["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]):
            target_modules.append(n.split(".")[-1])
    target_modules = list(set(target_modules))
    
    print(f"⚡ LoRA targets: {target_modules}")
    
    lora_cfg = LoraConfig(
        r=32,
        lora_alpha=64, 
        target_modules=target_modules,
        lora_dropout=0.1,
        task_type="CAUSAL_LM", 
        bias="none"
    )
    
    model = get_peft_model(model, lora_cfg)
    model.print_trainable_parameters()
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    return model, tok
# ------------------------------ COLLATOR
def collate_fn(batch):
    max_len = max(len(x["input_ids"]) for x in batch)
    
    input_ids, attention_mask, labels = [], [], []
    
    for item in batch:
        pad_len = max_len - len(item["input_ids"])
        input_ids.append(item["input_ids"] + [0] * pad_len)
        attention_mask.append(item["attention_mask"] + [0] * pad_len)
        labels.append(item["labels"] + [-100] * pad_len)
    
    return {
        "input_ids": torch.tensor(input_ids),
        "attention_mask": torch.tensor(attention_mask),
        "labels": torch.tensor(labels)
    }

# ------------------------------ TRAINING
def train_model(data, tokenizer):
    model, tok = prepare_model()
    
    train_dataset = Dataset.from_list(data["train"])
    
    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=5 if QUICK_TEST else 3,
        logging_steps=10,
        save_strategy="epoch",
        save_total_limit=1,
        report_to="none",
        warmup_ratio=0.1,
        fp16=False,
        bf16=False,
        remove_unused_columns=False,
        label_names=["labels"],
        max_grad_norm=1.0,
    )
    
    print("\n🚀 Starting training...")
    print(f"   Effective batch size: {BATCH_SIZE * GRAD_ACCUM}")
    print(f"   Epochs: {args.num_train_epochs}")
    
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collate_fn
    )
    
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    trainer.train()
    
    print("\n💾 Saving model...")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION (CLEAN)
def evaluate_model(model, tok, eval_data):
    """Evaluate and save results to JSONL files."""
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    preds = {"eng_hin": [], "hin_eng": []}
    refs = {"eng_hin": [], "hin_eng": []}
    inputs = {"eng_hin": [], "hin_eng": []}
    
    print("\n" + "="*80)
    print("📊 EVALUATION")
    print("="*80)
    
    for ex in tqdm(eval_data, desc="Evaluating"):
        src = ex["src"]
        tgt = ex["tgt"]
        dirn = ex["dirn"]
        
        # Create prompt
        if dirn == "eng_hin":
            prompt = f"""Task: Translate English to Hindi.
IMPORTANT: Output ONLY Hindi translation. Do NOT continue the English text.

English text:
{src}

Hindi translation:"""
        else:
            prompt = f"""Task: Translate Hindi to English.
IMPORTANT: Output ONLY English translation. Do NOT continue the Hindi text.

Hindi text:
{src}

English translation:"""
        
        # Generate
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN//2).to(device)
        input_len = enc["input_ids"].shape[1]
        
        with torch.no_grad():
            out = model.generate(
                **enc, 
                max_new_tokens=512,  # ← INCREASED from 256 for longer docs
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id, 
                do_sample=False,
                num_beams=1
            )
        
        # Extract prediction
        generated_ids = out[0, input_len:]
        pred_text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        
        # Don't split on newlines - keep full translation!
        # Only clean up excessive newlines
        pred_text = "\n".join(line.strip() for line in pred_text.split("\n") if line.strip())
        
        preds[dirn].append(pred_text)
        refs[dirn].append(tgt.strip())
        inputs[dirn].append(src.strip())
    
    # Save results
    save_results(preds, refs, inputs)
    
    # Calculate metrics
    bleu_scores, chrf_scores = calculate_metrics(preds, refs)
    return bleu_scores, chrf_scores

def save_results(preds, refs, inputs):
    """Save predictions to JSONL files."""
    print("\n💾 Saving results...")
    
    # Save individual JSONL files
    for direction in ["eng_hin", "hin_eng"]:
        if not preds[direction]:
            continue
            
        jsonl_file = OUTPUT_DIR / f"{direction}_pred_ref.jsonl"
        
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for inp, pred, ref in zip(inputs[direction], preds[direction], refs[direction]):
                f.write(json.dumps({
                    "input_text": inp, 
                    "prediction": pred, 
                    "reference": ref
                }, ensure_ascii=False) + "\n")
        
        print(f"   ✅ Saved {direction}_pred_ref.jsonl ({len(preds[direction])} examples)")
    
    # Create zip file
    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for direction in ["eng_hin", "hin_eng"]:
            jsonl_file = OUTPUT_DIR / f"{direction}_pred_ref.jsonl"
            if jsonl_file.exists():
                zf.write(jsonl_file, jsonl_file.name)
    
    print(f"   ✅ Created submission.zip")

def calculate_metrics(preds, refs):
    """Calculate BLEU and chrF scores."""
    bleu_scores, chrf_scores = {}, {}
    
    for direction in preds:
        if not preds[direction]: 
            continue
        
        bleu_scores[direction] = sacrebleu.corpus_bleu(
            preds[direction], 
            [refs[direction]]
        ).score
        
        chrf_scores[direction] = sacrebleu.corpus_chrf(
            preds[direction], 
            [[r] for r in refs[direction]]
        ).score
    
    return bleu_scores, chrf_scores

def plot_training(trainer):
    """Plot training loss curve."""
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    
    if not steps: 
        return
    
    plt.figure(figsize=(10, 5))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Loss", fontsize=12)
    plt.title("Training Loss", fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    print("📉 Training loss curve saved")
# ------------------------------ MAIN ENTRY
if __name__ == "__main__":
    print("="*60)
    print("🚀 TRANSLATION TRAINING (VS CODE VERSION)")
    print("="*60)

    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    data = load_translation_data(tok, max_samples=TRAIN_SAMPLES)

    model, tok, trainer = train_model(data,tok)

    bleu, chrf = evaluate_model(model, tok, data["dev"])

    # Plot
    plot_training(trainer)

    # Print results
    print("\n" + "="*80)
    print("📊 FINAL RESULTS")
    print("="*80)
    for direction in ["eng_hin", "hin_eng"]:
        if direction in bleu:
            print(f"{direction.upper()}: BLEU={bleu[direction]:.2f}, chrF={chrf[direction]:.2f}")
    print("="*80)
    
    print("\n✅ Training complete!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"   - Model weights")
    print(f"   - eng_hin_pred_ref.jsonl")
    print(f"   - hin_eng_pred_ref.jsonl")
    print(f"   - submission.zip")
    print(f"   - training_loss.png")
