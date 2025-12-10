# ======================================================
# 🚀 UNIVERSAL VS CODE VERSION (LoRA + Streaming + SFTTrainer) - FINAL
# ======================================================

import os, json, zipfile, warnings, gc, math
from pathlib import Path
import torch
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("./universal_output_sft")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 384
BATCH_SIZE = 1
GRAD_ACCUM = 4
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

    target_ids = tokenizer(f" {tgt}", truncation=True,
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
    data = {"train": [], "dev": []}
    print("\n📚 LOADING TRAINING DATA")
    dataset_name = "ai4bharat/Pralekha"
    splits = get_dataset_split_names(dataset_name, "train")
    count = 0
    skipped = 0

    for split in tqdm(splits, desc="Loading train"):
        if max_samples and count >= max_samples: break
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in ["eng","hin"] or tl not in ["eng","hin"]: continue
        ds = load_dataset(dataset_name, split=split, streaming=True, name="train")
        for row in ds:
            if max_samples and count >= max_samples: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t or len(s.split()) < 5 or len(t.split()) < 5:
                skipped += 1
                continue
            if s[:100] == t[:100]:
                skipped += 1
                continue
            eng, hin = (s, t) if sl == "eng" else (t, s)
            data["train"].append({**format_example(eng, hin, "eng_hin", tokenizer), "src": eng, "tgt": hin, "dirn": "eng_hin"})
            data["train"].append({**format_example(hin, eng, "hin_eng", tokenizer), "src": hin, "tgt": eng, "dirn": "hin_eng"})
            count += 1
    print(f"✅ Loaded {len(data['train'])} training examples (skipped {skipped})")

    print("\n📊 LOADING EVAL DATA")
    dev_splits = get_dataset_split_names(dataset_name, "dev")
    eval_count = 0
    eval_max = EVAL_SAMPLES or (max_samples // 5 if max_samples else 100)
    for split in tqdm(dev_splits, desc="Loading eval"):
        if eval_count >= eval_max: break
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

# ------------------------------ MODEL PREP (safe fp16)
def prepare_model():
    warnings.filterwarnings("ignore", message=".*label_names.*")
    print("\n🔧 Loading model...")
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    tok.padding_side = 'right'

    # T4-safe dtype: float16
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
        low_cpu_mem_usage=True
    )

    target_modules = list({n.split(".")[-1] for n, m in model.named_modules() if any(x in n.lower() for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"])})

    print(f"⚡ LoRA targets: {target_modules}")
    lora_cfg = LoraConfig(
        r=2,
        lora_alpha=4,
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
    return {"input_ids": torch.tensor(input_ids),
            "attention_mask": torch.tensor(attention_mask),
            "labels": torch.tensor(labels)}

# ------------------------------ SFT TRAINER
def train_model(model, tok, dataset, output_dir=str(OUTPUT_DIR), max_steps=100):
    sft_config = SFTConfig(
        output_dir=output_dir,
        overwrite_output_dir=True,
        num_train_epochs=1,
        max_steps=max_steps,
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=1,
        gradient_accumulation_steps=GRAD_ACCUM,
        logging_steps=10,
        save_steps=25,
        save_total_limit=2,
        learning_rate=2e-4,
        warmup_ratio=0.03,
        lr_scheduler_type="cosine",
        bf16=False,
        fp16=True,
        report_to="none",
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tok,
        args=sft_config,
        train_dataset=dataset,
        formatting_func=None,
        dataset_text_field=None,
        max_seq_length=MAX_SEQ_LEN,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tok.save_pretrained(output_dir)
    return model, tok, trainer

# ------------------------------ EVALUATION
def evaluate_model(model, tok, eval_data):
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    preds, refs, inputs = {"eng_hin": [], "hin_eng": []}, {"eng_hin": [], "hin_eng": []}, {"eng_hin": [], "hin_eng": []}

    for ex in tqdm(eval_data, desc="Evaluating"):
        src, tgt, dirn = ex["src"], ex["tgt"], ex["dirn"]
        prompt = f"""Task: Translate English to Hindi.\nIMPORTANT: Output ONLY Hindi translation.\n\nEnglish text:\n{src}\n\nHindi translation:""" if dirn=="eng_hin" else f"""Task: Translate Hindi to English.\nIMPORTANT: Output ONLY English translation.\n\nHindi text:\n{src}\n\nEnglish translation:"""
        enc = tok(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN//2).to(device)
        input_len = enc["input_ids"].shape[1]
        with torch.no_grad():
            out = model.generate(**enc, max_new_tokens=512, pad_token_id=tok.pad_token_id, eos_token_id=tok.eos_token_id, do_sample=False, num_beams=1)
        generated_ids = out[0, input_len:]
        pred_text = tok.decode(generated_ids, skip_special_tokens=True).strip()
        pred_text = "\n".join(line.strip() for line in pred_text.split("\n") if line.strip())
        preds[dirn].append(pred_text)
        refs[dirn].append(tgt.strip())
        inputs[dirn].append(src.strip())

    save_results(preds, refs, inputs)
    return calculate_metrics(preds, refs)

def save_results(preds, refs, inputs):
    print("\n💾 Saving results...")
    for dirn in ["eng_hin","hin_eng"]:
        if not preds[dirn]: continue
        jsonl_file = OUTPUT_DIR / f"{dirn}_pred_ref.jsonl"
        with open(jsonl_file, "w", encoding="utf-8") as f:
            for inp, pred, ref in zip(inputs[dirn], preds[dirn], refs[dirn]):
                f.write(json.dumps({"input_text": inp, "prediction": pred, "reference": ref}, ensure_ascii=False)+"\n")
        print(f"   ✅ Saved {dirn}_pred_ref.jsonl ({len(preds[dirn])} examples)")
    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for dirn in ["eng_hin","hin_eng"]:
            jsonl_file = OUTPUT_DIR / f"{dirn}_pred_ref.jsonl"
            if jsonl_file.exists(): zf.write(jsonl_file, jsonl_file.name)
    print(f"   ✅ Created submission.zip")

def calculate_metrics(preds, refs):
    bleu_scores, chrf_scores = {}, {}
    for dirn in preds:
        if not preds[dirn]: continue
        bleu_scores[dirn] = sacrebleu.corpus_bleu(preds[dirn],[refs[dirn]]).score
        chrf_scores[dirn] = sacrebleu.corpus_chrf(preds[dirn], [[r] for r in refs[dirn]]).score
    return bleu_scores, chrf_scores

def plot_training(trainer):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    if not steps: return
    plt.figure(figsize=(10,5))
    plt.plot(steps, losses, marker='o', linewidth=2, markersize=4)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss"); plt.grid(True, alpha=0.3); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png", dpi=150)
    print("📉 Training loss curve saved")

# ------------------------------ MAIN ENTRY
if __name__ == "__main__":
    print("="*60)
    print("🚀 TRANSLATION TRAINING (VS CODE + LoRA + Streaming + SFTTrainer)")
    print("="*60)
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    if torch.cuda.is_available(): torch.cuda.empty_cache(); gc.collect()

    # Prepare tokenizer & data
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    data = load_translation_data(tok, max_samples=TRAIN_SAMPLES)

    # Prepare model (PEFT/LoRA) and train using SFTTrainer
    model, tok = prepare_model()
    train_dataset = list(data["train"])  # SFTTrainer accepts list-like tokenized samples
    model, tok, trainer = train_model(model, tok, train_dataset, output_dir=str(OUTPUT_DIR), max_steps=50 if QUICK_TEST else 1000)

    # Evaluate and plot
    bleu, chrf = evaluate_model(model, tok, data["dev"])
    plot_training(trainer)

    # Final results
    print("\n📊 FINAL RESULTS")
    for dirn in ["eng_hin","hin_eng"]:
        if dirn in bleu:
            print(f"{dirn.upper()}: BLEU={bleu[dirn]:.2f}, chrF={chrf[dirn]:.2f}")
    print("\n✅ Training complete!")
    print(f"📁 All outputs saved to: {OUTPUT_DIR}")
    print(f"   - Model weights")
    print(f"   - eng_hin_pred_ref.jsonl")
    print(f"   - hin_eng_pred_ref.jsonl")
    print(f"   - submission.zip")
    print(f"   - training_loss.png")
