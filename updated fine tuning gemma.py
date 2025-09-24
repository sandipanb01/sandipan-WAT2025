# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Loss + Eval with chrF)
# ======================================================

import os
import json
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 1024
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"   # dataset has no dev → we’ll fallback automatically

LANGUAGE_PAIRS = ["eng_hin", "hin_eng"]

TRAIN_SAMPLES = 100
EVAL_SAMPLES = 10

# ------------------------------
# Manual prompt
# ------------------------------
def build_prompt_manual(src_text, src_lang, tgt_lang):
    return f"Translate this {src_lang} document to {tgt_lang}:\n{src_text}\nAssistant: "

# ------------------------------
# Dataset builder
# ------------------------------
def load_streaming_dataset(tokenizer, split="train", max_samples=100):
    examples = []
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")

        # Fallback: if dev requested but not available, use train instead
        actual_split = "train" if split == "dev" else split

        try:
            ds = load_dataset("ai4bharat/Pralekha", split=actual_split, data_dir=actual_split, streaming=False)
        except Exception as e:
            print(f"  [WARN] cannot load split={split} for pair {pair}: {e}")
            continue

        added = 0
        for i, row in enumerate(ds):
            # If we faked dev, skip the first 1000 examples to avoid overlap with training
            if split == "dev" and i < 1000:
                continue

            src_txt = row.get("src_txt") or row.get("src_text") or ""
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text") or ""
            if not src_txt or not tgt_txt:
                continue

            prompt_str = build_prompt_manual(src_txt, src, tgt)
            prompt_ids = tokenizer(prompt_str, add_special_tokens=False)["input_ids"]
            target_ids = tokenizer(tgt_txt, truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]

            input_ids = (prompt_ids + target_ids)[:MAX_SEQ_LEN]
            attention_mask = [1] * len(input_ids)

            prompt_len = min(len(prompt_ids), len(input_ids))
            labels = [-100] * prompt_len + input_ids[prompt_len:]
            labels = labels[:MAX_SEQ_LEN]

            examples.append({
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels,
                "src_txt": src_txt,
                "tgt_txt": tgt_txt,
                "src_lang": src,
                "tgt_lang": tgt
            })

            added += 1
            if max_samples and added >= max_samples:
                break
    return Dataset.from_list(examples)

# ------------------------------
# Training
# ------------------------------
def train_model():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float32
    print(f"[INFO] Using compute dtype: {compute_dtype}")

    print("[INFO] Loading base model (attn=eager)...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager",
    )

    lora_config = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Building datasets...")
    training_data = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=TRAIN_SAMPLES)
    eval_data = load_streaming_dataset(tokenizer, split=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=1,
        save_strategy="epoch",
        eval_strategy="no",  # manual eval below
        bf16=False,
        fp16=False,
        optim="paged_adamw_32bit",
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-pralekha",
        dataloader_pin_memory=False,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_config,
        train_dataset=training_data,
        tokenizer=tokenizer,
    )

    print("[INFO] Starting training...")
    trainer.train()

    # Save model + tokenizer
    trainer.model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    return model, tokenizer, eval_data

# ------------------------------
# Evaluation (chrF)
# ------------------------------
def evaluate_model(model, tokenizer, eval_data):
    print("[INFO] Starting evaluation...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()

    preds, refs = [], []
    total_loss = 0

    for ex in eval_data:
        input_ids = torch.tensor([ex["input_ids"]]).to(device)
        attention_mask = torch.tensor([ex["attention_mask"]]).to(device)
        labels = torch.tensor([ex["labels"]]).to(device)

        with torch.no_grad():
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            total_loss += outputs.loss.item()

            generated = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=64,
                do_sample=False
            )
            pred_text = tokenizer.decode(generated[0], skip_special_tokens=True)
            preds.append(pred_text)
            refs.append(ex["tgt_txt"])

    if len(eval_data) == 0:
        print("[WARN] No evaluation data available!")
        return

    avg_loss = total_loss / len(eval_data)
    chrf = sacrebleu.corpus_chrf(preds, [refs])

    print(f"[RESULT] Avg Eval Loss: {avg_loss:.4f}")
    print(f"[RESULT] chrF: {chrf.score:.2f}")
    print("[SAMPLE OUTPUTS]")
    for i in range(min(3, len(preds))):
        print(f"  SRC: {eval_data[i]['src_txt']}")
        print(f"  REF: {refs[i]}")
        print(f"  PRED: {preds[i]}")
        print("-" * 50)

# ------------------------------
# Run
# ------------------------------
if __name__ == "__main__":
    model, tokenizer, eval_data = train_model()
    evaluate_model(model, tokenizer, eval_data)
