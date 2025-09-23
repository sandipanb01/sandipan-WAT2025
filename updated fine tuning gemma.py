# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Memory-Safe Version)
# ======================================================

import os
from pathlib import Path
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 1024       # Reduced for memory
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
EVAL_SAMPLES = 200

LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal", "eng_mar",
    "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "guj_eng", "hin_eng", "kan_eng", "mal_eng", "mar_eng",
    "ori_eng", "pan_eng", "tam_eng", "tel_eng", "urd_eng",
]

# ------------------------------
# Build prompt utility
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, tokenizer):
    messages = [
        {"role": "user", "content": f"Translate this {src_lang} document to {tgt_lang}:\n{src_text}\n"},
        {"role": "assistant", "content": ""}
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ------------------------------
# Streaming dataset loader
# ------------------------------
def load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=None):
    all_instances = []

    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        print(f"[INFO] Loading {src} → {tgt} ({split})...")
        ds = load_dataset("ai4bharat/Pralekha", split=split, data_dir=split, streaming=False)

        processed = 0
        for row in ds:
            src_txt = row.get("src_txt") or row.get("src_text", "")
            tgt_txt = row.get("tgt_txt") or row.get("tgt_text", "")
            if not src_txt or not tgt_txt:
                continue

            # Encode prompt and labels
            prompt = build_prompt(src_txt, src, tgt, tokenizer)
            encoded = tokenizer(prompt, max_length=MAX_SEQ_LEN, truncation=True)
            labels = [-100] * len(encoded["input_ids"])

            # Only compute loss on assistant tokens
            assistant_tokens = tokenizer(tgt_txt, max_length=MAX_SEQ_LEN, truncation=True)["input_ids"]
            start_idx = len(encoded["input_ids"]) - len(assistant_tokens)
            start_idx = max(0, start_idx)
            for i in range(start_idx, len(encoded["input_ids"])):
                labels[i] = encoded["input_ids"][i]

            instance = {
                "input_ids": encoded["input_ids"],
                "attention_mask": encoded["attention_mask"],
                "labels": labels
            }
            all_instances.append(instance)
            processed += 1
            if max_samples and processed >= max_samples:
                break

        print(f"  Added {processed} samples from {pair}")

    return Dataset.from_list(all_instances)

# ------------------------------
# Evaluation
# ------------------------------
def evaluate_model(model_path, tokenizer, lang_pairs, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES):
    print("[INFO] Starting evaluation...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager"  # memory-efficient
    )

    results = {}
    for pair in lang_pairs:
        src, tgt = pair.split("_")
        print(f"[EVAL] {src} → {tgt}")
        try:
            ds = load_dataset("ai4bharat/Pralekha", split=pair, data_dir=subset)
        except Exception as e:
            print(f"  [WARN] Skipping {pair}: {e}")
            continue

        ds = ds.select(range(min(max_samples, len(ds))))
        preds, refs = [], []

        for row in ds:
            src_text = row.get("src_txt") or row.get("src_text", "")
            ref_text = row.get("tgt_txt") or row.get("tgt_text", "")
            if not src_text or not ref_text:
                continue

            messages = [{"role": "user", "content": f"Translate this {src} document to {tgt}:\n{src_text}\n"}]
            prompt = tokenizer.chat_prepare_for_model(messages, add_generation_prompt=True, tokenize=False)
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(**inputs, max_new_tokens=MAX_SEQ_LEN, do_sample=False)
            decoded = tokenizer.decode(outputs[0], skip_special_tokens=True)

            preds.append(decoded.strip())
            refs.append(ref_text.strip())

        if preds and refs:
            bleu = sacrebleu.corpus_bleu(preds, [refs])
            chrf = sacrebleu.corpus_chrf(preds, [refs])
            results[pair] = {"BLEU": bleu.score, "chrF2": chrf.score}
            print(f"  BLEU={bleu.score:.2f}, chrF2={chrf.score:.2f}")

    out_path = Path(model_path) / "eval_results.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved evaluation → {out_path}")

# ------------------------------
# Main training flow
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    if torch.cuda.is_available():
        major, _ = torch.cuda.get_device_capability()
        compute_dtype = torch.bfloat16 if major >= 8 else torch.float16
        print(f"[INFO] Using dtype: {compute_dtype}")
    else:
        compute_dtype = torch.float32
        print("[WARN] CUDA not available. Training on CPU will be very slow.")

    print("[INFO] Loading base model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=compute_dtype,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="eager"
    )

    # LoRA configuration
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Preparing training data...")
    training_data = load_streaming_dataset(tokenizer, TRAIN_SPLIT, max_samples=100)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,   # reduced for memory
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=(compute_dtype == torch.bfloat16),
        fp16=(compute_dtype == torch.float16),
        optim="paged_adamw_32bit",       # memory-efficient optimizer
        warmup_ratio=0.1,
        lr_scheduler_type="linear",
        max_grad_norm=0.3,
        report_to="none",
        run_name="gemma3-pralekha",
        dataloader_pin_memory=False,
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=training_data,
        max_seq_length=MAX_SEQ_LEN,  # prevent SFT warning
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model + tokenizer + adapter...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("[INFO] Running evaluation...")
    evaluate_model(OUTPUT_DIR, tokenizer, LANGUAGE_PAIRS, subset=EVAL_SPLIT, max_samples=EVAL_SAMPLES)

if __name__ == "__main__":
    main()
