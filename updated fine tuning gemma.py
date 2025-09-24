# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Loss Debug Fix)
# ======================================================

import os
import json
from pathlib import Path
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 1024
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
EVAL_SAMPLES = 200

LANGUAGE_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal","eng_mar","eng_ori","eng_pan",
    "eng_tam","eng_tel","eng_urd","ben_eng","guj_eng","hin_eng","kan_eng","mal_eng",
    "mar_eng","ori_eng","pan_eng","tam_eng","tel_eng","urd_eng",
]

# ------------------------------
# Manual prompt
# ------------------------------
def build_prompt_manual(src_text, src_lang, tgt_lang):
    return f"Translate this {src_lang} document to {tgt_lang}:\n{src_text}\nAssistant: "

# ------------------------------
# Dataset builder
# ------------------------------
def load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=500):  # increased samples
    examples = []
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        try:
            ds = load_dataset("ai4bharat/Pralekha", split=split, data_dir=split, streaming=False)
        except Exception as e:
            print(f"  [WARN] cannot load split={split} for pair {pair}: {e}")
            continue

        added = 0
        for row in ds:
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
                "labels": labels
            })

            added += 1
            if max_samples and added >= max_samples:
                break
    return Dataset.from_list(examples)

# ------------------------------
# Main training flow
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True, from_slow=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    compute_dtype = torch.float32  # switched to fp32 to see real loss
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
        r=16,  # increased rank to get more trainable params
        lora_alpha=16,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    print("[INFO] Building training dataset...")
    training_data = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=500)

    from trl import SFTTrainer, SFTConfig

    training_config = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=1,  # reduced to see per-step loss
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=1,  # log every step
        save_strategy="epoch",
        eval_strategy="no",
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

if __name__ == "__main__":
    main()
