# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Doc-level Translation)
# ======================================================

# ------------------------------
# Install dependencies (Colab)
# ------------------------------
!pip install -q transformers datasets peft bitsandbytes accelerate trl sacrebleu

# ------------------------------
# Imports
# ------------------------------
import os
from pathlib import Path
import json
import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, apply_chat_template
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import sacrebleu

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha")
MAX_SEQ_LEN = 4096
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
EVAL_SAMPLES = 200

LANGUAGE_PAIRS = [
    "eng_ben", "eng_guj", "eng_hin", "eng_kan", "eng_mal", "eng_mar",
    "eng_ori", "eng_pan", "eng_tam", "eng_tel", "eng_urd",
    "ben_eng", "guj_eng", "hin_eng", "kan_eng", "mal_eng", "mar_eng",
    "ori_eng", "pan_eng", "tam_eng", "tel_eng", "urd_eng"
]

# ------------------------------
# Build prompt utility (doc-level)
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, target_text):
    """
    Build doc-level translation prompt using HF apply_chat_template.
    Trains only on assistant response.
    """
    return apply_chat_template(
        instruction=f"Translate this document from {src_lang} to {tgt_lang}.",
        input_text=src_text,
        output_text=target_text
    )

# ------------------------------
# Load dataset (streaming)
# ------------------------------
def load_training_dataset(tokenizer, split=TRAIN_SPLIT):
    all_instances = []

    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        print(f"[INFO] Loading {src} → {tgt} ({split})...")
        try:
            ds = load_dataset("ai4bharat/Pralekha", split=split)
        except Exception as e:
            print(f"  [WARN] Skipping {pair}: {e}")
            continue

        for row in ds:
            src_text = row.get(f"doc.{src}_2_{tgt}")
            tgt_text = row.get(f"doc.{src}_2_{tgt}")  # assistant output only
            if not src_text or not tgt_text:
                continue

            prompt = build_prompt(src_text, src, tgt, tgt_text)
            all_instances.append({"prompt": prompt})

    return Dataset.from_list(all_instances)

# ------------------------------
# Main training function
# ------------------------------
def main():
    print("[INFO] Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("[INFO] Loading model...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    # ------------------------------
    # LoRA Config
    # ------------------------------
    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=["q_proj", "v_proj"],
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    # ------------------------------
    # Load training data
    # ------------------------------
    print("[INFO] Loading training dataset...")
    train_dataset = load_training_dataset(tokenizer, TRAIN_SPLIT)

    # ------------------------------
    # Training Arguments
    # ------------------------------
    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=3,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        fp16=torch.cuda.is_available(),
        bf16=torch.cuda.is_available(),
        report_to="none",
        run_name="gemma3-pralekha"
    )

    # ------------------------------
    # SFT Trainer
    # ------------------------------
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=False,  # doc-level
        args=training_args,
        peft_config=lora_config,
        max_new_tokens=MAX_SEQ_LEN  # ensure full document translation
    )

    # ------------------------------
    # Start training
    # ------------------------------
    print("[INFO] Starting fine-tuning...")
    trainer.train()

    print("[INFO] Saving model and tokenizer...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("[INFO] Fine-tuning complete!")

if __name__ == "__main__":
    main()
