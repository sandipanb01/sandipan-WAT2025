# ======================================================
# Gemma-IT Fine-tuning for Pralekha (Streaming, Doc-Level MT)
# Shared Task 2025: https://sites.google.com/view/indic-doc?pli=1
# Dataset: https://huggingface.co/datasets/ai4bharat/Pralekha
# ======================================================

# ------------------------------
# Install dependencies
# ------------------------------
!pip install -q git+https://github.com/huggingface/transformers.git
!pip install -q datasets
!pip install -q git+https://github.com/huggingface/trl.git
!pip install -q peft accelerate bitsandbytes sacrebleu

import torch
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments, apply_chat_template
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer
import sacrebleu
from pathlib import Path

# ------------------------------
# Config
# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha-streaming")
MAX_SEQ_LEN = 4096
TRAIN_SPLIT = "train"
EVAL_SPLIT = "dev"
LANGUAGE_PAIRS = [
    "eng_ben","eng_guj","eng_hin","eng_kan","eng_mal","eng_mar",
    "eng_ori","eng_pan","eng_tam","eng_tel","eng_urd",
    "ben_eng","guj_eng","hin_eng","kan_eng","mal_eng","mar_eng",
    "ori_eng","pan_eng","tam_eng","tel_eng","urd_eng"
]

# ------------------------------
# Build prompt utility
# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, target_text=""):
    """
    Build document-level translation prompt using HF chat template
    """
    return apply_chat_template(
        instruction=f"Translate this document from {src_lang} to {tgt_lang}",
        input_text=src_text,
        output_text=target_text
    )

# ------------------------------
# Streaming dataset loader
# ------------------------------
def load_streaming_dataset(tokenizer, split=TRAIN_SPLIT, max_samples=None):
    all_instances = []
    for pair in LANGUAGE_PAIRS:
        src, tgt = pair.split("_")
        col_name = f"doc.{src}_2_{tgt}"
        print(f"[INFO] Streaming {src} → {tgt} ({split})...")
        try:
            ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True)
        except Exception as e:
            print(f"  [WARN] Skipping {pair}: {e}")
            continue

        processed = 0
        for row in ds:
            if col_name not in row:
                continue
            src_text = row[col_name]
            if not src_text:
                continue

            # Build assistant-only prompt
            prompt = build_prompt(src_text, src, tgt, target_text=src_text)

            instance = {
                "input_ids": tokenizer(prompt['input_ids'], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"],
                "labels": tokenizer(prompt['input_ids'], truncation=True, max_length=MAX_SEQ_LEN)["input_ids"]
            }
            all_instances.append(instance)

            processed += 1
            if max_samples and processed >= max_samples:
                break

        print(f"  Added {processed} samples from {pair}")

    return Dataset.from_list(all_instances)

# ------------------------------
# Main training flow
# ------------------------------
def main():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        device_map="auto",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        trust_remote_code=True
    )

    # LoRA config
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

    # Load streaming dataset
    print("[INFO] Preparing streaming training dataset...")
    train_dataset = load_streaming_dataset(tokenizer, split=TRAIN_SPLIT)

    training_args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="no",
        bf16=torch.cuda.is_available(),
        fp16=torch.cuda.is_available(),
        optim="paged_adamw_8bit",
        report_to="none",
        max_grad_norm=0.3,
        run_name="gemma3-pralekha-streaming"
    )

    trainer = SFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        tokenizer=tokenizer,
        max_seq_length=MAX_SEQ_LEN,
        packing=False  # document-level translation
    )

    print("[INFO] Starting training...")
    trainer.train()

    print("[INFO] Saving model + tokenizer + adapters...")
    trainer.save_model()
    tokenizer.save_pretrained(OUTPUT_DIR)
    model.save_pretrained(OUTPUT_DIR)

    print("[INFO] Training completed!")

if __name__ == "__main__":
    main()
