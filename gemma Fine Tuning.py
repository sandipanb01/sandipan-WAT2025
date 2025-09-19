# ==============================================
# Colab Fine-tuning Script: Gemma-3-270M + Pralekha
# ==============================================

# -------------------- Install Dependencies --------------------
!pip install -q transformers==4.44.2 trl==0.11.4 peft==0.11.1 bitsandbytes==0.43.3 datasets accelerate sentencepiece

# -------------------- Imports --------------------
import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model
from datasets import load_dataset
from trl import SFTTrainer, SFTConfig

# -------------------- Load Tokenizer --------------------
model_id = "google/gemma-3-270m"

# Force slow tokenizer to avoid Colab "ModelWrapper" crash
tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=False)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"


# -------------------- Load Model with 4-bit Quantization --------------------
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype=torch.bfloat16,
)

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    quantization_config=bnb_config,
    device_map="auto"
)

# -------------------- Apply LoRA --------------------
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # standard for transformers
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_config)

# -------------------- Load Dataset --------------------
print("[INFO] Loading Pralekha train split...")
ds = load_dataset("ai4bharat/Pralekha", "train", split="train")

print("[INFO] Example row:", ds[0])

# -------------------- Preprocessing --------------------
def preprocess(batch):
    sources = batch["src_txt"]
    targets = batch["tgt_txt"]

    conversations = []
    for src, tgt in zip(sources, targets):
        conversations.append(
            [
                {"role": "user", "content": src},
                {"role": "assistant", "content": tgt},
            ]
        )

    tokenized = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        truncation=True,
        max_length=1024,
        return_tensors=None,
    )
    return {"input_ids": tokenized}

print("[INFO] Tokenizing dataset...")
tokenized_ds = ds.map(
    preprocess,
    batched=True,
    remove_columns=["src_lang", "src_txt", "tgt_lang", "tgt_txt"],  # ✅ only valid cols
)

print("[INFO] Tokenized example:", tokenized_ds[0])

# -------------------- Training Config --------------------
training_config = SFTConfig(
    output_dir="./gemma3-pralekha-ft",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    learning_rate=2e-4,
    num_train_epochs=1,
    logging_steps=10,
    save_steps=200,
    save_total_limit=2,
    fp16=True,
    optim="paged_adamw_32bit",
    report_to="none",
)

# -------------------- Trainer --------------------
trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    args=training_config,
    packing=True,  # efficient for long sequences
)

# -------------------- Train --------------------
trainer.train()

# -------------------- Save --------------------
trainer.save_model("./gemma3-pralekha-ft")
tokenizer.save_pretrained("./gemma3-pralekha-ft")

print("[INFO] Training complete. Model saved at ./gemma3-pralekha-ft")
