# ============================================
# Cell 1 - Setup, GPU check, and HF login
# ============================================
import os, sys, textwrap
print("Python", sys.version)
print("CUDA available:", "yes" if os.environ.get("CUDA_VISIBLE_DEVICES","") else "no")

# Hugging Face login
from huggingface_hub import login
HF_TOKEN = os.environ.get("HF_TOKEN", None)
if not HF_TOKEN:
    print(textwrap.dedent("""
    [INFO] HF_TOKEN not found. You must log in.
    Run the login() prompt below:
    """))
    login()  # This will ask for your token interactively
else:
    login(HF_TOKEN)

# ============================================
# Cell 2 - Install packages
# ============================================
!pip install -q transformers==4.44.2 accelerate bitsandbytes datasets peft trl

# ============================================
# Cell 3 - Config
# ============================================
MODEL_ID = "google/gemma-3-270m-it"   # instruct Gemma-3
OUTPUT_DIR = "./gemma3_pralekha_lora"
MAX_SEQ_LEN = 4096   # document-level context
BATCH_SIZE = 1       # keep 1 for Colab T4
GRAD_ACCUM = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
MAX_SAMPLES = 200    # for debugging; set None for full training

LANGS = ["ben","guj","hin","kan","mal","mar","ori","pan","tam","tel","urd","eng"]

# ============================================
# Cell 4 - Load model + tokenizer
# ============================================
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_quant_type="nf4",
    bnb_4bit_compute_dtype="bfloat16"
)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    quantization_config=bnb_config,
    device_map="auto"
)

# ============================================
# Cell 5 - LoRA config
# ============================================
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj","k_proj","v_proj","o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

model = get_peft_model(model, lora_config)

# ============================================
# Cell 6 - Load Pralekha dataset
# ============================================
from datasets import load_dataset

ds = load_dataset("ai4bharat/Pralekha", "train", split="train")

if MAX_SAMPLES:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))

print(ds)

# ============================================
# Cell 7 - Utility: build chat messages
# ============================================
from transformers import apply_chat_template

def build_prompt(example):
    """Turn src/tgt into a chat for instruct models."""
    src_lang, tgt_lang = example["src_lang"], example["tgt_lang"]
    src_txt, tgt_txt = example["src_txt"], example["tgt_txt"]

    user_msg = f"Translate from {src_lang} to {tgt_lang}:\n{src_txt}"
    assistant_msg = tgt_txt

    chat = [
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": assistant_msg},
    ]
    return {"messages": chat}

ds = ds.map(build_prompt)

# ============================================
# Cell 8 - Tokenization with chat template
# ============================================
def tokenize_fn(example):
    text = apply_chat_template(
        example["messages"],
        tokenizer,
        add_generation_prompt=False
    )
    return tokenizer(
        text,
        truncation=True,
        padding="max_length",
        max_length=MAX_SEQ_LEN
    )

tokenized_ds = ds.map(tokenize_fn, batched=False, remove_columns=ds.column_names)

print(tokenized_ds)

# ============================================
# Cell 9 - Trainer
# ============================================
from trl import SFTTrainer
from transformers import TrainingArguments, DataCollatorForLanguageModeling

args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=LEARNING_RATE,
    num_train_epochs=NUM_EPOCHS,
    logging_dir=f"{OUTPUT_DIR}/logs",
    logging_steps=10,
    save_strategy="epoch",
    evaluation_strategy="no",
    fp16=True,
    save_total_limit=2,
    push_to_hub=False,
)

trainer = SFTTrainer(
    model=model,
    tokenizer=tokenizer,
    train_dataset=tokenized_ds,
    dataset_text_field=None,   # we already tokenized
    max_seq_length=MAX_SEQ_LEN,
    packing=False,             # doc-level, no packing
    args=args,
    data_collator=DataCollatorForLanguageModeling(tokenizer, mlm=False),
)

# ============================================
# Cell 10 - Train
# ============================================
trainer.train()

# ============================================
# Cell 11 - Save model
# ============================================
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)

print("[DONE] Fine-tuned Gemma-3 on Pralekha with LoRA")
