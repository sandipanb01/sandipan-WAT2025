# %% Cell 0 - Installs (run once)
# (You can comment-out installs if you already installed these)
!pip install -q "transformers>=4.35.0" datasets accelerate peft huggingface_hub sentencepiece
!pip install -q trl safetensors

# %% Cell 1 - Environment checks & HF login (interactive)
import os, sys, textwrap
print("Python", sys.version)
import torch
print("CUDA available:", torch.cuda.is_available())

# Login to HF (interactive). If you already set HF_TOKEN env var, tokenizer/model load will pick it up.
from huggingface_hub import login
if not os.environ.get("HF_TOKEN"):
    print(textwrap.dedent("""
    If the model is gated (google/gemma-3-270m-it) you must:
      1) Accept the license on Hugging Face
      2) Run interactive login below and paste your HF token
    """))
    login()  # interactive prompt
else:
    login(os.environ["HF_TOKEN"])

# %% Cell 2 - Main configuration (tweak these)
MODEL_ID = "google/gemma-3-270m-it"
OUTPUT_DIR = "./gemma3_pralekha_lora"
MAX_SEQ_LEN = 4096     # doc-level (warning: large -> OOM risk on small GPUs)
MAX_NEW_TOKENS = 4096  # generation target (kept as information)
BATCH_SIZE = 1         # per-device
GRAD_ACCUM = 4
NUM_EPOCHS = 1
LEARNING_RATE = 2e-4
LANGS = ["ben","guj","hin","kan","mal","mar","ori","pan","tam","tel","urd","eng"]

# For quick debugging/testing in Colab set a small MAX_SAMPLES (e.g. 500 or 1000).
# For full training set MAX_SAMPLES = None (BUT see streaming / hardware notes below).
MAX_SAMPLES = 1000  # set None to try full dataset (not recommended on Colab free tier)

# %% Cell 3 - Load tokenizer & model (force slow tokenizer; load in fp16 on GPU)
from transformers import AutoTokenizer, AutoModelForCausalLM
print("Loading tokenizer (slow) and model (fp16 if GPU available)...")
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, use_fast=False, trust_remote_code=True)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

device_map = "auto" if torch.cuda.is_available() else None
torch_dtype = "auto"
# For the small 270M Gemma, loading in fp16 with device_map='auto' is stable and avoids bitsandbytes+PEFT issues.
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    trust_remote_code=True,
    device_map=device_map,
    torch_dtype=torch.float16 if torch.cuda.is_available() else None,
)

print("Model & tokenizer loaded. dtype:", next(model.parameters()).dtype)

# %% Cell 4 - Attach LoRA (PEFT)
from peft import LoraConfig, get_peft_model
print("Applying LoRA adapters (PEFT)...")
lora_cfg = LoraConfig(
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],  # common module names; PEFT will ignore non-existing
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(model, lora_cfg)

# Print trainable params
def print_trainable(m):
    trainable = 0
    total = 0
    for n, p in m.named_parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    print(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.4f}%)")
print_trainable(model)

# %% Cell 5 - Load Pralekha train config (safe load) and filter language pairs
from datasets import load_dataset
print("Loading Pralekha dataset (train config)...")
ds = load_dataset("ai4bharat/Pralekha", "train", split="train", use_auth_token=True)

print("Columns:", ds.column_names)
# Expected columns: ['src_lang','src_txt','tgt_lang','tgt_txt']
# Filter rows to only our LANGS and non-equal src/tgt
def keep_pair(example):
    s = example.get("src_lang")
    t = example.get("tgt_lang")
    if s is None or t is None: 
        return False
    return (s in LANGS and t in LANGS and s != t)

# Note: If MAX_SAMPLES is small we will take a sample after filtering
ds = ds.filter(keep_pair)
print("Filtered dataset size (lang pairs):", len(ds))

if MAX_SAMPLES is not None:
    ds = ds.select(range(min(MAX_SAMPLES, len(ds))))
    print("Using small sample for debug. length ->", len(ds))
else:
    print("Using full dataset (be careful - this may not fit on Colab).")

# %% Cell 6 - Utility: build chat string & tokenization that masks prompt (ONLY assistant loss)
# We'll attempt to use tokenizer.apply_chat_template if available; otherwise fallback to a stable manual template.
print("Preparing tokenization utilities...")

def build_messages(src_lang, tgt_lang, src_txt, tgt_txt):
    """Return messages list (user then assistant)."""
    user_msg = f"Translate the following document from {src_lang} to {tgt_lang}:\n\n{src_txt}"
    assistant_msg = tgt_txt
    return [{"role":"user","content":user_msg}, {"role":"assistant","content":assistant_msg}]

# Try to detect apply_chat_template on tokenizer
use_apply_chat = hasattr(tokenizer, "apply_chat_template")
if use_apply_chat:
    print("Detected tokenizer.apply_chat_template -> will try to use it where possible.")
else:
    print("tokenizer.apply_chat_template NOT detected -> using stable manual chat format.")

# Tokenize a batch and return input_ids, attention_mask, labels (labels=-100 for prompt positions)
def tokenize_and_mask(batch):
    """
    batch is a dict with lists: src_lang, tgt_lang, src_txt, tgt_txt
    Returns dict of lists: input_ids, attention_mask, labels
    """
    input_ids_batch = []
    attention_mask_batch = []
    labels_batch = []
    for src_lang, tgt_lang, src_txt, tgt_txt in zip(batch["src_lang"], batch["tgt_lang"], batch["src_txt"], batch["tgt_txt"]):
        messages = build_messages(src_lang, tgt_lang, src_txt, tgt_txt)

        # If apply_chat_template available, try to use it to construct the textual representation
        prompt_text = None
        full_text = None
        if use_apply_chat:
            try:
                # many tokenizers implement apply_chat_template(messages, add_generation_prompt=..., tokenize=False)
                # but the signature can vary; we try a safe call and fall back on exception
                prompt_text = tokenizer.apply_chat_template([messages[0]], add_generation_prompt=False, tokenize=False)
                full_text = tokenizer.apply_chat_template(messages, add_generation_prompt=False, tokenize=False)
            except Exception:
                # fallback to manual textual template
                use_apply_chat_local = False
                prompt_text = messages[0]["content"]
                full_text = messages[0]["content"] + "\n\n" + messages[1]["content"]
        else:
            # manual chat text: include a short system/instruction then user/assistant markers
            prompt_text = f"User: {messages[0]['content']}\n\nAssistant:"
            # full_text should place the assistant content after the assistant marker (so we can detect start)
            full_text = f"User: {messages[0]['content']}\n\nAssistant: {messages[1]['content']}"
        
        # Tokenize prompt and full using same tokenizer settings
        # We compute lengths on tokenized (not padded) forms
        prompt_enc = tokenizer(prompt_text, truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
        full_enc = tokenizer(full_text, truncation=True, max_length=MAX_SEQ_LEN, padding="max_length", return_tensors=None, add_special_tokens=True)

        input_ids = full_enc["input_ids"]
        attention_mask = full_enc.get("attention_mask", [1]*len(input_ids))

        # Determine prompt token length (number of tokens belonging to prompt_text)
        prompt_len = len(prompt_enc["input_ids"])
        # If prompt_len > len(input_ids) due to truncation, clamp
        if prompt_len >= len(input_ids):
            # prompt consumed entire sequence (no assistant tokens are left). This example gives no label tokens -> skip by assigning all -100
            labels = [-100] * len(input_ids)
        else:
            labels = input_ids.copy()
            for i in range(prompt_len):
                labels[i] = -100

        # Ensure lengths
        assert len(input_ids) == len(labels) == len(attention_mask)

        input_ids_batch.append(input_ids)
        attention_mask_batch.append(attention_mask)
        labels_batch.append(labels)

    return {
        "input_ids": input_ids_batch,
        "attention_mask": attention_mask_batch,
        "labels": labels_batch
    }

# %% Cell 7 - Apply tokenization to dataset (batched)
print("Tokenizing dataset and creating labels (masking prompt tokens)...")
ds_tokenized = ds.map(
    tokenize_and_mask,
    batched=True,
    batch_size=8,
    remove_columns=ds.column_names,
)

# Set torch format
ds_tokenized.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
print("Tokenized dataset example shapes:", {k: ds_tokenized[0][k].shape if k in ds_tokenized[0] else None for k in ["input_ids","attention_mask","labels"]})

# %% Cell 8 - Trainer (HuggingFace Trainer) with masked labels
from transformers import TrainingArguments, Trainer, DataCollatorWithPadding

print("Setting up Trainer...")
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    num_train_epochs=NUM_EPOCHS,
    learning_rate=LEARNING_RATE,
    fp16=torch.cuda.is_available(),
    logging_steps=50,
    save_total_limit=3,
    save_strategy="steps",
    save_steps=500,
    remove_unused_columns=False,  # we already prepared labels
)

# Use a simple padding collator (our sequences already padded to MAX_SEQ_LEN during tokenization)
data_collator = DataCollatorWithPadding(tokenizer, padding="longest")

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_tokenized,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

# %% Cell 9 - Start training
print("Starting training - this trains on assistant tokens only (labels masked for prompt).")
trainer.train()
print("Training finished. Saving model and tokenizer to:", OUTPUT_DIR)
trainer.save_model(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
