# ======================================================
# Colab-ready LoRA Fine-Tuning Script
# Gemma-3 270M IT + Pralekha Corpus (Doc-level MT)
# ======================================================

# -------------------- Install Packages --------------------
!pip install -q transformers datasets accelerate bitsandbytes peft

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    set_seed,
    apply_chat_template,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch

# -------------------- Global Config --------------------
MODEL_NAME = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
OUTPUT_DIR = "/content/gemma3-pralekha-ft"
MAX_NEW_TOKENS = 4096
BATCH_SIZE = 2
EPOCHS = 2
LR = 2e-4   # higher LR works better with LoRA
SEED = 42

set_seed(SEED)

# -------------------- Load Tokenizer + Base Model --------------------
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    load_in_4bit=True,   # memory efficient
)

# Prepare model for LoRA fine-tuning
base_model = prepare_model_for_kbit_training(base_model)

# -------------------- Apply LoRA --------------------
lora_config = LoraConfig(
    r=64,                      # LoRA rank
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],  
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)
model = get_peft_model(base_model, lora_config)

print("\n[INFO] Trainable parameters with LoRA:")
model.print_trainable_parameters()

# -------------------- Utility: Build Chat Prompt --------------------
def build_prompt(source_text, target_text, src_lang, tgt_lang):
    """
    Wraps source (user) and target (assistant) into a chat-style prompt
    using Hugging Face's apply_chat_template.
    """
    messages = [
        {"role": "user", "content": f"Translate the following document from {src_lang} to {tgt_lang}:\n\n{source_text}"},
        {"role": "assistant", "content": target_text},
    ]
    return apply_chat_template(tokenizer, messages, tokenize=False, add_generation_prompt=False)

# -------------------- Preprocessing Function --------------------
def preprocess_function(examples, src_lang, tgt_lang):
    sources = examples[src_lang]
    targets = examples[tgt_lang]

    prompts = [
        build_prompt(src, tgt, src_lang, tgt_lang)
        for src, tgt in zip(sources, targets)
    ]

    model_inputs = tokenizer(
        prompts,
        max_length=MAX_NEW_TOKENS,
        truncation=True,
        padding="max_length",
    )

    labels = model_inputs["input_ids"].copy()
    model_inputs["labels"] = labels
    return model_inputs

# -------------------- Language Pairs --------------------
LANGUAGES = ["ben", "guj", "hin", "kan", "mal", "mar", "ori", "pan", "tam", "tel", "urd"]
LANGUAGE_PAIRS = [(src, "eng") for src in LANGUAGES] + [("eng", tgt) for tgt in LANGUAGES]

# -------------------- Fine-tune Loop --------------------
for src_lang, tgt_lang in LANGUAGE_PAIRS:
    print(f"\n[INFO] Training {src_lang} → {tgt_lang} ...")

    # Load dataset (train split only, eval from split)
    dataset = load_dataset(DATASET_NAME, split="train")

    # Map preprocessing
    tokenized_dataset = dataset.map(
        lambda ex: preprocess_function(ex, src_lang, tgt_lang),
        batched=True,
        remove_columns=dataset.column_names,
    )

    # Train/val split
    split_dataset = tokenized_dataset.train_test_split(test_size=0.01, seed=SEED)
    train_dataset = split_dataset["train"]
    eval_dataset = split_dataset["test"]

    # Data collator
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    # Training args
    training_args = TrainingArguments(
        output_dir=f"{OUTPUT_DIR}/{src_lang}-{tgt_lang}",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=8,
        learning_rate=LR,
        num_train_epochs=EPOCHS,
        weight_decay=0.01,
        warmup_ratio=0.1,
        logging_steps=50,
        save_strategy="epoch",
        evaluation_strategy="epoch",   # <--- explicit eval_strategy
        predict_with_generate=True,
        bf16=True,
        report_to="none",
        push_to_hub=False,
    )

    # Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        tokenizer=tokenizer,
    )

    # Train
    trainer.train()
    trainer.save_model(f"{OUTPUT_DIR}/{src_lang}-{tgt_lang}/final")

print("\n✅ All language pairs finished LoRA fine-tuning.")
