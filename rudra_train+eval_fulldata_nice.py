# ============================================================
# 0. INSTALL DEPENDENCIES
# ============================================================
!pip install -U \
  transformers \
  datasets \
  accelerate \
  peft \
  trl \
  sentencepiece \
  sacrebleu \
  langid

from huggingface_hub import notebook_login
notebook_login()

# ============================================================
# 0. IMPORTS (TRAIN)
# ============================================================
import torch
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from trl.data_utils import is_conversational
from pathlib import Path
import matplotlib.pyplot as plt
import os
import json
import torch
import numpy as np
import pandas as pd
import unicodedata
import matplotlib.pyplot as plt

from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

import sacrebleu
from langdetect import detect
def safe_detect(text):
    try:
        return detect(text)
    except:
        return "unk"

# ============================================================
# 1. TRAINING DATA PREP
# ============================================================
MAX_TRAIN_SAMPLES = 100
MAX_LENGTH = 4800

dataset = load_dataset("ai4bharat/pralekha", data_dir="train")

eng_hin = dataset["train"].filter(
    lambda x: x["src_lang"]=="eng" and x["tgt_lang"]=="hin" and x["src_txt"]!=x["tgt_txt"],
    num_proc=4
)

hin_eng = dataset["train"].filter(
    lambda x: x["src_lang"]=="hin" and x["tgt_lang"]=="eng" and x["src_txt"]!=x["tgt_txt"],
    num_proc=4
)

def format_example(example):
    if example["src_lang"] == "eng":
        prompt = (
            "Translate the following sentence from English to Hindi.\n\n"
            f"English: {example['src_txt']}"
        )
    else:
        prompt = (
            "Translate the following sentence from Hindi to English.\n\n"
            f"Hindi: {example['src_txt']}"
        )

    return {
        "prompt": [{"role": "user", "content": prompt}],
        "completion": [{"role": "assistant", "content": example["tgt_txt"]}]
    }

eng_hin = eng_hin.map(format_example, remove_columns=eng_hin.column_names, num_proc=4)
hin_eng = hin_eng.map(format_example, remove_columns=hin_eng.column_names, num_proc=4)

train_ds = concatenate_datasets([eng_hin, hin_eng])

if MAX_TRAIN_SAMPLES is not None:
    train_ds = train_ds.shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))

# ============================================================
# 2. TOKENIZER + CHAT TEMPLATE
# ============================================================
model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

train_ds = train_ds.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    remove_columns=train_ds.column_names,
    num_proc=4
)

print("TRAIN SAMPLE:\n", train_ds[0])
print("Is conversational?", is_conversational(train_ds[0]))  # expected False

# ============================================================
# 3. MODEL + LoRA + TRAINING
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.float32
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ]
)

sft_config = SFTConfig(
    output_dir="./gemma-eng-hin-bidirectional",
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    num_train_epochs=10,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=10,
    #bf16=True,
    #max_length=MAX_LENGTH,
    #packing=False,
    report_to="none",
    completion_only_loss=True,
    save_strategy="no"

)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer
)

trainer.train()

# ============================================================
# 4. TRAIN LOSS PLOT
# ============================================================
logs = trainer.state.log_history
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]

plt.figure()
plt.plot(*zip(*train_loss))
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Bidirectional Training Loss")
plt.tight_layout()
plt.savefig(Path(sft_config.output_dir) / "train_loss.png")
plt.close()

# ============================================================
# 5. MERGE + SAVE MODEL
# ============================================================
merged_model = trainer.model.merge_and_unload().to("cpu").eval()

FINAL_DIR = Path(sft_config.output_dir) / "final_merged"
FINAL_DIR.mkdir(parents=True, exist_ok=True)

merged_model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("✅ TRUE BIDIRECTIONAL MODEL SAVED AT:", FINAL_DIR)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "./gemma-eng-hin-bidirectional/final_merged"  # trained model
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# QUICK TOGGLE
SANITY_RUN = True
SANITY_SAMPLES = 50

BATCH_SIZE = 2
MAX_NEW_TOKENS = 512

RESULTS_DIR = Path("./gemma-eng-hin-bidirectional/eval")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
    device_map="auto",
    attn_implementation="sdpa",
)

model.eval()

def build_prompt_eng_hin(example, current_tokenizer):
    # Example will always be from 'eng_hin' split.
    # For ENG->HIN, src_txt is English, tgt_txt is Hindi.
    src = example["src_txt"]
    ref = example["tgt_txt"]

    prompt = (
        "Translate the following text from English to Hindi:\n"
        f"English: {src}\n"
        "Hindi: "
    )

    messages = [{"role": "user", "content": prompt}]

    prompt_text = current_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = current_tokenizer(
        prompt_text, # Use the generated prompt text
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": ref,
        "source": src, # The actual English source
    }


def build_prompt_hin_eng(example, current_tokenizer):
    # Example will always be from 'eng_hin' split.
    # For HIN->ENG, src_txt from example is English, tgt_txt from example is Hindi.
    # So, we need to swap them for prompt building.
    src = example["tgt_txt"] # Hindi text from dataset's target
    ref = example["src_txt"] # English text from dataset's source

    prompt = (
        "Translate the following text from Hindi to English:\n"
        f"Hindi: {src}\n"
        "English: "
    )

    messages = [{"role": "user", "content": prompt}]

    prompt_text = current_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    tokens = current_tokenizer(
        prompt_text, # Use the generated prompt text
        truncation=True,
        padding=False,
    )

    return {
        "input_ids": tokens["input_ids"],
        "attention_mask": tokens["attention_mask"],
        "reference": ref,
        "source": src, # The actual Hindi source
    }

def run_direction(eval_src_lang, eval_tgt_lang, build_fn, tag):

    # Always load the 'eng_hin' split from the test config
    loaded_dataset = load_dataset(DATASET_NAME, EVAL_SPLIT, split="eng_hin")

    if SANITY_RUN:
        loaded_dataset = loaded_dataset.select(range(min(SANITY_SAMPLES, len(loaded_dataset))))

    # Map the dataset using the build_fn, passing additional parameters
    dataset = loaded_dataset.map(
        lambda ex: build_fn(ex, tokenizer), # Pass the tokenizer to the build_fn
        remove_columns=loaded_dataset.column_names
    )


    print(dataset)

    predictions = []
    references = []
    sources = []

    print(f"\nRunning inference: {eval_src_lang} → {eval_tgt_lang}")

    for i in tqdm(range(0, len(dataset), BATCH_SIZE)):

        batch = dataset[i:i+BATCH_SIZE]

        padded = tokenizer.pad(
            {
                "input_ids": batch["input_ids"],
                "attention_mask": batch["attention_mask"],
            },
            padding=True,
            return_tensors="pt",
        )

        input_ids = padded["input_ids"].to(model.device)
        attention_mask = padded["attention_mask"].to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
                temperature=0.1,
                repetition_penalty=1.1
            )

        new_tokens = outputs[:, input_ids.shape[1]:]

        decoded = tokenizer.batch_decode(
            new_tokens,
            skip_special_tokens=True,
        )

        predictions.extend(decoded)
        references.extend(batch["reference"])
        sources.extend(batch["source"])
    bleu = sacrebleu.corpus_bleu(predictions, [references]).score
    chrf = sacrebleu.corpus_chrf(predictions, [references]).score
    exact = np.mean([p.strip() == r.strip() for p, r in zip(predictions, references)])

    lid_acc = np.mean([
        safe_detect(p) == ("hi" if eval_tgt_lang == "hin" else "en")
        for p in predictions
    ])

    df = pd.DataFrame({
        "src": sources,
        "ref": references,
        "pred": predictions,
    })

    df.to_json(
        RESULTS_DIR / f"{tag}.jsonl",
        orient="records",
        lines=True,
        force_ascii=False,
    )

    print(f"\nTOP-5 SAMPLES ({eval_src_lang}→{eval_tgt_lang})")
    print(df.head(5))

    return {
        "direction": f"{eval_src_lang}->{eval_tgt_lang}",
        "BLEU": bleu,
        "chrF": chrf,
        "ExactMatch": exact,
        "LID_Accuracy": lid_acc,
    }
metrics = []

metrics.append(
    run_direction("eng", "hin", build_prompt_eng_hin, "eng_hin")
)

metrics.append(
    run_direction("hin", "eng", build_prompt_hin_eng, "hin_eng")
)

metrics_df = pd.DataFrame(metrics)
metrics_df.to_csv(RESULTS_DIR / "metrics.csv", index=False)

print("\n===== FINAL METRICS =====")
print(metrics_df)
