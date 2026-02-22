# ============================================================
# 0. IMPORTS
# ============================================================
import os
import torch
from pathlib import Path
from difflib import SequenceMatcher
from datasets import load_dataset, Value
from transformers import AutoTokenizer, AutoModelForCausalLM, set_seed
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig

# ============================================================
# 1. REPRODUCIBILITY
# ============================================================
set_seed(42)

# ============================================================
# 2. CONFIG
# ============================================================
MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"

OUTPUT_DIR = Path("./gemma3_fewshot_outputs")
CKPT_DIR   = OUTPUT_DIR / "checkpoints"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CKPT_DIR.mkdir(parents=True, exist_ok=True)

FEW_SHOT_K = 2                     # number of in-context examples
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_SEQ_LEN = 8000                 # larger to allow few-shot blocks

# ============================================================
# 3. TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 4. FILTERS (IDENTICAL TO YOUR BASELINE)
# ============================================================
def strict_filter(example):
    s = str(example["src_txt"] or "").lower()
    t = str(example["tgt_txt"] or "").lower()
    return SequenceMatcher(None, s, t).ratio() < 0.65

def length_filter(example):
    src_len = len(tokenizer(example["src_txt"], truncation=False)["input_ids"])
    tgt_len = len(tokenizer(example["tgt_txt"], truncation=False)["input_ids"])
    return src_len <= MAX_SRC_LEN and tgt_len <= MAX_TGT_LEN

def clean_utf8(example):
    example["src_txt"] = example["src_txt"].decode("utf-8", "ignore")
    example["tgt_txt"] = example["tgt_txt"].decode("utf-8", "ignore")
    return example

# ============================================================
# 5. LOAD TRAIN + DEV
# ============================================================
train_raw = load_dataset(DATASET_NAME, "train", split="eng_hin")
dev_raw   = load_dataset(DATASET_NAME, "dev",   split="eng_hin")

def preprocess(ds):
    ds = ds.cast_column("src_txt", Value("binary"))
    ds = ds.cast_column("tgt_txt", Value("binary"))
    ds = ds.map(clean_utf8, num_proc=16)
    ds = ds.cast_column("src_txt", Value("string"))
    ds = ds.cast_column("tgt_txt", Value("string"))
    ds = ds.filter(lambda x: x["src_txt"].strip() and x["tgt_txt"].strip())
    ds = ds.filter(strict_filter)
    ds = ds.filter(length_filter)
    return ds

train_raw = preprocess(train_raw)
dev_raw   = preprocess(dev_raw)

# ============================================================
# 6. FEW-SHOT PROMPT BUILDER (ADVISOR-STYLE)
# ============================================================
def build_fewshot_prompt(batch):
    prompts, completions = [], []

    for i in range(len(batch["src_txt"])):

        if i % 2 == 0:
            # ENG → HIN
            src_lang, tgt_lang = "English", "Hindi"
            src = batch["src_txt"][i]
            tgt = batch["tgt_txt"][i]
            ex_src = batch["src_txt"]
            ex_tgt = batch["tgt_txt"]
        else:
            # HIN → ENG
            src_lang, tgt_lang = "Hindi", "English"
            src = batch["tgt_txt"][i]
            tgt = batch["src_txt"][i]
            ex_src = batch["tgt_txt"]
            ex_tgt = batch["src_txt"]

        # ---- few-shot examples ----
        examples = []
        for j in range(1, FEW_SHOT_K + 1):
            idx = (i - j) % len(batch["src_txt"])
            examples.append(
                f"{src_lang}: {ex_src[idx]}\n\n{tgt_lang}: {ex_tgt[idx]}"
            )

        instruction = (
            f"Translate the given input document to {tgt_lang}. "
            "Generate only the translation. Do not generate any other tokens."
        )

        prompt = (
            f"<start_of_turn>user\n"
            f"{instruction}\n\n"
            + "\n\n".join(examples)
            + f"\n\n{src_lang}: {src}\n\n{tgt_lang}:"
            f"<end_of_turn>\n"
            f"<start_of_turn>model\n"
        )

        completion = f"{tgt}<end_of_turn>"

        prompts.append(prompt)
        completions.append(completion)

    return {"prompt": prompts, "completion": completions}

train_ds = train_raw.map(build_fewshot_prompt, batched=True, remove_columns=train_raw.column_names)
dev_ds   = dev_raw.map(build_fewshot_prompt,   batched=True, remove_columns=dev_raw.column_names)

# ============================================================
# 7. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    attn_implementation="sdpa"
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=64,
    lora_dropout=0.05,
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj",
        "gate_proj","up_proj","down_proj"
    ],
    task_type="CAUSAL_LM",
    bias="none"
)

# ============================================================
# 8. TRAINER
# ============================================================
trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    eval_dataset=dev_ds,
    peft_config=peft_config,
    args=SFTConfig(
        output_dir=str(CKPT_DIR),
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=2e-4,
        num_train_epochs=2,
        logging_steps=400,
        bf16=True,
        eval_strategy="steps",
        eval_steps=2000,
        save_strategy="steps",
        save_steps=2000,
        save_total_limit=20,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        max_length=MAX_SEQ_LEN,
        gradient_checkpointing=True,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        weight_decay=0.01,
        completion_only_loss=True,
        packing=False,
        report_to="none",
        ddp_find_unused_parameters=False
    )
)

trainer.train()

# ============================================================
# 9. MERGE & SAVE
# ============================================================
final_model = trainer.model.merge_and_unload().cpu().eval()
FINAL_DIR = OUTPUT_DIR / "final_merged"
FINAL_DIR.mkdir(exist_ok=True)

final_model.save_pretrained(FINAL_DIR)
tokenizer.save_pretrained(FINAL_DIR)

print("✅ Few-shot fine-tuning complete")
