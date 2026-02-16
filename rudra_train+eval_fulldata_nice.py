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
# 1. IMPORTS
# ============================================================
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from pathlib import Path
import matplotlib.pyplot as plt
from trl.data_utils import is_conversational
import sacrebleu
import langid
import json
import pandas as pd
from tqdm import tqdm
import random

# ============================================================
# 2. TRAINING DATASET TOGGLE
# ============================================================
MAX_TRAIN_SAMPLES = None  # None = full train set, or integer
MAX_EVAL_SAMPLES = None   # None = full eval set, or integer for quick test
MAX_SRC_LEN = 2400
MAX_TGT_LEN = 2400
MAX_LENGTH = MAX_SRC_LEN + MAX_TGT_LEN

# ============================================================
# 3. LOAD OFFICIAL TRAIN SPLIT
# ============================================================
dataset = load_dataset("ai4bharat/pralekha", data_dir="train")

# ENG→HIN filter
dataset['train'] = dataset['train'].filter(
    lambda x: x["src_lang"]=="eng" and x["tgt_lang"]=="hin" and x["src_txt"]!=x["tgt_txt"],
    num_proc=4
)

if MAX_TRAIN_SAMPLES is not None:
    dataset['train'] = dataset['train'].shuffle(seed=42).select(range(MAX_TRAIN_SAMPLES))

train_ds = dataset['train']

# ============================================================
# 4. FORMAT EXAMPLES + CHAT TEMPLATE
# ============================================================
model_name = "google/gemma-3-270m-it"
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
tokenizer.pad_token = tokenizer.eos_token

def format_example(example):
    return {
        "prompt": [{"role": "user", "content": f"Translate the following sentence from English to Hindi.\n\nEnglish: {example['src_txt']}"}],
        "completion": [{"role": "assistant", "content": example["tgt_txt"]}]
    }

train_ds = train_ds.map(
    format_example,
    num_proc=2,
    remove_columns=train_ds.column_names
)

train_ds = train_ds.map(
    apply_chat_template,
    fn_kwargs={"tokenizer": tokenizer},
    num_proc=2,
    remove_columns=train_ds.column_names
)

print("Train sample:", train_ds[0])
print("Is conversational?", is_conversational(train_ds[0]))

# ============================================================
# 5. MODEL + LoRA
# ============================================================
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"
    ]
)

sft_config = SFTConfig(
    output_dir="./gemma-eng-hin-bi",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=8,
    num_train_epochs=2,
    learning_rate=2e-4,
    gradient_checkpointing=True,
    lr_scheduler_type="cosine",
    warmup_ratio=0.1,
    weight_decay=0.01,
    logging_steps=50,
    bf16=True,
    max_length=MAX_LENGTH,
    packing=False,
    report_to="none",
    completion_only_loss=True
)

trainer = SFTTrainer(
    model=model,
    train_dataset=train_ds,
    peft_config=lora_config,
    args=sft_config,
    processing_class=tokenizer
)

# ============================================================
# 6. TRAINING
# ============================================================
trainer.train()

# ============================================================
# 7. SAVE TRAINING LOSS PLOT
# ============================================================
logs = trainer.state.log_history
train_loss = [(x["step"], x["loss"]) for x in logs if "loss" in x]

plt.figure()
plt.plot(*zip(*train_loss), label="Train Loss")
plt.xlabel("Steps")
plt.ylabel("Loss")
plt.title("Training Loss")
plt.legend()
plt.tight_layout()
plt.savefig(Path(sft_config.output_dir) / "train_loss_curve.png")
plt.close()

# ============================================================
# 8. MERGE AND SAVE FINAL MODEL
# ============================================================
merged_model = trainer.model.merge_and_unload()
merged_model = merged_model.to("cpu").eval()
FINAL_MODEL_DIR = Path(sft_config.output_dir) / "final_merged"
FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
merged_model.save_pretrained(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print("✅ TRAINING COMPLETE - Final model saved at:", FINAL_MODEL_DIR)

# ============================================================
# 9. FINAL BIDIRECTIONAL EVALUATION WITH CSV LOG
# ============================================================
EVAL_SPLIT = "test"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4
MAX_NEW_TOKENS = 2048
METRICS_CSV = FINAL_MODEL_DIR / "metrics_log.csv"

def prepare_eval_dataset(src_lang, tgt_lang):
    dataset_eval = load_dataset("ai4bharat/Pralekha", EVAL_SPLIT, split="eng_hin")
    dataset_eval = dataset_eval.filter(
        lambda x: x["src_lang"]==src_lang and x["tgt_lang"]==tgt_lang and x["src_txt"]!=x["tgt_txt"],
        num_proc=4
    )
    if MAX_EVAL_SAMPLES is not None:
        dataset_eval = dataset_eval.shuffle(seed=42).select(range(MAX_EVAL_SAMPLES))

    def build_prompt(example):
        messages = [{"role": "user", "content": f"Translate the following text from {src_lang.upper()} to {tgt_lang.upper()}:\n\n{src_lang.upper()}: {example['src_txt']}"}]
        ref = example["tgt_txt"]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = tokenizer(prompt, truncation=True, padding=False)
        return {"input_ids": tokens["input_ids"], "attention_mask": tokens["attention_mask"], "reference": ref, "src_txt": example["src_txt"]}
    return dataset_eval.map(build_prompt)

def run_eval(dataset_eval, direction_name):
    predictions, references, src_texts = [], [], []

    for i in tqdm(range(0, len(dataset_eval), BATCH_SIZE)):
        batch = dataset_eval[i:i+BATCH_SIZE]
        padded = tokenizer.pad(
            {"input_ids":[b["input_ids"] for b in batch], "attention_mask":[b["attention_mask"] for b in batch]},
            padding=True, return_tensors="pt"
        )
        input_ids = padded["input_ids"].to(DEVICE)
        attention_mask = padded["attention_mask"].to(DEVICE)

        with torch.no_grad():
            outputs = merged_model.generate(
                input_ids=input_ids,
                attention_mask=attention_mask,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )

        new_tokens = outputs[:, input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        predictions.extend(decoded)
        references.extend([b["reference"] for b in batch])
        src_texts.extend([b["src_txt"] for b in batch])

    # Metrics
    bleu = sacrebleu.corpus_bleu(predictions, [references])
    chrf = sacrebleu.corpus_chrf(predictions, [references], beta=2.0)
    script_acc = 100 * sum([langid.classify(p)[0]==direction_name.split("_")[1] for p in predictions]) / len(predictions)

    # Save JSONL
    output_jsonl = FINAL_MODEL_DIR / f"eval_{direction_name}.jsonl"
    with open(output_jsonl, "w", encoding="utf-8") as f:
        for s,r,p in zip(src_texts, references, predictions):
            f.write(json.dumps({"src": s, "ref": r, "pred": p}, ensure_ascii=False)+"\n")

    # Show 5 samples
    print(f"\n=== SAMPLE PREDICTIONS ({direction_name}) ===")
    for i in random.sample(range(len(predictions)), min(5, len(predictions))):
        print(f"SRC: {src_texts[i]}")
        print(f"REF: {references[i]}")
        print(f"PRED: {predictions[i]}")
        print("--------")

    print(f"\n=== {direction_name} EVAL RESULTS ===")
    print("BLEU:", round(bleu.score,2))
    print("chrF2:", round(chrf.score,2))
    print("Strict Script Accuracy:", round(script_acc,2), "%")

    # Save/update CSV log
    row = pd.DataFrame([{
        "Direction": direction_name,
        "BLEU": round(bleu.score,2),
        "chrF2": round(chrf.score,2),
        "ScriptAcc": round(script_acc,2)
    }])
    if METRICS_CSV.exists():
        row.to_csv(METRICS_CSV, mode='a', header=False, index=False)
    else:
        row.to_csv(METRICS_CSV, index=False)

    return {"BLEU": bleu.score, "chrF2": chrf.score, "ScriptAcc": script_acc, "JSONL": str(output_jsonl)}

# ENG→HIN
dataset_eval_eng_hin = prepare_eval_dataset("eng","hin")
metrics_eng_hin = run_eval(dataset_eval_eng_hin, "eng_hin")

# HIN→ENG
dataset_eval_hin_eng = prepare_eval_dataset("hin","eng")
metrics_hin_eng = run_eval(dataset_eval_hin_eng, "hin_eng")

# Save combined metrics JSON
metrics_file = FINAL_MODEL_DIR / "eval_metrics.json"
with open(metrics_file,"w") as f:
    json.dump({"eng_hin": metrics_eng_hin, "hin_eng": metrics_hin_eng}, f, indent=2)

print("\n✅ BIDIRECTIONAL EVAL COMPLETE. Metrics saved at:", metrics_file)
print("✅ CSV log saved at:", METRICS_CSV)
