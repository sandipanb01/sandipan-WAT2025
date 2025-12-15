# ------------------ SETUP ------------------
!pip uninstall -y transformers tokenizers torchvision trl
!pip install -U \
    transformers==4.54.1 \
    tokenizers==0.21.1 \
    datasets==3.5.0 \
    sacrebleu==2.5.1 \
    torch==2.6.0 \
    torchvision==0.21.0 \
    tqdm==4.66.5 \
    peft==0.13.2 \
    bitsandbytes \
    accelerate
!pip install trl==0.25.0
!pip install evaluate

# Hugging Face login
from huggingface_hub import login
login(token="USE_UR_HF_TOKEN")

# ------------------ IMPORTS ------------------
import os, json, zipfile, torch, random
from pathlib import Path
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from tqdm import tqdm
import sacrebleu
from trl import apply_chat_template
import os
import torch
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from datasets import load_dataset
from tqdm import tqdm
import sacrebleu
import json
import zipfile
from pathlib import Path

# ------------------ CONFIG ------------------
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 256
BATCH_SIZE = 2
GRAD_ACCUM = 8
MAX_TRAIN_STEPS = None  # ✅ None ensures full epoch
EVAL_BATCH_SIZE = 8
FULL_DATASET = True
MAX_COLAB_SAMPLES = False

DIRECTIONS = ["eng_hin", "hin_eng"]  # bidirectional evaluation

# ------------------ MODEL PREP ------------------
def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    try: model.gradient_checkpointing_enable()
    except: pass
    return model, tok

# ------------------ STREAM + TOKENIZE ------------------
def stream_examples_list(max_samples=None, tokenizer=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    split = splits[0]

    def build_prompt(example):
        prompt = f"Translate this {example['src_lang']} text to {example['tgt_lang']}:\n{example['src_txt']}"
        messages = {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["tgt_txt"]}
            ]
        }
        return messages

    dataset = load_dataset(dataset_name, split=split, streaming=True, name=config_name)

    if not FULL_DATASET and max_samples is not None:
        dataset = dataset.take(max_samples)

    dataset = dataset.map(build_prompt)
    dataset = dataset.map(apply_chat_template, fn_kwargs={"tokenizer": tokenizer})

    return dataset

# ------------------ TRAINING ------------------
def train_model(max_samples=None):
    model, tok = prepare_model()
    dataset = stream_examples_list(max_samples=max_samples, tokenizer=tok)

    peft_config = LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,  # None = full epoch
        logging_steps=100,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.05,
        gradient_checkpointing=True,
        completion_only_loss=True,
        packing=False
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        peft_config=peft_config,
    )

    trainer.train()
    return model, tok, trainer

# ------------------ EVAL HELPERS ------------------
def build_eval_prompt_messages(example, src_lang, tgt_lang):
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}
    ]

class EvalDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __iter__(self):
        for ex in self.dataset:
            if self.src_lang == "eng" and self.tgt_lang == "hin":
                src_text = ex["src_txt"]
                ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]
                ref_text = ex["src_txt"]

            messages = build_eval_prompt_messages({"src_txt": src_text}, self.src_lang, self.tgt_lang)
            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip()
            }

def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    enc = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], refs

def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )
    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return preds

def load_pralekha_split(lang1, lang2):
    split = "eng_hin"
    print(f"Dataset load info: split='{split}'")
    return load_dataset("ai4bharat/Pralekha", name="train", split=split, streaming=True)

def evaluate_direction(model, tokenizer, src_lang, tgt_lang, max_samples=None, batch_size=8):
    raw_ds = load_pralekha_split(src_lang, tgt_lang)
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate, num_workers=0)
    preds, refs = [], []
    processed = 0
    if max_samples is None:
        max_samples = float("inf")
    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")
    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        processed += len(batch_refs)
        pbar.update(len(batch_refs))
        if processed >= max_samples:
            break
    pbar.close()
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score
    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}\n")
    return bleu, chrf

# ------------------ MAIN ------------------
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer, trainer = train_model(max_samples=max_samples)

    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(model, tokenizer, src, tgt,
                                        batch_size=EVAL_BATCH_SIZE,
                                        max_samples=max_samples)
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")

# ------------------ JSONL EXPORT & ZIP ------------------
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
directions = ["eng_hin", "hin_eng"]
max_samples_export = 100
batch_size = 8
jsonl_files = []

for split in directions:
    src, tgt = split.split("_")
    raw_ds = load_pralekha_split(src, tgt)
    eval_ds = EvalDataset(raw_ds, tokenizer, src, tgt)
    collate = partial(eval_collate_fn, tokenizer=tokenizer)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=collate, num_workers=0)
    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    processed = 0
    with open(save_path, "w", encoding="utf-8") as f:
        for input_ids, attention_mask, refs in loader:
            preds = generate_batch(model, tokenizer, input_ids, attention_mask)
            for p, r in zip(preds, refs):
                f.write(json.dumps({"prediction": p, "reference": r}, ensure_ascii=False) + "\n")
            processed += len(refs)
            if processed >= max_samples_export:
                break
    jsonl_files.append(save_path)
    print(f"Saved {processed} examples to {save_path}")

zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for f in jsonl_files:
        zipf.write(f, arcname=f.name)
print(f"ZIP saved at: {zip_path}")
