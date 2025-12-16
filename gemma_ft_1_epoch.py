# ======================================================
# Environment setup
# ======================================================
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

#!pip install git+https://github.com/Unbabel/COMET.git
!pip install evaluate

# ======================================================
# Hugging Face Authentication
# ======================================================
from huggingface_hub import login
login(token="USE_UR_HF_TOKEN")

# ======================================================
# Imports
# ======================================================
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

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "google/gemma-3-4b-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_NEW_TOKENS = 256
BATCH_SIZE = 2
GRAD_ACCUM = 8
EVAL_BATCH_SIZE = 1
MAX_TRAIN_STEPS = None
FULL_DATASET = True
MAX_COLAB_SAMPLES = None

DIRECTIONS = ["eng_hin", "hin_eng"]

# ======================================================
# MODEL PREP
# ======================================================
def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )

    try:
        model.gradient_checkpointing_enable()
    except:
        pass

    return model, tok

# ======================================================
# TRAIN DATASET (NON-STREAMING → TRUE EPOCH)
# ======================================================
def stream_examples_list(max_samples=None, tokenizer=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    split = get_dataset_split_names(dataset_name, config_name)[0]

    def build_prompt(example):
        prompt = f"Translate this {example['src_lang']} text to {example['tgt_lang']}:\n{example['src_txt']}"
        return {
            "messages": [
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": example["tgt_txt"]}
            ]
        }

    dataset = load_dataset(
        dataset_name,
        name=config_name,
        split=split,
        streaming=False   #  REQUIRED FOR EPOCH
    )

    if not FULL_DATASET and max_samples is not None:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(build_prompt)
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names
    )

    return dataset

# ======================================================
# TRAINING (TRUE 1-EPOCH)
# ======================================================
def train_model(max_samples=None):
    model, tok = prepare_model()
    dataset = stream_examples_list(max_samples=max_samples, tokenizer=tok)

    peft_config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules="all-linear"
    )

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        max_steps=MAX_TRAIN_STEPS,
        learning_rate=2e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,          # TRUE FULL EPOCH
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
        peft_config=peft_config
    )

    trainer.train()
    return model, tok, trainer

# ================================================================
# EVALUATION (Bidirectional ENG↔HIN from eng_hin split)
# ================================================================

# ------------------------- EVAL PROMPT --------------------------
def build_eval_prompt_messages(example, src_lang, tgt_lang):
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"
    return [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}
    ]


# -------------------- Streaming Dataset Wrapper -----------------
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

            messages = build_eval_prompt_messages(
                {"src_txt": src_text}, self.src_lang, self.tgt_lang
            )

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip()
            }


# ---------------------- Collate Function ------------------------
def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]

    enc = tokenizer.pad(
        {"input_ids": input_ids},
        padding=True,
        return_tensors="pt"
    )

    return enc["input_ids"], enc["attention_mask"], refs


# -------------------- Generation (SAFE slicing) -----------------
def generate_batch(model, tokenizer, input_ids, attention_mask):
    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids.to(model.device),
            attention_mask=attention_mask.to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            use_cache=False,
            eos_token_id=tokenizer.eos_token_id,
            pad_token_id=tokenizer.pad_token_id
        )

    preds = []
    for i in range(len(outputs)):
        prompt_len = attention_mask[i].sum().item()
        gen_ids = outputs[i][prompt_len:]
        preds.append(
            tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        )

    del outputs
    torch.cuda.empty_cache()
    return preds


# ------------------ Dataset Loader ----------------------
def load_pralekha_split():
    return load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split="eng_hin",
        streaming=True
    )


# ----------------- Evaluation Function --------------------------
def evaluate_direction(model, tokenizer, src_lang, tgt_lang,
                       max_samples=None, batch_size=1):

    model.eval()
    torch.cuda.empty_cache()

    raw_ds = load_pralekha_split()
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    preds, refs = [], []
    processed = 0

    if max_samples is None:
        max_samples = float("inf")

    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")

    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(
            model, tokenizer, input_ids, attention_mask
        )

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


# ------------------------- MAIN (EVAL ONLY) ----------------------------
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
    model, tokenizer, trainer = train_model(max_samples=None)

    results = {}
    for split in DIRECTIONS:
        torch.cuda.empty_cache()
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(
            model,
            tokenizer,
            src,
            tgt,
            batch_size=EVAL_BATCH_SIZE,
            max_samples=None
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")


# ------------------ JSONL EXPORT --------------------------------
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

directions = ["eng_hin", "hin_eng"]
max_samples_export = 100

jsonl_files = []

for split in directions:
    src, tgt = split.split("_")

    raw_ds = load_pralekha_split()
    eval_ds = EvalDataset(raw_ds, tokenizer, src, tgt)

    loader = DataLoader(
        eval_ds,
        batch_size=1,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
    processed = 0

    with open(save_path, "w", encoding="utf-8") as f:
        for input_ids, attention_mask, refs in loader:
            preds = generate_batch(model, tokenizer, input_ids, attention_mask)

            for p, r in zip(preds, refs):
                f.write(json.dumps(
                    {"prediction": p, "reference": r},
                    ensure_ascii=False
                ) + "\n")

            processed += len(refs)
            if processed >= max_samples_export:
                break

    jsonl_files.append(save_path)
    print(f"Saved {processed} examples to {save_path}")


# ------------------ ZIP -----------------------------------------
zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
with zipfile.ZipFile(zip_path, "w") as zipf:
    for f in jsonl_files:
        zipf.write(f, arcname=f.name)

print(f"ZIP saved at: {zip_path}")
