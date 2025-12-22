# ======================================================
# ✅ Universal 2-Epoch BIDIRECTIONAL LoRA Fine-tuning + Evaluation
# ENG↔HIN (true supervised both ways)
# Token length histograms collected during evaluation
# ======================================================

import os, json, zipfile, torch
from pathlib import Path
from datasets import load_dataset, concatenate_datasets
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from tqdm import tqdm
import sacrebleu
import matplotlib.pyplot as plt

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 2
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 100

# logical directions
DIRECTIONS = ["eng_hin", "hin_eng"]

# ------------------------------ TOKEN STATS
INPUT_TOKEN_LENS = []
OUTPUT_TOKEN_LENS = []

# ------------------------------ MODEL PREP
def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, tokenizer

# ------------------------------ TRAIN DATASET (BIDIRECTIONAL FIX)
def stream_examples_list(max_samples=None, tokenizer=None):
    base_ds = load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split="eng_hin",
        streaming=False
    )

    if not FULL_DATASET and max_samples is not None:
        base_ds = base_ds.select(range(max_samples))

    # eng → hin
    def build_eng_hin(example):
        return {
            "messages": [
                {"role": "user", "content": f"Translate this eng text to hin:\n{example['src_txt']}"},
                {"role": "assistant", "content": example["tgt_txt"]}
            ]
        }

    # hin → eng (SWAPPED)
    def build_hin_eng(example):
        return {
            "messages": [
                {"role": "user", "content": f"Translate this hin text to eng:\n{example['tgt_txt']}"},
                {"role": "assistant", "content": example["src_txt"]}
            ]
        }

    ds_eng_hin = base_ds.map(build_eng_hin)
    ds_hin_eng = base_ds.map(build_hin_eng)

    dataset = concatenate_datasets([ds_eng_hin, ds_hin_eng])

    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names
    )

    return dataset

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tokenizer = prepare_model()
    dataset = stream_examples_list(max_samples=max_samples, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=256,
        lora_alpha=16,
        target_modules="all-linear"
    )

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
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
    return model, tokenizer, trainer

# ------------------------------ EVAL DATASET
class EvalDataset(IterableDataset):
    def __init__(self, dataset, tokenizer, src_lang, tgt_lang):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

    def __iter__(self):
        for ex in self.dataset:
            if self.src_lang == "eng":
                src_text = ex["src_txt"]
                ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]
                ref_text = ex["src_txt"]

            messages = [
                {"role": "user",
                 "content": f"Translate this {self.src_lang} text to {self.tgt_lang}:\n{src_text}"},
                {"role": "assistant", "content": ""}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip()
            }

# ------------------------------ COLLATE
def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]
    enc = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], refs

# ------------------------------ GENERATION
def generate_batch(model, tokenizer, input_ids, attention_mask, collect_tokens=True):
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

        if collect_tokens:
            INPUT_TOKEN_LENS.append(prompt_len)
            OUTPUT_TOKEN_LENS.append(len(gen_ids))

        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return preds

# ------------------------------ LOAD SPLIT (ONLY eng_hin EXISTS)
def load_pralekha_split(max_samples):
    ds = load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split="eng_hin",
        streaming=False
    )
    if not FULL_DATASET and max_samples is not None:
        ds = ds.select(range(max_samples))
    return ds

# ------------------------------ EVALUATION
def evaluate_direction(model, tokenizer, src_lang, tgt_lang,
                       max_samples=None, batch_size=EVAL_BATCH_SIZE):

    model.eval()
    raw_ds = load_pralekha_split(max_samples)
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)

    loader = DataLoader(
        eval_ds,
        batch_size=batch_size,
        collate_fn=partial(eval_collate_fn, tokenizer=tokenizer),
        num_workers=0
    )

    preds, refs = [], []
    pbar = tqdm(total=len(raw_ds), desc=f"Evaluating {src_lang}→{tgt_lang}")

    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask)
        preds.extend(batch_preds)
        refs.extend(batch_refs)
        pbar.update(len(batch_refs))

    pbar.close()

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [refs]).score
    print(f"{src_lang}→{tgt_lang} | BLEU={bleu:.2f} | chrF={chrf:.3f}\n")
    return preds, refs, bleu, chrf

# ------------------------------ HISTOGRAMS
def plot_hist(data, title, xlabel, path):
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # -------- TRAIN
    model, tokenizer, trainer = train_model(max_samples=max_samples)
    del trainer
    torch.cuda.empty_cache()

    # -------- EVALUATION
    results, all_preds_refs = {}, {}

    for split in DIRECTIONS:
        src, tgt = split.split("_")
        preds, refs, bleu, chrf = evaluate_direction(
            model, tokenizer, src, tgt, max_samples=max_samples
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}
        all_preds_refs[split] = list(zip(preds, refs))

    # -------- RESULTS
    print("\n✅ Final Results (ENG↔HIN):")
    for k, v in results.items():
        print(f"{k}: BLEU={v['BLEU']:.2f}, chrF={v['chrF']:.3f}")

    # -------- SAVE
    jsonl_files = []
    for split in DIRECTIONS:
        p = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
        with open(p, "w", encoding="utf-8") as f:
            for pr, rf in all_preds_refs[split]:
                f.write(json.dumps({"prediction": pr, "reference": rf}, ensure_ascii=False) + "\n")
        jsonl_files.append(p)

    with zipfile.ZipFile(OUTPUT_DIR / "pred_refs_eng_hin.zip", "w") as z:
        for f in jsonl_files:
            z.write(f, arcname=f.name)

    plot_hist(INPUT_TOKEN_LENS, "Input Prompt Token Length Distribution", "Tokens",
              OUTPUT_DIR / "input_token_histogram.png")
    plot_hist(OUTPUT_TOKEN_LENS, "Generated Output Token Length Distribution", "Tokens",
              OUTPUT_DIR / "output_token_histogram.png")

    print("\n🎉 DONE — TRUE bidirectional training + evaluation complete.")

from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig, apply_chat_template
from torch.utils.data import DataLoader, IterableDataset
from functools import partial
from tqdm import tqdm
import sacrebleu
import matplotlib.pyplot as plt

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./universal_output_best")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
MAX_NEW_TOKENS = 512
BATCH_SIZE = 1
GRAD_ACCUM = 4
NUM_EPOCHS = 2
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 100
DIRECTIONS = ["eng_hin", "hin_eng"]

# ------------------------------ TOKEN STATS
INPUT_TOKEN_LENS = []
OUTPUT_TOKEN_LENS = []

# ------------------------------ MODEL PREP
def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        device_map="auto"
    )

    try:
        model.gradient_checkpointing_enable()
    except Exception:
        pass

    return model, tokenizer

# ------------------------------ TRAIN DATASET
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

    dataset = load_dataset(dataset_name, name=config_name, split=split, streaming=False)

    if not FULL_DATASET and max_samples is not None:
        dataset = dataset.select(range(max_samples))

    dataset = dataset.map(build_prompt)
    dataset = dataset.map(
        apply_chat_template,
        fn_kwargs={"tokenizer": tokenizer},
        remove_columns=dataset.column_names
    )

    return dataset

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tokenizer = prepare_model()
    dataset = stream_examples_list(max_samples=max_samples, tokenizer=tokenizer)

    peft_config = LoraConfig(
        r=256,
        lora_alpha=16,
        target_modules="all-linear"
    )

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=NUM_EPOCHS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1,
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
    return model, tokenizer, trainer

# ------------------------------ EVAL DATASET
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

            messages = [
                {"role": "user", "content": f"Translate this {self.src_lang} text to {self.tgt_lang}:\n{src_text}"},
                {"role": "assistant", "content": ""}
            ]

            input_ids = self.tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True
            )

            yield {
                "input_ids": torch.tensor(input_ids, dtype=torch.long),
                "reference": ref_text.strip()
            }

# ------------------------------ COLLATE
def eval_collate_fn(batch, tokenizer):
    input_ids = [x["input_ids"] for x in batch]
    refs = [x["reference"] for x in batch]

    enc = tokenizer.pad({"input_ids": input_ids}, padding=True, return_tensors="pt")
    return enc["input_ids"], enc["attention_mask"], refs

# ------------------------------ GENERATION
def generate_batch(model, tokenizer, input_ids, attention_mask, collect_tokens=True):
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

        if collect_tokens:
            INPUT_TOKEN_LENS.append(prompt_len)
            OUTPUT_TOKEN_LENS.append(len(gen_ids))

        preds.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())
    return preds

# ------------------------------ LOAD SPLIT
def load_pralekha_split(split="eng_hin"):
    return load_dataset("ai4bharat/Pralekha", name="train", split=split, streaming=True)

# ------------------------------ EVALUATION
def evaluate_direction(model, tokenizer, src_lang, tgt_lang,
                       max_samples=None, batch_size=EVAL_BATCH_SIZE, collect_tokens=True):

    model.eval()
    raw_ds = load_pralekha_split()
    eval_ds = EvalDataset(raw_ds, tokenizer, src_lang, tgt_lang)
    loader = DataLoader(eval_ds, batch_size=batch_size, collate_fn=partial(eval_collate_fn, tokenizer=tokenizer), num_workers=0)

    preds, refs = [], []
    processed = 0
    if max_samples is None:
        max_samples = float("inf")

    pbar = tqdm(desc=f"Evaluating {src_lang}→{tgt_lang}")
    for input_ids, attention_mask, batch_refs in loader:
        batch_preds = generate_batch(model, tokenizer, input_ids, attention_mask, collect_tokens=collect_tokens)
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
    return preds, refs, bleu, chrf

# ------------------------------ HISTOGRAMS
def plot_hist(data, title, xlabel, path):
    plt.figure()
    plt.hist(data, bins=50)
    plt.xlabel(xlabel)
    plt.ylabel("Frequency")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # ---------------- TRAIN
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer, trainer = train_model(max_samples=max_samples)
    del trainer
    torch.cuda.empty_cache()

    # ---------------- EVALUATION
    results = {}
    all_preds_refs = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        preds, refs, bleu, chrf = evaluate_direction(
            model, tokenizer, src, tgt, max_samples=max_samples, collect_tokens=True
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}
        all_preds_refs[split] = list(zip(preds, refs))

    # ---------------- PRINT FINAL SCORES
    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")

    # ---------------- JSONL EXPORT + TOP 10
    OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
    jsonl_files = []
    for split in DIRECTIONS:
        save_path = OUTPUT_DIR / f"{split}_pred_refs.jsonl"
        with open(save_path, "w", encoding="utf-8") as f:
            for i, (p, r) in enumerate(all_preds_refs[split]):
                f.write(json.dumps({"prediction": p, "reference": r}, ensure_ascii=False) + "\n")
        jsonl_files.append(save_path)
        print(f"\nSaved {len(all_preds_refs[split])} examples to {save_path}")

        # Top 10 preview
        print(f"\n📄 Top 10 predictions for {split}:")
        for p, r in all_preds_refs[split][:10]:
            print(f"P: {p}\nR: {r}\n---")

    # ---------------- ZIP
    zip_path = OUTPUT_DIR / "pred_refs_eng_hin.zip"
    with zipfile.ZipFile(zip_path, "w") as zipf:
        for f in jsonl_files:
            zipf.write(f, arcname=f.name)
    print(f"\nZIP saved at: {zip_path}")

    # ---------------- TOKEN HISTOGRAMS (from evaluation)
    if INPUT_TOKEN_LENS and OUTPUT_TOKEN_LENS:
        plot_hist(INPUT_TOKEN_LENS, "Input Prompt Token Length Distribution", "Tokens", OUTPUT_DIR / "input_token_histogram.png")
        plot_hist(OUTPUT_TOKEN_LENS, "Generated Output Token Length Distribution", "Tokens", OUTPUT_DIR / "output_token_histogram.png")
        print("📊 Token length histograms saved.")
