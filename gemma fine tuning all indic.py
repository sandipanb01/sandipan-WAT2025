# -*- coding: utf-8 -*-
# ======================================================
# ✅ Gemma-IT Fine-tuning for Pralekha (Full Corpus + Streaming + Batched Eval + Safe on T4)
# ======================================================

import os, json, zipfile
from pathlib import Path
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu, evaluate
import matplotlib.pyplot as plt
from itertools import islice

# ------------------------------
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./gemma3-pralekha-full")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)
MAX_SEQ_LEN = 1024
BATCH_SIZE = 8  # for batched evaluation
TRAIN_SAMPLES_PER_PAIR = None  # None = full corpus streaming

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_CODE_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil","tel":"Telugu",
    "mal":"Malayalam","kan":"Kannada","mar":"Marathi","guj":"Gujarati",
    "urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------
def build_prompt(src_text, src_lang, tgt_lang, example_pair, tokenizer):
    example_src, example_tgt = example_pair
    src_name = LANG_CODE_MAP.get(src_lang, src_lang)
    tgt_name = LANG_CODE_MAP.get(tgt_lang, tgt_lang)
    messages = [
        {"role": "user", "content": f"Translate this {src_name} text to {tgt_name}:\n{example_src}"},
        {"role": "assistant", "content": example_tgt},
        {"role": "user", "content": f"Now translate this {src_name} text to {tgt_name}:\n{src_text}"},
        {"role": "assistant", "content": ""}
    ]
    return tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

# ------------------------------
def stream_examples(tokenizer, samples_per_pair=TRAIN_SAMPLES_PER_PAIR):
    """Stream all language directions from full Pralekha corpus."""
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    available_splits = get_dataset_split_names(dataset_name, config_name)

    for split_name in available_splits:
        parts = split_name.split("_")
        if len(parts) != 2:
            continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]:
            continue
        lang = tl if sl == "eng" else sl
        if lang not in INDIAN_LANGS:
            continue

        ds = load_dataset(dataset_name, split=split_name, streaming=True, name=config_name)

        # one-shot example
        one_shot = ("", "")
        for row in ds:
            src_txt = row.get("src_txt") or ""
            tgt_txt = row.get("tgt_txt") or ""
            if len(src_txt.split()) > 5 and len(tgt_txt.split()) > 5:
                one_shot = (src_txt, tgt_txt)
                break

        # re-open for actual streaming
        ds = load_dataset(dataset_name, split=split_name, streaming=True, name=config_name)
        added = 0
        for row in ds:
            src_txt, tgt_txt = row.get("src_txt",""), row.get("tgt_txt","")
            if not src_txt or not tgt_txt:
                continue
            eng, indic = (src_txt, tgt_txt) if sl == "eng" else (tgt_txt, src_txt)
            for src, tgt, direction in [(eng, indic, f"eng_{lang}"), (indic, eng, f"{lang}_eng")]:
                yield {
                    "input_text": build_prompt(src, direction.split("_")[0], direction.split("_")[1], one_shot, tokenizer),
                    "target_text": tgt,
                    "direction": direction
                }
            added += 1
            if samples_per_pair and added >= samples_per_pair // len(available_splits):
                break

# ------------------------------
class PralekhaIterableDataset(IterableDataset):
    def __init__(self, tokenizer, max_seq_len=MAX_SEQ_LEN, samples_per_pair=TRAIN_SAMPLES_PER_PAIR):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.samples_per_pair = samples_per_pair
        self.expected_examples = len(INDIAN_LANGS) * 2 * (samples_per_pair or 50000)

    def __iter__(self):
        for ex in stream_examples(self.tokenizer, samples_per_pair=self.samples_per_pair):
            prompt_ids = self.tokenizer(ex["input_text"], truncation=True, max_length=self.max_seq_len, add_special_tokens=False)["input_ids"]
            target_ids = self.tokenizer(ex["target_text"], truncation=True, max_length=self.max_seq_len)["input_ids"]
            input_ids = (prompt_ids + target_ids)[:self.max_seq_len]
            attention_mask = [1]*len(input_ids)
            labels = [-100]*len(prompt_ids) + target_ids
            labels = labels[:self.max_seq_len]
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ------------------------------
def prepare_model():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    tokenizer.model_input_names = ["input_ids", "attention_mask"]

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float16,
        device_map="auto",
        attn_implementation="eager"
    )

    lora_cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=["q_proj","v_proj","k_proj","o_proj"],
        lora_dropout=0.05, bias="none", task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_cfg), tokenizer

# ------------------------------
def train_model():
    model, tokenizer = prepare_model()
    train_dataset = PralekhaIterableDataset(tokenizer)

    trainer_cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=1.5e-4,
        num_train_epochs=1,
        max_seq_length=MAX_SEQ_LEN,
        logging_steps=10,
        save_strategy="epoch",
        report_to="none",
        max_steps=None,  # Streaming full dataset
    )

    trainer = SFTTrainer(model=model, args=trainer_cfg, train_dataset=train_dataset, tokenizer=tokenizer)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    return model, tokenizer, trainer

# ------------------------------
def batch_iterable(iterable, n=BATCH_SIZE):
    it = iter(iterable)
    while True:
        batch = list(islice(it, n))
        if not batch:
            break
        yield batch

# ------------------------------
def evaluate_model(model, tokenizer, max_new_tokens=128):
    """Memory-safe streaming evaluation with incremental BLEU, chrF, and COMET."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)
    model.eval()
    comet = evaluate.load("comet")
    dataset_name = "ai4bharat/Pralekha"
    config_name = "dev"
    available_splits = get_dataset_split_names(dataset_name, config_name)

    # Prepare files and metric accumulators per direction
    out_files = {}
    metric_accumulators = {}
    for lang in INDIAN_LANGS:
        for direction in [f"eng_{lang}", f"{lang}_eng"]:
            path = OUTPUT_DIR / f"{direction}.jsonl"
            out_files[direction] = open(path, "w", encoding="utf-8")
            metric_accumulators[direction] = {
                "preds": [],
                "refs": [],
                "comet_scores": []
            }

    # Stream dataset
    for split_name in available_splits:
        parts = split_name.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split_name, streaming=True, name=config_name)
        for row in ds:
            src_txt, tgt_txt = row.get("src_txt",""), row.get("tgt_txt","")
            if not src_txt or not tgt_txt: continue
            eng, indic = (src_txt, tgt_txt) if sl=="eng" else (tgt_txt, src_txt)

            for src, tgt, direction in [(eng, indic, f"eng_{lang}"), (indic, eng, f"{lang}_eng")]:
                prompt = build_prompt(src, direction.split("_")[0], direction.split("_")[1], ("Example source","Example target"), tokenizer)
                enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    output = model.generate(**enc, max_new_tokens=max_new_tokens)
                pred_text = tokenizer.decode(output[0], skip_special_tokens=True).strip()

                # Write prediction to file
                out_files[direction].write(json.dumps([pred_text], ensure_ascii=False)+"\n")

                # Incremental metrics
                metric_accumulators[direction]["preds"].append(pred_text)
                metric_accumulators[direction]["refs"].append(tgt)
                if len(metric_accumulators[direction]["preds"]) % 32 == 0:
                    batch_preds = metric_accumulators[direction]["preds"]
                    batch_refs = metric_accumulators[direction]["refs"]
                    batch_comet = comet.compute(predictions=batch_preds, references=batch_refs, sources=[""]*len(batch_refs))["mean_score"]
                    metric_accumulators[direction]["comet_scores"].append(batch_comet)
                    metric_accumulators[direction]["preds"] = []
                    metric_accumulators[direction]["refs"] = []

    # Close files
    for f in out_files.values():
        f.close()

    # Zip submission
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path, "w") as zf:
        for lang in INDIAN_LANGS:
            for direction in [f"eng_{lang}", f"{lang}_eng"]:
                path = OUTPUT_DIR / f"{direction}.jsonl"
                zf.write(path, path.name)
    print(f"✅ Submission saved at {zip_path}")

    # Final metric computation
    print("\n📊 Evaluation Metrics per direction:")
    for lang in INDIAN_LANGS:
        for direction in [f"eng_{lang}", f"{lang}_eng"]:
            preds = metric_accumulators[direction]["preds"]
            refs = metric_accumulators[direction]["refs"]
            comet_scores = metric_accumulators[direction]["comet_scores"]
            if preds and refs:
                bleu = sacrebleu.corpus_bleu(preds,[refs]).score
                chrf = sacrebleu.corpus_chrf(preds,[[r] for r in refs]).score
                comet_final = 0 if not comet_scores else sum(comet_scores)/len(comet_scores)
                print(f"[{direction}] BLEU={bleu:.2f}  chrF={chrf:.2f}  COMET={comet_final:.2f}")

# ------------------------------
def plot_training_metrics(trainer):
    logs = trainer.state.log_history
    steps = [l.get("step") for l in logs if "loss" in l]
    losses = [l.get("loss") for l in logs if "loss" in l]
    plt.figure(figsize=(8,4))
    plt.plot(steps, losses)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"training_metrics.png")
    print("📈 Training metrics saved.")

# ------------------------------
if __name__ == "__main__":
    model, tokenizer, trainer = train_model()
    evaluate_model(model, tokenizer)
    plot_training_metrics(trainer)
