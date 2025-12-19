thsi is the final ibt code now: # -*- coding: utf-8 -*-
# ======================================================
# ✅ CORRECTED & CONSTRAINED IBT PIPELINE (FINAL)
# IndicTrans R0 + Gemma SFT + 2 IBT Rounds
# Model: google/gemma-3-270m-it
# GPU: A6000 (BF16)
# ======================================================

import os, gc, json, random, zipfile, warnings, re
from pathlib import Path
from itertools import islice
from functools import partial

import torch
from torch.utils.data import DataLoader, IterableDataset
from datasets import load_dataset, Dataset

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    AutoModelForSeq2SeqLM,
)

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig, apply_chat_template

import sacrebleu
from tqdm import tqdm

warnings.filterwarnings("ignore")

# ======================================================
# CONFIG
# ======================================================
MODEL_NAME = "google/gemma-3-270m-it"
INDIC_TO_EN = "ai4bharat/indictrans2-indic-en-1B"
EN_TO_INDIC = "ai4bharat/indictrans2-en-indic-1B"

WORK_DIR = Path("./ibt_pipeline")
OUTPUT_DIR = Path("./ibt_outputs")
WORK_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

N_MONO = 100
N_TEST = 100
MAX_NEW_TOKENS = 256
MAX_STEPS = 100

BATCH_SIZE = 1
GRAD_ACCUM = 2
EVAL_BATCH_SIZE = 2

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEED = 42

random.seed(SEED)
torch.manual_seed(SEED)
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

# ======================================================
# LANGUAGE FILTERS (CRITICAL FIX)
# ======================================================
DEVANAGARI_RE = re.compile(r"[\u0900-\u097F]")
LATIN_RE = re.compile(r"[A-Za-z]")

def is_hindi(text):
    return bool(DEVANAGARI_RE.search(text))

def is_english(text):
    return bool(LATIN_RE.search(text))

def filter_lang_pairs(srcs, tgts, src_check, tgt_check):
    clean_src, clean_tgt = [], []
    for s, t in zip(srcs, tgts):
        if src_check(s) and tgt_check(t):
            clean_src.append(s)
            clean_tgt.append(t)
    return clean_src, clean_tgt

# ======================================================
# LOAD PRALEKHA
# ======================================================
stream = load_dataset("ai4bharat/Pralekha", name="train", split="eng_hin", streaming=True)
train_samples = list(islice(stream, N_MONO))

en_docs = [x["src_txt"] for x in train_samples]
hi_docs = [x["tgt_txt"] for x in train_samples]

stream_test = load_dataset("ai4bharat/Pralekha", name="train", split="eng_hin", streaming=True)
test_samples = list(islice(stream_test, N_MONO, N_MONO + N_TEST))

test_en = [x["src_txt"] for x in test_samples]
test_hi = [x["tgt_txt"] for x in test_samples]

assert not set(en_docs) & set(test_en)

# ======================================================
# ROUND-0: INDIC TRANS
# ======================================================
from IndicTransToolkit.processor import IndicProcessor
from indicnlp.tokenize.sentence_tokenize import sentence_split

ip = IndicProcessor(inference=True)

def init_indic(ckpt):
    tok = AutoTokenizer.from_pretrained(ckpt, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt, torch_dtype=torch.float32
    ).to(DEVICE).eval()
    return tok, model

def translate_sentences(sents, src, tgt, model, tok):
    inp = ip.preprocess_batch(sents, src_lang=src, tgt_lang=tgt)
    enc = tok(inp, return_tensors="pt", padding=True, truncation=True).to(DEVICE)
    out = model.generate(**enc, max_length=256, use_cache=False)
    dec = tok.batch_decode(out, skip_special_tokens=True)
    return ip.postprocess_batch(dec, lang=tgt)

def translate_docs(docs, src, tgt, lang):
    out_docs = []
    for d in docs:
        sents = sentence_split(d, lang)
        trans = translate_sentences(sents, src, tgt, model, tok)
        out_docs.append(" ".join(trans))
    return out_docs

tok, model = init_indic(INDIC_TO_EN)
bt_en_r0 = translate_docs(hi_docs, "hin_Deva", "eng_Latn", "hin")
del model, tok; gc.collect(); torch.cuda.empty_cache()

tok, model = init_indic(EN_TO_INDIC)
bt_hi_r0 = translate_docs(en_docs, "eng_Latn", "hin_Deva", "eng")
del model, tok; gc.collect(); torch.cuda.empty_cache()

# ======================================================
# BUILD SFT DATASET
# ======================================================
def build_dataset(src1, tgt1, src2, tgt2):
    rows = []
    for s, t in zip(src1, tgt1):
        rows.append({
            "messages": [
                {"role": "user", "content": f"Translate this English text to Hindi:\n{s}"},
                {"role": "assistant", "content": t}
            ]
        })
    for s, t in zip(src2, tgt2):
        rows.append({
            "messages": [
                {"role": "user", "content": f"Translate this Hindi text to English:\n{s}"},
                {"role": "assistant", "content": t}
            ]
        })
    return Dataset.from_list(rows)

round0_ds = build_dataset(bt_en_r0, hi_docs, bt_hi_r0, en_docs)

# ======================================================
# GEMMA + LORA
# ======================================================
tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
tok.pad_token = tok.eos_token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME, torch_dtype=torch.bfloat16, device_map="auto"
)

model = get_peft_model(
    model,
    LoraConfig(r=32, lora_alpha=64, target_modules="all-linear")
)

# ======================================================
# TRAIN ROUND-0
# ======================================================
cfg = SFTConfig(
    output_dir=str(WORK_DIR / "r0"),
    per_device_train_batch_size=BATCH_SIZE,
    gradient_accumulation_steps=GRAD_ACCUM,
    learning_rate=2e-5,
    max_steps=MAX_STEPS,
    bf16=True,
    completion_only_loss=True,
    report_to="none"
)

trainer = SFTTrainer(model=model, args=cfg, train_dataset=round0_ds)
trainer.train()

# ======================================================
# SAFE GENERATION (FIXED)
# ======================================================
def safe_generate(texts, src, tgt):
    preds = []
    for t in texts:
        msgs = [{"role": "user", "content": f"Translate this {src} text to {tgt}:\n{t}"}]
        ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=True)
        out = model.generate(
            torch.tensor([ids]).to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )
        gen = tok.decode(out[0][len(ids):], skip_special_tokens=True).strip()
        preds.append(gen)
    return preds

# ======================================================
# IBT ROUNDS (FILTERED + BLEU-GATED)
# ======================================================
for r in [1, 2]:
    gen_en = safe_generate(hi_docs, "Hindi", "English")
    gen_hi = safe_generate(en_docs, "English", "Hindi")

    gen_en, hi_docs_f = filter_lang_pairs(gen_en, hi_docs, is_english, is_hindi)
    gen_hi, en_docs_f = filter_lang_pairs(gen_hi, en_docs, is_hindi, is_english)

    bleu_en = sacrebleu.corpus_bleu(gen_en, [en_docs_f]).score
    bleu_hi = sacrebleu.corpus_bleu(gen_hi, [hi_docs_f]).score

    if bleu_en < 1.0 or bleu_hi < 1.0:
        print(f"[STOP] IBT round {r} rejected (BLEU too low)")
        break

    ds = build_dataset(gen_en, hi_docs_f, gen_hi, en_docs_f)
    cfg.output_dir = str(WORK_DIR / f"r{r}")

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds)
    trainer.train()

# ======================================================
# DONE
# ======================================================
print("✅ IBT PIPELINE COMPLETE (CONSTRAINED, STABLE, REPRODUCIBLE)")
# ======================================================
# FINAL EVALUATION + JSONL EXPORT + ZIP
# ======================================================
from google.colab import files

def evaluate_and_save(model, tokenizer, src_texts, ref_texts,
                      src_lang, tgt_lang, tag):

    preds = []

    for src in tqdm(src_texts, desc=f"Evaluating {src_lang}→{tgt_lang}"):
        msgs = [{"role": "user",
                 "content": f"Translate this {src_lang} text to {tgt_lang}:\n{src}"}]

        ids = tokenizer.apply_chat_template(
            msgs, tokenize=True, add_generation_prompt=True
        )

        out = model.generate(
            torch.tensor([ids]).to(model.device),
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False
        )

        gen = tokenizer.decode(
            out[0][len(ids):], skip_special_tokens=True
        ).strip()

        preds.append(gen)

    bleu = sacrebleu.corpus_bleu(preds, [ref_texts]).score
    chrf = sacrebleu.metrics.CHRF(word_order=0).corpus_score(preds, [ref_texts]).score

    print(f"\n{tag} | {src_lang}→{tgt_lang}")
    print(f"BLEU = {bleu:.2f}")
    print(f"chrF = {chrf:.2f}")

    # Save JSONL
    jsonl_path = OUTPUT_DIR / f"{tag}_{src_lang}_{tgt_lang}.jsonl"
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for s, p, r in zip(src_texts, preds, ref_texts):
            f.write(json.dumps({
                "input": s,
                "prediction": p,
                "reference": r
            }, ensure_ascii=False) + "\n")

    return bleu, chrf, jsonl_path


# ================= RUN FINAL EVAL =====================
all_jsonls = []

b1, c1, f1 = evaluate_and_save(
    model, tok, test_en, test_hi,
    "English", "Hindi", "FINAL"
)
all_jsonls.append(f1)

b2, c2, f2 = evaluate_and_save(
    model, tok, test_hi, test_en,
    "Hindi", "English", "FINAL"
)
all_jsonls.append(f2)

# ================= ZIP EVERYTHING =====================
zip_path = OUTPUT_DIR / "ibt_results.zip"
with zipfile.ZipFile(zip_path, "w") as z:
    for f in all_jsonls:
        z.write(f, arcname=f.name)

print(f"\n✅ ZIP saved at: {zip_path}")

# ================= COLAB DOWNLOAD =====================
files.download(str(zip_path))
