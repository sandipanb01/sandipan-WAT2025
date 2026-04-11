# -*- coding: utf-8 -*-
# ======================================================
# VARIANT 2 — BT Data Generation
# Language-aware sentence splitting:
#   Hindi docs  → split on danda (।)  → translate HI→EN → rejoin with period (.)
#   English docs → split on period (.) → translate EN→HI → rejoin with danda (।)
# ======================================================

import os
import re
import json
import random
import gc
import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from IndicTransToolkit.processor import IndicProcessor
from datasets import load_dataset

# ──────────────────────────────────────────────────────
# CONFIG
# ──────────────────────────────────────────────────────
INDIC_TO_EN_CKPT = "ai4bharat/indictrans2-indic-en-1B"
EN_TO_INDIC_CKPT = "ai4bharat/indictrans2-en-indic-1B"

BATCH_SIZE = 16
N_DOCS     = 20000

OUTPUT_DIR = "./synthetic_bt_docs"
TRAIN_DIR  = "data/train/eng_hin"   # where your local Pralekha JSOLs live

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

DANDA = "\u0964"   # । Devanagari danda


# ──────────────────────────────────────────────────────
# LANGUAGE-AWARE SENTENCE SPLITTING & REJOINING
# ──────────────────────────────────────────────────────
def split_sentences(text: str, lang: str) -> list:
    """
    lang: "hindi" or "english"

    Hindi  → split on danda (।) with period (.) as fallback
    English → split on period (.) using regex that avoids
              splitting on abbreviations/decimals
    """
    if lang == "hindi":
        # Primary: split on danda
        sentences = re.split(r"।+", text)
        # Fallback: if no danda found, split on period
        if len(sentences) <= 1:
            sentences = re.split(r"(?<!\d)\.(?!\d)", text)
    else:
        # English: split on period but not on decimals/abbreviations
        sentences = re.split(r"(?<!\d)\.(?!\d)", text)

    return [s.strip() for s in sentences if s.strip()]


def rejoin_sentences(sentences: list, tgt_lang: str) -> str:
    """
    tgt_lang: "hindi" or "english"  (the OUTPUT language after translation)

    Hindi output  → rejoin with danda (।)
    English output → rejoin with period (.)
    """
    if tgt_lang == "hindi":
        return (" " + DANDA + " ").join(sentences) + " " + DANDA
    else:
        return ". ".join(sentences) + "."


# ──────────────────────────────────────────────────────
# MODEL INIT
# ──────────────────────────────────────────────────────
def initialize_model(ckpt_dir):
    tokenizer = AutoTokenizer.from_pretrained(ckpt_dir, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        ckpt_dir,
        trust_remote_code=True,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    ).to(DEVICE)
    model.eval()
    return tokenizer, model


# ──────────────────────────────────────────────────────
# TRANSLATION HELPER
# ──────────────────────────────────────────────────────
def batch_translate(sentences, src_lang, tgt_lang, model, tokenizer, ip):
    inputs = ip.preprocess_batch(sentences, src_lang=src_lang, tgt_lang=tgt_lang)
    model_inputs = tokenizer(
        inputs, return_tensors="pt", padding=True,
        truncation=True, max_length=256
    ).to(model.device)
    with torch.no_grad():
        translated_tokens = model.generate(
            **model_inputs, use_cache=False, max_length=256
        )
    outputs = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)
    torch.cuda.empty_cache()
    return ip.postprocess_batch(outputs, lang=tgt_lang)


# ──────────────────────────────────────────────────────
# DOC LOADING
# ──────────────────────────────────────────────────────
def load_mono_docs(lang_code: str, n: int) -> list:
    print(f"  [LOAD] IndicMonoDoc lang={lang_code}, n={n}")
    mono = load_dataset(
        "cfilt/IITB-IndicMonoDoc", split="test",
        trust_remote_code=True, streaming=True
    )
    docs = []
    for item in mono:
        if item.get("lang") != lang_code:
            continue
        text = item.get("text", "")
        if isinstance(text, bytes):
            text = text.decode("utf-8", errors="ignore")
        text = text.strip()
        if text:
            docs.append(text)
        if len(docs) >= n:
            break
    print(f"  ✓ Loaded {len(docs)} docs")
    return docs


# ──────────────────────────────────────────────────────
# DOC-LEVEL BACKTRANSLATION
# ──────────────────────────────────────────────────────
def backtranslate_docs(docs, src_lang_indic, tgt_lang_indic,
                       src_lang, tgt_lang,
                       model, tokenizer, ip,
                       save_prefix: str) -> list:
    """
    src_lang / tgt_lang: "hindi" or "english"  (for split/rejoin)
    src_lang_indic / tgt_lang_indic: IndicTrans lang tags
    """
    translated_docs = []

    for doc_idx, doc in enumerate(docs, 1):
        # ── SPLIT with language-aware delimiter ──────────────
        sents = split_sentences(doc, lang=src_lang)
        if not sents:
            translated_docs.append("")
            continue

        # ── TRANSLATE sentence by sentence in batches ────────
        translations = []
        for i in range(0, len(sents), BATCH_SIZE):
            batch = sents[i : i + BATCH_SIZE]
            translations.extend(
                batch_translate(batch, src_lang_indic, tgt_lang_indic,
                                model, tokenizer, ip)
            )

        # ── REJOIN with target language delimiter ─────────────
        merged = rejoin_sentences(translations, tgt_lang=tgt_lang)
        translated_docs.append(merged)

        # Save aligned file for inspection
        aligned_path = os.path.join(
            OUTPUT_DIR, f"{save_prefix}_doc{doc_idx}_aligned.txt"
        )
        with open(aligned_path, "w", encoding="utf-8") as f:
            for s, t in zip(sents, translations):
                f.write(f"SRC: {s}\nTGT: {t}\n\n")

        if doc_idx % 500 == 0:
            print(f"    Translated {doc_idx}/{len(docs)} docs")

    print(f"  ✓ {len(translated_docs)} docs translated")
    return translated_docs


# ──────────────────────────────────────────────────────
# READ LOCAL PRALEKHA JSONL
# ──────────────────────────────────────────────────────
def read_jsonl(path: str) -> list:
    docs = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                arr = json.loads(line)
                if isinstance(arr, list) and arr:
                    docs.append(arr[0])
                elif isinstance(arr, str):
                    docs.append(arr)
            except Exception:
                continue
    return docs


# ══════════════════════════════════════════════════════
# STEP 1: HI → EN
#   Split Hindi on danda (।)
#   Translate each sentence HI→EN
#   Rejoin with period (.)
#   Pair: (synthetic_EN, original_HI)
# ══════════════════════════════════════════════════════
print("\n[STEP 1] HI→EN backtranslation")
hi_docs = load_mono_docs("hi", N_DOCS)

print("  Loading Indic→EN model...")
tok_ie, mdl_ie = initialize_model(INDIC_TO_EN_CKPT)
ip = IndicProcessor(inference=True)

synthetic_en_docs = backtranslate_docs(
    docs=hi_docs,
    src_lang_indic="hin_Deva",
    tgt_lang_indic="eng_Latn",
    src_lang="hindi",           # split on danda (।)
    tgt_lang="english",         # rejoin with period (.)
    model=mdl_ie, tokenizer=tok_ie, ip=ip,
    save_prefix="hi2en",
)
del mdl_ie, tok_ie
torch.cuda.empty_cache(); gc.collect()
print("  Cleaned up Indic→EN model")


# ══════════════════════════════════════════════════════
# STEP 2: EN → HI
#   Split English on period (.)
#   Translate each sentence EN→HI
#   Rejoin with danda (।)
#   Pair: (original_EN, synthetic_HI)
# ══════════════════════════════════════════════════════
print("\n[STEP 2] EN→HI backtranslation")
en_docs = load_mono_docs("en", N_DOCS)

print("  Loading EN→Indic model...")
tok_ei, mdl_ei = initialize_model(EN_TO_INDIC_CKPT)

synthetic_hi_docs = backtranslate_docs(
    docs=en_docs,
    src_lang_indic="eng_Latn",
    tgt_lang_indic="hin_Deva",
    src_lang="english",         # split on period (.)
    tgt_lang="hindi",           # rejoin with danda (।)
    model=mdl_ei, tokenizer=tok_ei, ip=ip,
    save_prefix="en2hi",
)
del mdl_ei, tok_ei, ip
torch.cuda.empty_cache(); gc.collect()
print("  Cleaned up EN→Indic model")


# ══════════════════════════════════════════════════════
# STEP 3: Read local Pralekha JSONL
# ══════════════════════════════════════════════════════
print("\n[STEP 3] Reading local Pralekha JSONL")
pralekha_en = read_jsonl(os.path.join(TRAIN_DIR, "doc.eng.jsonl"))
pralekha_hi = read_jsonl(os.path.join(TRAIN_DIR, "doc.hin.jsonl"))

if len(pralekha_en) != len(pralekha_hi):
    raise ValueError(
        f"Pralekha mismatch: {len(pralekha_en)} EN vs {len(pralekha_hi)} HI"
    )
print(f"  ✓ Pralekha: {len(pralekha_en)} pairs")


# ══════════════════════════════════════════════════════
# STEP 4: Combine all pairs
#
#   (original_EN,   original_HI)  ← Pralekha genuine parallel
#   (synthetic_EN,  original_HI)  ← HI→EN BT augmentation
#   (original_EN,   synthetic_HI) ← EN→HI BT augmentation
# ══════════════════════════════════════════════════════
print("\n[STEP 4] Combining datasets")

all_en = pralekha_en + synthetic_en_docs + en_docs
all_hi = pralekha_hi + hi_docs           + synthetic_hi_docs

assert len(all_en) == len(all_hi), "Length mismatch after combining!"

indices = list(range(len(all_en)))
random.seed(42)
random.shuffle(indices)

shuffled_en = [all_en[i] for i in indices]
shuffled_hi = [all_hi[i] for i in indices]

print(f"  Total pairs       : {len(shuffled_en)}")
print(f"    Pralekha genuine: {len(pralekha_en)}")
print(f"    HI→EN synthetic : {len(synthetic_en_docs)}")
print(f"    EN→HI synthetic : {len(en_docs)}")


# ══════════════════════════════════════════════════════
# STEP 5: Write combined JSONL
# ══════════════════════════════════════════════════════
print("\n[STEP 5] Writing combined JSONL")

combined_eng_jsonl = os.path.join(TRAIN_DIR, "doc.eng.both.jsonl")
with open(combined_eng_jsonl, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(shuffled_en, 1):
        f.write(json.dumps([doc.strip()], ensure_ascii=False) + "\n")
        if idx % 1000 == 0:
            print(f"  Written {idx} EN docs...")
print(f"  ✓ Saved: {combined_eng_jsonl}")

combined_hin_jsonl = os.path.join(TRAIN_DIR, "doc.hin.both.jsonl")
with open(combined_hin_jsonl, "w", encoding="utf-8") as f:
    for idx, doc in enumerate(shuffled_hi, 1):
        f.write(json.dumps([doc.strip()], ensure_ascii=False) + "\n")
        if idx % 1000 == 0:
            print(f"  Written {idx} HI docs...")
print(f"  ✓ Saved: {combined_hin_jsonl}")

print("\n✅ Variant 2 BT data generation complete.")
print(f"   Use doc.eng.both.jsonl + doc.hin.both.jsonl for fine-tuning.")
