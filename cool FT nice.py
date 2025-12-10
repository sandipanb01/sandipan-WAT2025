# ======================================================
# 🚀 UNIVERSAL SFT (Production-ready, VS Code / Colab)
# ======================================================
import os, json, zipfile, math, warnings, gc
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
import pandas as pd
from IPython.display import display, Markdown

# ------------------------------ CONFIG (merged defaults)
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("./universal_output_fixed")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

# Sequence / batching
MAX_SEQ_LEN = 384         # kept smaller for production-style setting
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 100     # used by SFTConfig if you want step-limited runs
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 100

# Quick test toggles (from production)
QUICK_TEST = True
if QUICK_TEST:
    MAX_COLAB_SAMPLES = 50
    EVAL_SAMPLES = 10
    print("🧪 QUICK TEST MODE ENABLED")
else:
    EVAL_SAMPLES = None

# Beam control (keeps earlier BEAM switch ideas)
BEAM_MODE = "A"
if BEAM_MODE == "A":
    BEAM_KWARGS = dict(num_beams=5, early_stopping=True)
else:
    BEAM_KWARGS = dict(num_beams=5, length_penalty=1.0)

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ P3 PROMPT TEMPLATE (chosen)
# P3 format (chat-turn markers). We'll use explicit markers and end with assistant start.
# Template (training): "<sot>user\nTranslate to {tgt_lang}:\n{src}\n<eot>\n<sot>model\n"
# During eval we'll use the same prompt (only the model must generate translation).
P3_USER = "<sot>user\nTranslate to {tgt_lang}:\n{src}\n<eot>\n<sot>model\n"

# Helper to build a P3-style prompt (string)
def build_p3_prompt(src, src_lang, tgt_lang):
    # src_lang unused in the literal template but kept for clarity
    return P3_USER.format(tgt_lang=LANG_MAP[tgt_lang].lower() if tgt_lang in LANG_MAP else tgt_lang, src=src)

# Utility to extract model answer given full decoded string and prompt (strip markers)
def extract_answer(full_output, prompt):
    if not full_output:
        return ""
    try:
        if prompt and prompt in full_output:
            return full_output.split(prompt,1)[1].strip()
    except Exception:
        pass
    # fallback markers
    markers = ["<eot>", "Translation:", "Translation —", "Output:", "Answer:", "\n"]
    for m in markers:
        if m in full_output:
            return full_output.split(m,1)[1].strip()
    return full_output.strip()

# ------------------------------ STREAMING data from Pralekha
def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    for split in splits:
        parts = split.split("_")
        if len(parts) != 2:
            continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]:
            continue
        lang = tl if sl == "eng" else sl
        if lang not in INDIAN_LANGS:
            continue

        # make a one-shot example (first reasonable pair) to optionally include in prompt if desired
        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        for row in islice(ds, 50):
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5:
                one_shot = (s,t)
                break

        # actual streaming pass
        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t:
                continue
            # basic quality filters (mirror production)
            if len(s.split()) < 5 or len(t.split()) < 5:
                continue
            if s[:100] == t[:100]:
                continue

            eng, indic = (s,t) if sl == "eng" else (t,s)
            # produce both directions
            # eng -> lang
            yield {"src": eng.strip(), "tgt": indic.strip(), "dirn": f"eng_{lang}"}
            # lang -> eng
            yield {"src": indic.strip(), "tgt": eng.strip(), "dirn": f"{lang}_eng"}
            count += 1

# ------------------------------ ITERABLE WRAPPER (SFT dataset)
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tok = tokenizer
        self.max_samples = max_samples
        # choose eos id fallback to sep token if eos not present
        self.eos_id = tokenizer.eos_token_id if tokenizer.eos_token_id is not None else tokenizer.sep_token_id

    def __iter__(self):
        for ex in stream_examples(self.tok, max_samples=self.max_samples):
            src = ex["src"]
            tgt = ex["tgt"]
            dirn = ex["dirn"]  # like "eng_hin" or "hin_eng"
            parts = dirn.split("_")
            sl, tl = parts[0], parts[1]
            # build P3 prompt (we want the model to generate only the translation after the assistant marker)
            prompt = P3_USER.format(tgt_lang=tl, src=src)
            # Tokenize prompt and target separately (no special extra tokens added in middle)
            s_enc = self.tok(prompt, truncation=True, max_length=MAX_SEQ_LEN//2, add_special_tokens=False)
            t_enc = self.tok(" " + tgt, truncation=True, max_length=MAX_SEQ_LEN//2, add_special_tokens=False)
            input_ids = s_enc["input_ids"] + t_enc["input_ids"] + ([self.eos_id] if self.eos_id is not None else [])
            input_ids = input_ids[:MAX_SEQ_LEN]
            src_len = len(s_enc["input_ids"])
            # labels: mask prompt portion with -100, leave target token ids as labels (explicit EOS included)
            labels = [-100]*src_len + t_enc["input_ids"]
            # append eos label if we have eos in inputs and space
            if self.eos_id is not None and len(labels) < len(input_ids):
                # ensure lengths match
                if len(labels) < len(input_ids):
                    labels = labels[:len(input_ids)]
            labels = labels[:MAX_SEQ_LEN]
            attention_mask = [1]*len(input_ids)
            yield {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

# ------------------------------ MODEL PREP (your safe loader + LoRA r=8,alpha=16)
def detect_lora_modules(model):
    modules = []
    for n,m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj",
                                      "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # Safe load (no device_map auto / no meta tensors)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float32,
        low_cpu_mem_usage=True,
        device_map=None
    )

    # Move to GPU (expects GPU available)
    if torch.cuda.is_available():
        model = model.to("cuda")
        torch.cuda.empty_cache()

    # LoRA setup (r=8,lora_alpha=16)
    target_modules = detect_lora_modules(model)
    if not target_modules:
        target_modules = ["q_proj","k_proj","gate_proj","v_proj","o_proj",
                          "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]
    print(f"⚡ LoRA target modules detected/used: {target_modules}")

    lora_cfg = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    return model, tok

# ------------------------------ TRAINING (SFTTrainer-based)
def train_model(max_samples=None):
    model, tok = prepare_model()
    ds = PralekhaDataset(tok, max_samples=max_samples)

    # SFT config (keeps your earlier hyper choices)
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1
    )

    print("\n🚀 Starting SFT training...")
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()

    print("\n💾 Saving model + tokenizer...")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION (ENG<->HIN only)
# ======================================================
# 🔧 PATCHED EVALUATION (CLEAN, SIMPLE, EXACTLY LIKE YOUR FIRST SCRIPT)
# ======================================================

def evaluate_model(model, tok, eval_data):
    """Evaluate and save results to JSONL files (eng_hin + hin_eng)."""
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    
    preds = {"eng_hin": [], "hin_eng": []}
    refs  = {"eng_hin": [], "hin_eng": []}
    inputs = {"eng_hin": [], "hin_eng": []}

    print("\n" + "="*80)
    print("📊 EVALUATION (patched P3-compatible)")
    print("="*80)

    for ex in tqdm(eval_data, desc="Evaluating"):
        src = ex["src"]
        tgt = ex["tgt"]
        dirn = ex["dirn"]   # "eng_hin" or "hin_eng"

        # Build evaluation prompt (matches your clean first-version style)
        if dirn == "eng_hin":
            prompt = f"""Task: Translate English to Hindi.
IMPORTANT: Output ONLY Hindi translation.

English text:
{src}

Hindi translation:"""
        else:
            prompt = f"""Task: Translate Hindi to English.
IMPORTANT: Output ONLY English translation.

Hindi text:
{src}

English translation:"""

        # Tokenize
        enc = tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_SEQ_LEN//2
        ).to(device)

        input_len = enc["input_ids"].shape[1]

        # Generate
        with torch.no_grad():
            out = model.generate(
                **enc,
                max_new_tokens=512,
                do_sample=False,
                num_beams=1,
                pad_token_id=tok.pad_token_id,
                eos_token_id=tok.eos_token_id
            )

        # Extract prediction after prompt tokens
        gen_ids = out[0, input_len:]
        pred = tok.decode(gen_ids, skip_special_tokens=True).strip()

        # Clean excessive newlines
        pred = "\n".join([l.strip() for l in pred.split("\n") if l.strip()])

        preds[dirn].append(pred)
        refs[dirn].append(tgt.strip())
        inputs[dirn].append(src.strip())

    save_results(preds, refs, inputs)
    bleu_scores, chrf_scores = calculate_metrics(preds, refs)
    return bleu_scores, chrf_scores


# ======================================================
# 🔧 Save JSONL + ZIP (clean, simple)
# ======================================================

def save_results(preds, refs, inputs):
    print("\n💾 Saving results...")

    for direction in ["eng_hin", "hin_eng"]:
        if not preds[direction]:
            continue

        jsonl_file = OUTPUT_DIR / f"{direction}_pred_ref.jsonl"

        with open(jsonl_file, "w", encoding="utf-8") as f:
            for inp, pred, ref in zip(inputs[direction], preds[direction], refs[direction]):
                f.write(json.dumps({
                    "input_text": inp,
                    "prediction": pred,
                    "reference": ref
                }, ensure_ascii=False) + "\n")

        print(f"   ✅ Saved {jsonl_file.name}")

    # Create submission.zip
    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for direction in ["eng_hin", "hin_eng"]:
            fp = OUTPUT_DIR / f"{direction}_pred_ref.jsonl"
            if fp.exists():
                zf.write(fp, fp.name)

    print(f"   ✅ Created submission.zip")


# ======================================================
# 🔧 Metrics (BLEU + chrF)
# ======================================================

def calculate_metrics(preds, refs):
    bleu_scores, chrf_scores = {}, {}

    for direction in preds:
        if not preds[direction]:
            continue

        bleu_scores[direction] = sacrebleu.corpus_bleu(
            preds[direction],
            [refs[direction]]
        ).score

        chrf_scores[direction] = sacrebleu.corpus_chrf(
            preds[direction],
            [[r] for r in refs[direction]]
        ).score

    return bleu_scores, chrf_scores
