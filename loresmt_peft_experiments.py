# loresmt_hybrid_winner.py
# ======================================================
# LoResMT 2025 — Hybrid Winner Script (A + B + C merged)
# - LoRA + Adapters + Hybrid FT
# - HF chat-template prompt (your exact template)
# - Deterministic, T4-friendly defaults (LIGHT_MODE)
# - Streaming dataset, backtranslation hooks, sweep driver
# - Produces competition-ready JSONL files + submission.zip
# ======================================================

import os
import json
import zipfile
import math
import time
import warnings
import random
from pathlib import Path
from itertools import islice

import torch
import numpy as np
from torch.utils.data import IterableDataset

from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM

# PEFT imports
from peft import LoraConfig, get_peft_model
try:
    from peft import AdapterConfig
    PEFT_HAS_ADAPTER = True
except Exception:
    AdapterConfig = None
    PEFT_HAS_ADAPTER = False

from trl import SFTTrainer, SFTConfig

import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, Markdown

# -------------------------
# Basic config (tweak)
# -------------------------
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-270m-it")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "./loresmt_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 2000
EVAL_BATCH_SIZE = 8

FULL_DATASET = True
MAX_SAMPLES = None

NUM_WORKERS = 2

# dtype preferences
FORCE_FP16 = True
FORCE_BF16 = False

HF_AUTH_TOKEN_ENV = "HF_HUB_TOKEN"

# seeds & determinism
SEED = 42
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# langs
INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# -------------------------
# Competition-ready flags (LIGHT_MODE default)
# -------------------------
LIGHT_MODE = True
if LIGHT_MODE:
    MAX_TRAIN_STEPS = 500
    HYBRID_STEPS = 200
    if MAX_SAMPLES is None:
        MAX_SAMPLES = 15000

# -------------------------
# Research flags (safe defaults)
# -------------------------
ENABLE_LORA = True
ENABLE_ADAPTER = False
ADAPTER_REDUCTION = 16

LORA_R = 16
LORA_ALPHA = 16
LORA_DROPOUT = 0.05

ENABLE_HYBRID_FT = False
HYBRID_UNFREEZE_TOPK = 2
HYBRID_STEPS = HYBRID_STEPS if 'HYBRID_STEPS' in globals() else 500
HYBRID_LR = 5e-6

ENABLE_BT = False
BT_JSONL_PATHS = []

# Sweep config
RUN_SWEEP = False
SWEEP_LORA_R = [8,16,32]
SWEEP_USE_ADAPTER = [False, True]
SWEEP_HYBRID = [False, True]

EXPERIMENT_LOG = OUTPUT_DIR/"experiments.jsonl"

# ------------------------------
# PROMPT BUILDER (HF chat template EXACTLY as provided)
# ------------------------------
def build_prompt(src, src_lang, tgt_lang, example, tokenizer=None):
    ex_src, ex_tgt = example
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"},
            {"role":"assistant","content":ex_tgt},
            {"role":"user","content":f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"},
            {"role":"assistant","content":""}
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        return (f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n"
                f"{ex_src} → {ex_tgt}\n\n"
                f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}")

# ------------------------------
# STREAMING DATA + optional BT
# ------------------------------
def stream_examples(tokenizer, max_samples=None, bt_files=None):
    dataset_name = "ai4bharat/Pralekha"
    cfg = "train"
    splits = get_dataset_split_names(dataset_name, cfg)
    if not splits:
        raise RuntimeError("Could not get splits for ai4bharat/Pralekha (check datasets & internet).")

    # first yield parallel data
    for split in splits:
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]: continue
        lang = tl if sl == "eng" else sl
        if lang not in INDIAN_LANGS: continue

        # get one-shot example
        ds_temp = load_dataset(dataset_name, split=split, streaming=True, name=cfg)
        one_shot = ("","")
        try:
            for row in islice(ds_temp, 50):
                s, t = row.get("src_txt",""), row.get("tgt_txt","")
                if s and t and len(s.split())>5 and len(t.split())>5:
                    one_shot = (s,t)
                    break
        except Exception:
            one_shot = ("","")

        ds = load_dataset(dataset_name, split=split, streaming=True, name=cfg)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s, t) if sl=="eng" else (t, s)
            # yield both directions
            yield {"input_text": build_prompt(eng,"eng",lang, one_shot, tokenizer),
                   "target_text": indic, "direction": f"eng_{lang}"}
            yield {"input_text": build_prompt(indic,lang,"eng", one_shot, tokenizer),
                   "target_text": eng, "direction": f"{lang}_eng"}
            count += 1

    # then optionally yield backtranslated synthetic pairs
    if bt_files:
        for path in bt_files:
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for ln in f:
                        if not ln.strip(): continue
                        obj = json.loads(ln)
                        lang = obj.get("lang")
                        if not lang: continue
                        yield {"input_text": build_prompt(obj["src"], "eng", lang, ("Example","Example"), tokenizer),
                               "target_text": obj["tgt"], "direction": f"eng_{lang}"}
            except Exception as e:
                print("Warning loading BT file", path, e)

# ------------------------------
# Iterable dataset
# ------------------------------
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None, bt_files=None):
        self.tok = tokenizer
        self.max_samples = max_samples
        self.bt_files = bt_files

    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples, self.bt_files):
            src_enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
            tgt_enc = self.tok(ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)

            inp = src_enc["input_ids"] + tgt_enc["input_ids"]
            inp = inp[:MAX_SEQ_LEN]

            labels = ([-100]*len(src_enc["input_ids"]) + tgt_enc["input_ids"])[:MAX_SEQ_LEN]

            yield {"input_ids": inp, "attention_mask": [1]*len(inp), "labels": labels}

# ------------------------------
# LoRA target detection (expanded)
# ------------------------------
def detect_lora_modules(model):
    modules = []
    for n, m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","v_proj","o_proj",
                                      "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo",
                                      "wi","wo","fc1","fc2","w1","w2"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

# ------------------------------
# Param counter
# ------------------------------
def count_params(model):
    try:
        base = getattr(model, "base_model", None) or getattr(model, "model", None) or model
    except Exception:
        base = model
    total = sum(p.numel() for p in base.parameters())
    tuned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * tuned / total if total>0 else 0.0
    return total, tuned, pct

# ------------------------------
# Prepare model with PEFT options
# ------------------------------
def prepare_model(lora_r=16, lora_alpha=16, lora_dropout=0.05,
                  enable_lora=True, enable_adapter=False, adapter_reduction=16):
    hf_token = os.environ.get(HF_AUTH_TOKEN_ENV)
    if hf_token:
        os.environ["HF_HUB_TOKEN"] = hf_token

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dtype selection
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0).lower()
        if FORCE_BF16:
            dtype = torch.bfloat16
        elif FORCE_FP16:
            dtype = torch.float16
        else:
            dtype = torch.bfloat16 if any(x in gpu_name for x in ["a100","h100","a6000"]) else torch.float16
    else:
        dtype = torch.float32

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME} with dtype={dtype} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)

    peft_applied = False
    peft_name = "none"

    # adapters
    if enable_adapter and PEFT_HAS_ADAPTER:
        try:
            adapter_cfg = AdapterConfig(adapter_type="houlsby", reduction_factor=adapter_reduction, non_linearity="relu")
            model = get_peft_model(model, adapter_cfg)
            peft_applied = True
            peft_name = "adapter"
        except Exception as e:
            print("Adapter application failed (falling back to LoRA):", e)
            enable_adapter = False

    # LoRA
    if enable_lora and not peft_applied:
        target_modules = detect_lora_modules(model)
        print("Detected LoRA target modules (pre-filter):", target_modules)
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha,
                              target_modules=target_modules,
                              lora_dropout=lora_dropout,
                              bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        peft_applied = True
        peft_name = f"lora_r{lora_r}"

    model.to(device)
    total, tuned, pct = count_params(model)
    print(f"[PARAMS] total={total:,}, tuned={tuned:,}, tuned_pct={pct:.4f}%  (PEFT={peft_name})")
    with open(OUTPUT_DIR/"param_stats.json","w") as fh:
        json.dump({"total":total, "tuned":tuned, "pct":pct, "peft":peft_name}, fh, indent=2)
    return model, tok

# ------------------------------
# Hybrid: unfreeze top-k decoder layers
# ------------------------------
def unfreeze_topk_decoder_layers(model, k=2):
    base = getattr(model, "base_model", None) or getattr(model, "model", None) or model
    dec_layers = None
    for attr in ["decoder", "model.decoder", "transformer.decoder", "transformer.h", "transformer.layers"]:
        try:
            obj = base
            for part in attr.split("."):
                obj = getattr(obj, part)
            if hasattr(obj, "__len__"):
                dec_layers = obj
                break
        except Exception:
            continue
    if dec_layers is None:
        print("Could not locate decoder layers automatically for hybrid FT; skipping unfreeze.")
        return 0
    n = len(dec_layers)
    to_unfreeze = list(range(max(0, n-k), n))
    cnt = 0
    for idx in to_unfreeze:
        layer = dec_layers[idx]
        for p in layer.parameters():
            if not p.requires_grad:
                p.requires_grad = True
                cnt += p.numel()
    print(f"Unfrozen ~{cnt} params in top {k} decoder layers (indices {to_unfreeze})")
    return cnt

# ------------------------------
# Training wrapper (single run)
# ------------------------------
def train_run(lora_r=16, enable_lora=True, enable_adapter=False, adapter_reduction=16,
              hybrid_ft=False, hybrid_steps=500, hybrid_lr=5e-6, max_samples=None, bt_files=None):
    model, tok = prepare_model(lora_r, LORA_ALPHA, LORA_DROPOUT, enable_lora, enable_adapter, adapter_reduction)
    ds = PralekhaDataset(tok, max_samples, bt_files)

    # dtype detection
    p = next(model.parameters())
    use_fp16 = (p.dtype==torch.float16)
    use_bf16 = (p.dtype==torch.bfloat16)

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=20,
        save_strategy="steps",
        save_steps=300,
        save_total_limit=2,
        load_best_model_at_end=False,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none"
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    print("Starting training...")
    trainer.train()
    print("Training finished; saving model & tokenizer...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # evaluation
    bleu, chrf = evaluate_model(trainer.model, tok, max_samples_per_split=(200 if not FULL_DATASET else None))
    avg_bleu = sum(bleu.values())/len(bleu) if bleu else 0.0

    # hybrid selective FT
    if hybrid_ft:
        print("Starting hybrid selective FT stage...")
        unfreeze_topk_decoder_layers(trainer.model, k=HYBRID_UNFREEZE_TOPK)
        p = next(trainer.model.parameters())
        use_fp16 = (p.dtype==torch.float16)
        use_bf16 = (p.dtype==torch.bfloat16)
        cfg2 = SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=PER_DEVICE_BATCH,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=hybrid_lr,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            max_steps=hybrid_steps,
            logging_steps=20,
            save_strategy="steps",
            save_steps=200,
            save_total_limit=2,
            load_best_model_at_end=False,
            fp16=use_fp16, bf16=use_bf16,
            dataloader_num_workers=NUM_WORKERS,
            report_to="none"
        )
        trainer2 = SFTTrainer(model=trainer.model, args=cfg2, train_dataset=ds, tokenizer=tok)
        trainer2.train()
        trainer2.model.save_pretrained(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        bleu, chrf = evaluate_model(trainer2.model, tok, max_samples_per_split=(200 if not FULL_DATASET else None))
        avg_bleu = sum(bleu.values())/len(bleu) if bleu else 0.0

    # log result
    entry = {
        "timestamp": time.time(),
        "model": MODEL_NAME,
        "lora_r": lora_r,
        "enable_lora": enable_lora,
        "enable_adapter": enable_adapter,
        "adapter_reduction": adapter_reduction,
        "hybrid_ft": hybrid_ft,
        "hybrid_steps": hybrid_steps,
        "hybrid_lr": hybrid_lr,
        "max_samples": max_samples,
        "bt_files": bt_files,
        "avg_bleu": avg_bleu,
        "bleu": bleu,
        "chrf": chrf
    }
    with open(EXPERIMENT_LOG, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return entry

# ------------------------------
# Evaluation & submission formatter
# ------------------------------
def evaluate_model(model, tok, max_samples_per_split=None, max_new_tokens=256, batch_size=EVAL_BATCH_SIZE):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        preds[f"eng_{lang}"] = []; refs[f"eng_{lang}"] = []
        preds[f"{lang}_eng"] = []; refs[f"{lang}_eng"] = []

    splits = get_dataset_split_names("ai4bharat/Pralekha", "dev")
    for split in tqdm(splits, desc="Evaluating"):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        count = 0
        prompts = []; references = []; directions = []
        for row in ds:
            if max_samples_per_split and count>=max_samples_per_split: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            prompts.append(build_prompt(eng,"eng",lang,("Example","Example"), tok)); references.append(indic); directions.append(f"eng_{lang}")
            prompts.append(build_prompt(indic,lang,"eng",("Example","Example"), tok)); references.append(eng); directions.append(f"{lang}_eng")
            count += 1
            if len(prompts) >= batch_size:
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    outs = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
                decs = tok.batch_decode(outs, skip_special_tokens=True)
                for d, p, r in zip(directions, decs, references):
                    preds[d].append(p.strip()); refs[d].append(r.strip())
                prompts, references, directions = [], [], []
        if prompts:
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
            with torch.no_grad():
                outs = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            decs = tok.batch_decode(outs, skip_special_tokens=True)
            for d, p, r in zip(directions, decs, references):
                preds[d].append(p.strip()); refs[d].append(r.strip())

    # create LoResMT-compliant JSONL files (one file per direction; hyphenated)
    for d in preds:
        arr = preds[d]
        fname = OUTPUT_DIR / f"{d.replace('_','-')}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for p in arr:
                f.write(json.dumps([p], ensure_ascii=False) + "\n")

    # zip (optional)
    sub_zip = OUTPUT_DIR/"submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for d in preds:
            fname = OUTPUT_DIR / f"{d.replace('_','-')}.jsonl"
            if fname.exists():
                zf.write(fname, fname.name)
    print("Saved submission files and ZIP ->", sub_zip)

    # compute sacreBLEU / chrF per direction
    bleu = {}; chrf = {}
    for d in preds:
        if len(preds[d])==0:
            bleu[d]=0; chrf[d]=0; continue
        try:
            bleu[d] = sacrebleu.corpus_bleu(preds[d], [refs[d]]).score
            chrf[d] = sacrebleu.corpus_chrf(preds[d], [refs[d]]).score
        except Exception as e:
            print("Warning computing metrics for", d, e)
            bleu[d]=0; chrf[d]=0

    # save sacreBLEU signature
    signature_path = OUTPUT_DIR/"sacrebleu_signature.txt"
    with open(signature_path, "w") as f:
        f.write("EVAL SIGNATURE:\n")
        f.write(str(sacrebleu.corpus_bleu(["a"], [["a"]]).signature))
    print("Saved sacreBLEU signature ->", signature_path)

    return bleu, chrf

# ------------------------------
# Backtranslation helper (deterministic)
# ------------------------------
def backtranslate(model, tokenizer, input_sentences, src_lang, tgt_lang,
                  batch_size=8, max_new_tokens=128, output_file="bt_synthetic.jsonl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out_pairs = []
    torch.manual_seed(SEED)
    for i in tqdm(range(0, len(input_sentences), batch_size), desc="BT"):
        batch = input_sentences[i:i+batch_size]
        prompts = [build_prompt(s, src_lang, tgt_lang, ("Example","Example"), tokenizer) for s in batch]
        enc = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
        with torch.no_grad():
            outs = model.generate(**enc, max_new_tokens=max_new_tokens, do_sample=False, pad_token_id=tokenizer.pad_token_id)
        decs = tokenizer.batch_decode(outs, skip_special_tokens=True)
        for s_dec, orig in zip(decs, batch):
            out_pairs.append({"src": s_dec.strip(), "tgt": orig.strip(), "lang": tgt_lang})
    with open(output_file, "w", encoding="utf-8") as fh:
        for o in out_pairs:
            fh.write(json.dumps(o, ensure_ascii=False) + "\n")
    print(f"Saved BT pairs -> {output_file} (count={len(out_pairs)})")
    return output_file

# ------------------------------
# Sweep driver
# ------------------------------
def run_sweep(lora_r_list=[8,16,32], use_adapter_list=[False], hybrid_list=[False], max_samples=None, bt_files=None):
    results = []
    for r in lora_r_list:
        for use_ad in use_adapter_list:
            if use_ad and not PEFT_HAS_ADAPTER:
                print("Skipping adapter run since AdapterConfig not available in peft.")
                continue
            for hy in hybrid_list:
                print(f"\n=== RUN CONFIG: r={r}, adapter={use_ad}, hybrid={hy} ===")
                entry = train_run(lora_r=r, enable_lora=True, enable_adapter=use_ad,
                                  adapter_reduction=ADAPTER_REDUCTION, hybrid_ft=hy,
                                  hybrid_steps=HYBRID_STEPS, hybrid_lr=HYBRID_LR,
                                  max_samples=max_samples, bt_files=bt_files)
                results.append(entry)
    return results

# ------------------------------
# Reporting helper
# ------------------------------
def summarize_experiments(logfile=EXPERIMENT_LOG):
    rows = []
    if not logfile.exists(): return rows
    with open(logfile, "r", encoding="utf-8") as fh:
        for ln in fh:
            rows.append(json.loads(ln))
    if not rows:
        return []
    df = pd.DataFrame(rows)
    display(Markdown("## Experiment Summary"))
    display(df[["timestamp","lora_r","enable_adapter","hybrid_ft","avg_bleu"]].sort_values("avg_bleu", ascending=False).head(10))
    return df

# ------------------------------
# Training plot
# ------------------------------
def plot_training(trainer):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    if not steps:
        print("No training logs found to plot.")
        return
    plt.figure(figsize=(8,4))
    plt.plot(steps, losses)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training Loss Trend")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png")
    plt.close()
    print("Saved training loss curve.")

# ------------------------------
# MAIN
# ------------------------------
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--run_mode", choices=["single","sweep"], default="single")
    parser.add_argument("--lora_r", type=int, default=LORA_R)
    parser.add_argument("--enable_adapter", action="store_true")
    parser.add_argument("--enable_hybrid", action="store_true")
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--bt_files", nargs="*", default=BT_JSONL_PATHS)
    parser.add_argument("--sweep_rs", nargs="*", type=int, default=SWEEP_LORA_R)
    parser.add_argument("--sweep_adapters", nargs="*", type=lambda x: x.lower() in ("1","true","yes"), default=SWEEP_USE_ADAPTER)
    parser.add_argument("--sweep_hybrid", nargs="*", type=lambda x: x.lower() in ("1","true","yes"), default=SWEEP_HYBRID)
    args = parser.parse_args()

    # safety max samples selection
    safe_max = args.max_samples or (MAX_SAMPLES if MAX_SAMPLES is not None else (2000 if not FULL_DATASET else None))

    if args.run_mode == "single":
        print("Running single experiment...")
        entry = train_run(lora_r=args.lora_r, enable_lora=True, enable_adapter=args.enable_adapter,
                          adapter_reduction=ADAPTER_REDUCTION, hybrid_ft=args.enable_hybrid,
                          hybrid_steps=HYBRID_STEPS, hybrid_lr=HYBRID_LR,
                          max_samples=safe_max, bt_files=args.bt_files)
        print("Experiment finished. Entry saved to log.")
        summarize_experiments()
    else:
        print("Running sweep...")
        results = run_sweep(lora_r_list=args.sweep_rs, use_adapter_list=args.sweep_adapters,
                            hybrid_list=args.sweep_hybrid, max_samples=safe_max, bt_files=args.bt_files)
        print("Sweep finished. Summary:")
        summarize_experiments()

    print("ALL DONE. Artifacts in:", OUTPUT_DIR)
