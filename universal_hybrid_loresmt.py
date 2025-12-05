
# ======================================================
# Universal Fine-tuning + Evaluation + Hybrid FT for Hugging Face causal/instruct LMs
# - Streaming Pralekha dataset (IterableDataset)
# - LoRA + optional Adapter + Hybrid selective fine-tuning (unfreeze top-k decoder layers)
# - HF chat-template prompt kept exactly as requested
# - Produces competition-ready JSONL files + submission.zip and metrics
# - Designed to be copy-pasted and run in a GPU environment with internet access
# ======================================================

import os
import json
import zipfile
import math
import time
import warnings
from pathlib import Path
from itertools import islice

# NOTE: this file is written as a ready-to-run script. When running, ensure you have:
# pip install transformers datasets peft trl sacrebleu evaluate tqdm matplotlib pandas

# ------------------------------ CONFIG (tweak to your environment)
MODEL_NAME = os.environ.get("MODEL_NAME", "google/gemma-3-270m-it")
OUTPUT_DIR = Path(os.environ.get("OUTPUT_DIR", "/content/universal_hybrid_output"))
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "1"))
GRAD_ACCUM = int(os.environ.get("GRAD_ACCUM", "4"))
MAX_TRAIN_STEPS = int(os.environ.get("MAX_TRAIN_STEPS", "3000"))
EVAL_BATCH_SIZE = int(os.environ.get("EVAL_BATCH_SIZE", "8"))

FULL_DATASET = True
MAX_COLAB_SAMPLES = None

SEED = int(os.environ.get("SEED", "42"))

# PEFT/hybrid defaults
ENABLE_LORA = True
LORA_R = int(os.environ.get("LORA_R", "16"))
LORA_ALPHA = int(os.environ.get("LORA_ALPHA", "16"))
LORA_DROPOUT = float(os.environ.get("LORA_DROPOUT", "0.05"))

ENABLE_ADAPTER = False
ADAPTER_REDUCTION = int(os.environ.get("ADAPTER_REDUCTION", "16"))

ENABLE_HYBRID = True
HYBRID_UNFREEZE_TOPK = int(os.environ.get("HYBRID_UNFREEZE_TOPK", "2"))
HYBRID_STEPS = int(os.environ.get("HYBRID_STEPS", "500"))
HYBRID_LR = float(os.environ.get("HYBRID_LR", "5e-6"))

FORCE_FP16 = True
FORCE_BF16 = False

HF_AUTH_TOKEN_ENV = "HF_HUB_TOKEN"

# langs
INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# reproducibility
import random, numpy as np
import torch
random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

# ------------------------------ PROMPT BUILDER (exact HF chat template)
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

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer, max_samples=None):
    from datasets import load_dataset, get_dataset_split_names
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    if not splits:
        raise RuntimeError("Could not get splits for ai4bharat/Pralekha. Check dataset availability.")

    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        try:
            for row in islice(ds, 50):
                s,t = row.get("src_txt",""), row.get("tgt_txt","")
                if s and t and len(s.split())>5 and len(t.split())>5:
                    one_shot = (s,t); break
        except Exception:
            one_shot = ("","")

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            # yield both directions as in LoResMT
            yield {"input_text": build_prompt(eng,"eng",lang, one_shot, tokenizer),
                   "target_text": indic, "direction": f"eng_{lang}"}
            yield {"input_text": build_prompt(indic,lang,"eng", one_shot, tokenizer),
                   "target_text": eng, "direction": f"{lang}_eng"}
            count += 1

# ------------------------------ Iterable dataset wrapper
from torch.utils.data import IterableDataset
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None, bt_files=None):
        self.tok = tokenizer
        self.max_samples = max_samples
        self.bt_files = bt_files

    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            src_enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
            tgt_enc = self.tok(ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)

            inp = src_enc["input_ids"] + tgt_enc["input_ids"]
            inp = inp[:MAX_SEQ_LEN]

            labels = ([-100]*len(src_enc["input_ids"]) + tgt_enc["input_ids"])[:MAX_SEQ_LEN]

            yield {"input_ids": inp, "attention_mask": [1]*len(inp), "labels": labels}

# ------------------------------ MODEL UTILITIES (PEFT friendly)
def detect_lora_modules(model):
    modules = []
    for n, m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","v_proj","o_proj",
                                      "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo",
                                      "wi","wo","fc1","fc2","w1","w2"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def count_params(model):
    try:
        base = getattr(model, "base_model", None) or getattr(model, "model", None) or model
    except Exception:
        base = model
    total = sum(p.numel() for p in base.parameters())
    tuned = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * tuned / total if total>0 else 0.0
    return total, tuned, pct

# ------------------------------ PREPARE MODEL (LoRA / Adapter options)
def prepare_model(model_name=MODEL_NAME,
                  lora_r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=LORA_DROPOUT,
                  enable_lora=True, enable_adapter=False, adapter_reduction=ADAPTER_REDUCTION):
    # defer heavy imports until runtime
    from transformers import AutoTokenizer, AutoModelForCausalLM
    from peft import LoraConfig, get_peft_model
    try:
        from peft import AdapterConfig
        has_adapter = True
    except Exception:
        AdapterConfig = None
        has_adapter = False

    hf_token = os.environ.get(HF_AUTH_TOKEN_ENV)
    if hf_token:
        os.environ["HF_HUB_TOKEN"] = hf_token

    tok = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # dtype selection heuristics
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
    print(f"Loading model {model_name} with dtype={dtype} on {device} ...")
    model = AutoModelForCausalLM.from_pretrained(model_name, trust_remote_code=True, torch_dtype=dtype, low_cpu_mem_usage=True)
    model.to(device)

    peft_applied = False
    peft_name = "none"

    # Try adapter first if requested
    if enable_adapter and has_adapter:
        try:
            adapter_cfg = AdapterConfig(adapter_type="houlsby", reduction_factor=adapter_reduction, non_linearity="relu")
            model = get_peft_model(model, adapter_cfg)
            peft_applied = True
            peft_name = "adapter"
        except Exception as e:
            print("Adapter application failed; falling back to LoRA. Error:", e)
            enable_adapter = False

    # Apply LoRA when requested or if adapter not applied
    if enable_lora and not peft_applied:
        target_modules = detect_lora_modules(model)
        print("Detected LoRA target modules:", target_modules)
        lora_cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, target_modules=target_modules,
                              lora_dropout=lora_dropout, bias="none", task_type="CAUSAL_LM")
        model = get_peft_model(model, lora_cfg)
        peft_applied = True
        peft_name = f"lora_r{lora_r}"

    model.to(device)
    total, tuned, pct = count_params(model)
    print(f"[PARAMS] total={total:,}, tuned={tuned:,}, tuned_pct={pct:.4f}%  (PEFT={peft_name})")
    with open(OUTPUT_DIR/"param_stats.json","w", encoding="utf-8") as fh:
        json.dump({"total":total, "tuned":tuned, "pct":pct, "peft":peft_name}, fh, indent=2)
    return model, tok

# ------------------------------ Hybrid helper: unfreeze top-k decoder layers
def unfreeze_topk_decoder_layers(model, k=2):
    base = getattr(model, "base_model", None) or getattr(model, "model", None) or model
    dec_layers = None
    # try common attribute names for decoder or transformer blocks
    for attr in ["decoder", "model.decoder", "transformer.decoder", "transformer.h", "transformer.layers", "transformer.block"]:
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
        print("Could not locate decoder layers for hybrid FT; skipping unfreeze.")
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

# ------------------------------ TRAINING (SFT) wrapper with optional hybrid FT
def train_model(max_samples=None, enable_lora=ENABLE_LORA, enable_adapter=ENABLE_ADAPTER,
                enable_hybrid=ENABLE_HYBRID, hybrid_unfreeze_topk=HYBRID_UNFREEZE_TOPK,
                hybrid_steps=HYBRID_STEPS, hybrid_lr=HYBRID_LR):
    # Deferred imports
    from trl import SFTTrainer, SFTConfig

    model, tok = prepare_model(MODEL_NAME, LORA_R, LORA_ALPHA, LORA_DROPOUT, enable_lora, enable_adapter, ADAPTER_REDUCTION)
    ds = PralekhaDataset(tok, max_samples=max_samples)

    # detect dtype for trainer config
    p = next(model.parameters())
    use_fp16 = (p.dtype==torch.float16)
    use_bf16 = (p.dtype==torch.bfloat16)

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=20,
        save_strategy="no",
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=2,
        report_to="none"
    )

    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    print("Starting base SFT training...")
    trainer.train()
    print("Base SFT finished; saving artifacts...")
    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    # quick eval after base SFT
    try:
        from sacrebleu import corpus_bleu
    except Exception:
        pass

    # If hybrid selective FT requested: unfreeze top-k and run additional light FT
    if enable_hybrid:
        print("Starting hybrid selective fine-tuning stage (unfreeze top-k decoder layers).")
        unfreeze_topk_decoder_layers(trainer.model, k=hybrid_unfreeze_topk)
        # re-detect dtype/pad
        p = next(trainer.model.parameters())
        use_fp16 = (p.dtype==torch.float16)
        use_bf16 = (p.dtype==torch.bfloat16)
        cfg2 = SFTConfig(
            output_dir=str(OUTPUT_DIR),
            per_device_train_batch_size=BATCH_SIZE,
            gradient_accumulation_steps=GRAD_ACCUM,
            learning_rate=hybrid_lr,
            lr_scheduler_type="cosine",
            num_train_epochs=1,
            max_steps=hybrid_steps,
            logging_steps=20,
            save_strategy="no",
            fp16=use_fp16,
            bf16=use_bf16,
            dataloader_num_workers=2,
            report_to="none"
        )
        trainer2 = SFTTrainer(model=trainer.model, args=cfg2, train_dataset=ds, tokenizer=tok)
        trainer2.train()
        trainer2.model.save_pretrained(OUTPUT_DIR)
        tok.save_pretrained(OUTPUT_DIR)
        trainer = trainer2  # promote final trainer

    return trainer.model, tok, trainer

# ------------------------------ EVALUATION & submission
def evaluate_and_write_submission(model, tok, max_samples_per_split=None, max_new_tokens=256, batch_size=EVAL_BATCH_SIZE):
    from datasets import get_dataset_split_names, load_dataset
    import sacrebleu
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

    # write JSONL per direction and zip
    for d in preds:
        arr = preds[d]
        fname = OUTPUT_DIR / f"{d.replace('_','-')}.jsonl"
        with open(fname, "w", encoding="utf-8") as f:
            for p in arr:
                f.write(json.dumps([p], ensure_ascii=False) + "\n")
    sub_zip = OUTPUT_DIR / "submission.zip"
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
    with open(signature_path, "w", encoding="utf-8") as f:
        f.write("EVAL SIGNATURE:\n")
        try:
            f.write(str(sacrebleu.corpus_bleu(["a"], [["a"]]).signature))
        except Exception:
            f.write("sacrebleu not available to compute signature.\n")
    print("Saved sacreBLEU signature ->", signature_path)

    return bleu, chrf, preds, refs

# ------------------------------ Backtranslation helper (deterministic)
def backtranslate(model, tokenizer, input_sentences, src_lang, tgt_lang,
                  batch_size=8, max_new_tokens=128, output_file="bt_synthetic.jsonl"):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    out_pairs = []
    torch.manual_seed(SEED)
    for i in range(0, len(input_sentences), batch_size):
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

# ------------------------------ Plotting helpers (safe)
def plot_training(trainer, out_dir=OUTPUT_DIR):
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plot_training.")
        return
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    if not steps:
        print("No training logs found to plot.")
        return
    plt.figure(figsize=(8,4))
    plt.plot(steps, losses)
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss Trend")
    plt.tight_layout(); plt.savefig(out_dir / "training_loss.png"); plt.close()
    print("Saved training loss curve.")

# ------------------------------ CLI / MAIN
def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--enable_adapter", action="store_true")
    parser.add_argument("--enable_lora", action="store_true")
    parser.add_argument("--no_hybrid", action="store_true", help="Disable hybrid top-k unfreeze stage")
    parser.add_argument("--hybrid_topk", type=int, default=HYBRID_UNFREEZE_TOPK)
    parser.add_argument("--hybrid_steps", type=int, default=HYBRID_STEPS)
    parser.add_argument("--hybrid_lr", type=float, default=HYBRID_LR)
    parser.add_argument("--eval_samples_per_split", type=int, default=None)
    args = parser.parse_args()

    max_samples = args.max_samples or (None if FULL_DATASET else MAX_COLAB_SAMPLES)
    enable_adapter = args.enable_adapter
    enable_lora = args.enable_lora or ENABLE_LORA
    enable_hybrid = (not args.no_hybrid) and ENABLE_HYBRID

    print("CONFIG SUMMARY:")
    print(" MODEL_NAME:", MODEL_NAME)
    print(" OUTPUT_DIR:", OUTPUT_DIR)
    print(" ENABLE_LORA:", enable_lora, "ENABLE_ADAPTER:", enable_adapter, "ENABLE_HYBRID:", enable_hybrid)

    # 1) Train
    model, tok, trainer = train_model(max_samples=max_samples,
                                     enable_lora=enable_lora,
                                     enable_adapter=enable_adapter,
                                     enable_hybrid=enable_hybrid,
                                     hybrid_unfreeze_topk=args.hybrid_topk,
                                     hybrid_steps=args.hybrid_steps,
                                     hybrid_lr=args.hybrid_lr)
    # 2) Evaluate & write submission
    bleu, chrf, preds, refs = evaluate_and_write_submission(model, tok, max_samples_per_split=args.eval_samples_per_split)
    # 3) Plot training
    try:
        plot_training(trainer)
    except Exception as e:
        print("Plotting failed:", e)

    # Print a small summary
    avg_bleu = sum(bleu.values())/len(bleu) if bleu else 0.0
    print(f"Average BLEU: {avg_bleu:.3f}")
    print("Artifacts saved in:", OUTPUT_DIR)

if __name__ == "__main__":
    main()

# End of script
