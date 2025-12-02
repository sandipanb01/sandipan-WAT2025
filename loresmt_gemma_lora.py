# ======================================================
#  UNIVERSAL FINE-TUNING & EVALUATION FOR LoResMT SUBMISSION
#  Fully Fixed • Deterministic • LoRA Optimized • Gemma-Friendly
#  Works on T4 / A100 / H100 • Produces submission-ready ZIP
# ======================================================

import os, json, zipfile, math, time, warnings, random
from pathlib import Path
from itertools import islice

import torch
import numpy as np
from torch.utils.data import IterableDataset

from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM

from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig

import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from IPython.display import display, Markdown


# ======================================================
#  CONFIGURATION (LoResMT-safe defaults)
# ======================================================

MODEL_NAME = "google/gemma-3-270m-it"

OUTPUT_DIR = Path("./loresmt_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
PER_DEVICE_BATCH = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 2000
EVAL_BATCH_SIZE = 8

FULL_DATASET = True
MAX_SAMPLES = None

NUM_WORKERS = 2

FORCE_FP16 = True
FORCE_BF16 = False

HF_AUTH_TOKEN_ENV = "HF_HUB_TOKEN"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}


# ======================================================
#  PROMPT BUILDER (safe, clean)
# ======================================================

def build_prompt(src, src_lang, tgt_lang, example, tokenizer):
    ex_src, ex_tgt = example

    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = [
            {"role":"user","content":
             f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"},
            {"role":"assistant","content":ex_tgt},
            {"role":"user","content":
             f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"},
            {"role":"assistant","content":""}
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    return (f"{LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]} translation example:\n"
            f"{ex_src} → {ex_tgt}\n\n"
            f"Translate:\n{src}\n")


# ======================================================
#  DATASET (streaming)
# ======================================================

def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    cfg = "train"
    splits = get_dataset_split_names(dataset_name, cfg)

    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue

        # choose the Indic language
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        # small one-shot example
        ds_temp = load_dataset(dataset_name, split=split, streaming=True, name=cfg)
        one_shot = ("","")
        for row in islice(ds_temp,50):
            s, t = row["src_txt"], row["tgt_txt"]
            if len(s.split())>5 and len(t.split())>5:
                one_shot = (s,t)
                break

        # main stream
        ds = load_dataset(dataset_name, split=split, streaming=True, name=cfg)
        count = 0
        for row in ds:
            if max_samples and count>=max_samples: break
            s, t = row["src_txt"], row["tgt_txt"]

            eng, indic = (s,t) if sl=="eng" else (t,s)

            yield {
                "input_text": build_prompt(eng,"eng",lang, one_shot, tokenizer),
                "target_text": indic,
                "direction": f"eng_{lang}"
            }
            yield {
                "input_text": build_prompt(indic,lang,"eng", one_shot, tokenizer),
                "target_text": eng,
                "direction": f"{lang}_eng"
            }
            count += 1


# ======================================================
#  ITERABLE DATASET (with correct label masking)
# ======================================================

class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tok = tokenizer
        self.max_samples = max_samples

    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            src_enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN)
            tgt_enc = self.tok(ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN)

            # proper seq2seq-style causal labeling:
            inp = src_enc["input_ids"] + tgt_enc["input_ids"]
            inp = inp[:MAX_SEQ_LEN]

            labels = [-100]*len(src_enc["input_ids"]) + tgt_enc["input_ids"]
            labels = labels[:MAX_SEQ_LEN]

            yield {
                "input_ids": inp,
                "attention_mask": [1]*len(inp),
                "labels": labels
            }


# ======================================================
#  DETECT LORA MODULES (stable)
# ======================================================

def detect_lora_modules(model):
    mods = []
    for n,m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in [
            "q_proj","k_proj","v_proj","o_proj",
            "up_proj","down_proj",
            "attn.wq","attn.wk","attn.wv","attn.wo"
        ]):
            mods.append(n.split(".")[-1])
    return list(set(mods))



# ======================================================
#  PREPARE MODEL (error-free, correct dtype logic)
# ======================================================

def prepare_model():

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # --- dtype logic fixed ---
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

    print(f"\nLoading {MODEL_NAME} with dtype={dtype}...\n")
    device = "cuda" if torch.cuda.is_available() else "cpu"

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        trust_remote_code=True,
        torch_dtype=dtype,
        low_cpu_mem_usage=True
    )

    model.to(device)

    target_modules = detect_lora_modules(model)
    print(f"LoRA modules: {target_modules}")

    lora_cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )

    model = get_peft_model(model, lora_cfg)
    model.to(device)

    return model, tok



# ======================================================
#  TRAINING
# ======================================================

def train_model(max_samples=None):
    model, tok = prepare_model()
    dataset = PralekhaDataset(tok, max_samples)

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
        load_best_model_at_end=True,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=NUM_WORKERS,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        args=cfg,
        train_dataset=dataset,
        tokenizer=tok
    )

    trainer.train()

    trainer.model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)

    return trainer.model, tok, trainer



# ======================================================
#  EVALUATION (sacreBLEU-safe, signature saved)
# ======================================================

def evaluate_model(model, tok, max_samples_per_split=None, max_new_tokens=256):

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds, refs = { }, { }
    for lang in INDIAN_LANGS:
        preds[f"eng_{lang}"] = []; refs[f"eng_{lang}"] = []
        preds[f"{lang}_eng"] = []; refs[f"{lang}_eng"] = []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")

    for split in tqdm(splits, desc="Evaluating"):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts

        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]:
            continue

        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS:
            continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")

        count = 0
        prompts = []
        references = []
        directions = []

        for row in ds:
            if max_samples_per_split and count>=max_samples_per_split: break
            s, t = row["src_txt"], row["tgt_txt"]
            eng, indic = (s,t) if sl=="eng" else (t,s)

            prompts.append(build_prompt(eng,"eng",lang,("Example","Example"),tok))
            references.append(indic)
            directions.append(f"eng_{lang}")

            prompts.append(build_prompt(indic,lang,"eng",("Example","Example"),tok))
            references.append(eng)
            directions.append(f"{lang}_eng")

            count += 1

            if len(prompts)>=EVAL_BATCH_SIZE:
                enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
                with torch.no_grad():
                    outs = model.generate(**enc, max_new_tokens=max_new_tokens,
                                          pad_token_id=tok.pad_token_id)

                decs = tok.batch_decode(outs, skip_special_tokens=True)

                for d, p, r in zip(directions, decs, references):
                    preds[d].append(p.strip())
                    refs[d].append(r.strip())

                prompts, references, directions = [], [], []

        # remaining batch
        if prompts:
            enc = tok(prompts, return_tensors="pt", padding=True, truncation=True).to(device)
            with torch.no_grad():
                outs = model.generate(**enc, max_new_tokens=max_new_tokens,
                                      pad_token_id=tok.pad_token_id)
            decs = tok.batch_decode(outs, skip_special_tokens=True)
            for d, p, r in zip(directions, decs, references):
                preds[d].append(p.strip())
                refs[d].append(r.strip())

    # create LoResMT submission.zip
    sub_zip = OUTPUT_DIR/"submission.zip"
    with zipfile.ZipFile(sub_zip,"w") as zf:
        for d in preds:
            arr = preds[d]
            for i in range(0, len(arr), 1000):
                chunk = arr[i:i+1000]
                jfile = OUTPUT_DIR/f"{d.replace('_','_2_')}_{(i//1000)+1}.jsonl"
                with open(jfile,"w",encoding="utf-8") as f:
                    for p in chunk:
                        f.write(json.dumps([p],ensure_ascii=False)+"\n")
                zf.write(jfile, jfile.name)

    print("\nSubmission ZIP saved:", sub_zip)

    # metrics
    bleu, chrf = {}, {}
    for d in preds:
        if len(preds[d])==0:
            bleu[d]=0; chrf[d]=0
            continue
        bleu[d] = sacrebleu.corpus_bleu(preds[d], [refs[d]]).score
        chrf[d] = sacrebleu.corpus_chrf(preds[d], [refs[d]]).score

    # save sacreBLEU signature
    signature_path = OUTPUT_DIR/"sacrebleu_signature.txt"
    with open(signature_path,"w") as f:
        f.write("EVAL SIGNATURE:\n")
        f.write(str(sacrebleu.corpus_bleu(["a"], [["a"]]).signature))

    print("\nSaved sacreBLEU signature →", signature_path)

    return bleu, chrf



# ======================================================
#  TRAIN CURVE
# ======================================================

def plot_training(trainer):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]

    if not steps:
        print("No logs to plot.")
        return

    plt.figure(figsize=(7,4))
    plt.plot(steps, losses)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR/"training_loss.png")
    plt.close()

    print("Saved training loss plot.")



# ======================================================
#  MAIN
# ======================================================

if __name__ == "__main__":

    # GPU safety
    if torch.cuda.is_available():
        gname = torch.cuda.get_device_name(0).lower()
        big_gpu = any(x in gname for x in ["a100","h100","a6000"])
    else:
        big_gpu = False

    safe_max_samples = None if (FULL_DATASET and big_gpu) else (MAX_SAMPLES or 2000)

    # TRAIN
    t0 = time.time()
    model, tok, trainer = train_model(max_samples=safe_max_samples)
    print(f"\nTraining time: {(time.time()-t0)/60:.2f} min\n")

    # EVAL
    t1 = time.time()
    bleu, chrf = evaluate_model(
        model, tok,
        max_samples_per_split=None if FULL_DATASET else 200
    )
    print(f"\nEvaluation time: {(time.time()-t1)/60:.2f} min\n")

    plot_training(trainer)

    print("\n=== FINAL AVERAGE RESULTS ===")
    print("BLEU:", sum(bleu.values())/len(bleu))
    print("chrF:", sum(chrf.values())/len(chrf))

    print("\nAll results saved under:", OUTPUT_DIR)
