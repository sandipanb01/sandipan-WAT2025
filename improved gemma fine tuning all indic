# ======================================================
# ✅ Universal Fine-tuning + Evaluation for any Hugging Face instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, Top-10 Preview)
# Includes enhanced training visualizations (smoothed loss, derivative, LR trend)
# ======================================================

import os, json, zipfile, math, warnings
from pathlib import Path
from itertools import islice
import torch
from datasets import load_dataset, get_dataset_split_names
from torch.utils.data import IterableDataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu, evaluate
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from IPython.display import display, Markdown, Image

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("/content/universal_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 3000
EVAL_BATCH_SIZE = 8
FULL_DATASET = True
MAX_COLAB_SAMPLES = None

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ UNIVERSAL PROMPT BUILDER
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
        return f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n{ex_src} → {ex_tgt}\n\nTranslate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)

    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        for row in islice(ds, 50):
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if len(s.split())>5 and len(t.split())>5:
                one_shot = (s,t); break

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            for s_txt,t_txt,dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                yield {
                    "input_text": build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], one_shot, tokenizer),
                    "target_text": t_txt, "direction": dirn
                }
            count += 1

# ------------------------------ ITERABLE WRAPPER
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tok = tokenizer
        self.max_samples = max_samples
    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            s_enc = self.tok(ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=False)
            t_enc = self.tok(ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN, add_special_tokens=True)
            inp_ids = s_enc["input_ids"] + t_enc["input_ids"]
            lbl_ids = [-100]*len(s_enc["input_ids"]) + t_enc["input_ids"]
            if len(inp_ids) > MAX_SEQ_LEN:
                inp_ids = inp_ids[:MAX_SEQ_LEN]
                lbl_ids = lbl_ids[:MAX_SEQ_LEN]
            yield {"input_ids": inp_ids, "attention_mask":[1]*len(inp_ids), "labels": lbl_ids}

# ------------------------------ MODEL PREP
def detect_lora_modules(model):
    target_keywords = ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","gate_proj","proj","linear"]
    modules = []
    for n, m in model.named_modules():
        if isinstance(m, torch.nn.Linear):
            n_lower = n.lower()
            if any(k in n_lower for k in target_keywords):
                modules.append(n)
    modules = list(set(modules))
    print(f"⚡ LoRA target modules detected ({len(modules)}): {modules}")
    return modules

def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    target_modules = detect_lora_modules(model)
    lora_cfg = LoraConfig(
        r=16, lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05, task_type="CAUSAL_LM"
    )
    return get_peft_model(model, lora_cfg), tok

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tok = prepare_model()
    ds = PralekhaDataset(tok, max_samples=max_samples)
    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.03
    )
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    trainer.train()
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION
def evaluate_model(model, tok, max_new_tokens=256, max_samples_per_split=None, batch_size=EVAL_BATCH_SIZE):
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        for d in [f"eng_{lang}", f"{lang}_eng"]:
            preds[d], refs[d] = [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    print("\n🔍 Starting batched evaluation...\n")

    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        batch_prompts, batch_refs, batch_dirs, count = [], [], [], 0

        for row in ds:
            if max_samples_per_split and count >= max_samples_per_split: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            batch_prompts += [
                build_prompt(eng,"eng",lang,("Example","Example"),tok),
                build_prompt(indic,lang,"eng",("Example","Example"),tok)
            ]
            batch_refs += [indic, eng]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]
            count += 1

            if len(batch_prompts) >= batch_size:
                enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
                decs = tok.batch_decode(out, skip_special_tokens=True)
                for dirn, pred, ref in zip(batch_dirs,decs,batch_refs):
                    preds[dirn].append(pred.strip())
                    refs[dirn].append(ref.strip())
                batch_prompts, batch_refs, batch_dirs = [], [], []

    # Save JSONL
    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip,"w") as zf:
        for d in preds:
            n_chunks = math.ceil(len(preds[d])/1000)
            for i in range(n_chunks):
                chunk = preds[d][i*1000:(i+1)*1000]
                if not chunk: continue
                fp = OUTPUT_DIR / f"{d.replace('_','_2_')}_{i+1}.jsonl"
                with open(fp,"w",encoding="utf-8") as f:
                    for p in chunk: f.write(json.dumps([p],ensure_ascii=False)+"\n")
                zf.write(fp, fp.name)
    print(f"\n✅ Submission ZIP saved: {sub_zip}")

    # Metrics
    bleu_scores, chrf_scores = {}, {}
    for d in preds:
        if not preds[d]: continue
        bleu_scores[d] = sacrebleu.corpus_bleu(preds[d],[refs[d]]).score
        chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r] for r in refs[d]]).score

    # Top-10 samples
    print("\n🔠 Sample Translations (Top 10 per direction):\n")
    for d in preds.keys():
        display(Markdown(f"### {d.upper()}"))
        for i in range(min(10,len(preds[d]))):
            display(Markdown(f"**Ref:** {refs[d][i]}  \n**Pred:** {preds[d][i]}"))

    return bleu_scores, chrf_scores, {}

# ------------------------------ ENHANCED TRAINING VISUALS
def plot_training_enhanced(trainer):
    logs = trainer.state.log_history
    df = pd.DataFrame(logs)
    if df.empty: return
    df["loss_smooth"] = df["loss"].rolling(window=10, min_periods=1).mean()

    # Raw + Smoothed Loss
    plt.figure(figsize=(8,4))
    plt.plot(df["step"], df["loss"], alpha=0.5, color="gray", label="Raw Loss")
    plt.plot(df["step"], df["loss_smooth"], color="blue", linewidth=2, label="Smoothed Loss")
    plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss (Raw + Smoothed)"); plt.legend()
    plt.tight_layout(); plt.savefig(OUTPUT_DIR / "training_loss_smooth.png"); plt.close()
    display(Image(filename=OUTPUT_DIR / "training_loss_smooth.png"))

    # Learning Rate
    if "learning_rate" in df.columns:
        plt.figure(figsize=(8,4))
        plt.plot(df["step"], df["learning_rate"], color="orange", label="LR")
        plt.xlabel("Step"); plt.ylabel("LR"); plt.title("Learning Rate Schedule"); plt.legend()
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "learning_rate_trend.png"); plt.close()
        display(Image(filename=OUTPUT_DIR / "learning_rate_trend.png"))

    # Loss per epoch
    if "epoch" in df.columns:
        plt.figure(figsize=(8,4))
        plt.scatter(df["epoch"], df["loss"], color="green", alpha=0.6, s=20, label="Raw Loss")
        epoch_means = df.groupby("epoch")["loss"].mean()
        plt.plot(epoch_means.index, epoch_means.values, color="red", linewidth=2, label="Mean Loss per Epoch")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch"); plt.legend()
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "epoch_loss_trend.png"); plt.close()
        display(Image(filename=OUTPUT_DIR / "epoch_loss_trend.png"))

    # Loss derivative
    if len(df) > 5:
        df["loss_derivative"] = np.gradient(df["loss_smooth"])
        plt.figure(figsize=(8,4))
        plt.plot(df["step"], df["loss_derivative"], color="purple", label="d(Loss)/d(Step)")
        plt.axhline(0, color="black", linestyle="--", alpha=0.5)
        plt.xlabel("Step"); plt.ylabel("Loss Change"); plt.title("Loss Change Rate"); plt.legend()
        plt.tight_layout(); plt.savefig(OUTPUT_DIR / "loss_derivative_curve.png"); plt.close()
        display(Image(filename=OUTPUT_DIR / "loss_derivative_curve.png"))

    print("\n✅ Enhanced training plots saved to:", OUTPUT_DIR)

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # Train
    model, tok, trainer = train_model(max_samples=max_samples)

    # Evaluate
    bleu, chrf, comet = evaluate_model(
        model, tok,
        max_samples_per_split=None if FULL_DATASET else 200,
        batch_size=EVAL_BATCH_SIZE
    )

    # Plot enhanced training visuals
    plot_training_enhanced(trainer)
