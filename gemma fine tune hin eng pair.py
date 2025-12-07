# ======================================================
# ✅ Universal Fine-tuning + Evaluation for any Hugging Face instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, Top-10 Preview)
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
from tqdm import tqdm
from IPython.display import display, Markdown, Image
import pandas as pd
import numpy as np

# ------------------------------ CONFIG
MODEL_NAME = "google/gemma-3-270m-it"   # replace with any HF causal/instruct LM
OUTPUT_DIR = Path("/kaggle/working/universal_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
BATCH_SIZE = 1
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 100
EVAL_BATCH_SIZE = 8
FULL_DATASET = False
MAX_COLAB_SAMPLES = 500

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}

# ------------------------------ UNIVERSAL PROMPT BUILDER
def build_prompt(src, src_lang, tgt_lang, example=("", ""), tokenizer=None):
    ex_src, ex_tgt = example
    if tokenizer and hasattr(tokenizer, "apply_chat_template"):
        msgs = []
        if ex_src and ex_tgt:
            msgs.append({"role":"user","content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"})
            msgs.append({"role":"assistant","content":ex_tgt})
        msgs.append({"role":"user","content":f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"})
        msgs.append({"role":"assistant","content":""})
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    else:
        if ex_src and ex_tgt:
            return f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n{ex_src} → {ex_tgt}\n\nTranslate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"
        else:
            return f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"

# ------------------------------ EVAL PROMPT
def eval_prompt(src, src_lang, tgt_lang):
    return f"Translate the following {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}\nTranslation: "

# ------------------------------ UTILS
def extract_answer(full_output, prompt):
    if not full_output:
        return ""
    try:
        if prompt and prompt in full_output:
            return full_output.split(prompt, 1)[1].strip()
    except Exception:
        pass
    markers = ["Translation:", "Translation -", "Translation —", "Output:", "Answer:", "\n"]
    for m in markers:
        if m in full_output:
            return full_output.split(m, 1)[1].strip()
    return full_output.strip()

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
                example = one_shot if not "dev" in split else ("","")
                yield {
                    "input_text": build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], example, tokenizer),
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
            inp = (s_enc["input_ids"] + t_enc["input_ids"])[:MAX_SEQ_LEN]
            lbl = ([-100]*len(s_enc["input_ids"]) +
                   [min(i,self.tok.vocab_size-1) for i in t_enc["input_ids"]])[:MAX_SEQ_LEN]
            yield {"input_ids": inp, "attention_mask":[1]*len(inp), "labels": lbl}

# ------------------------------ MODEL PREP
def detect_lora_modules(model):
    modules = []
    for n,m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","gate_proj","v_proj","o_proj", "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None: tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float32, device_map="auto")
    target_modules = detect_lora_modules(model)
    print(f"⚡ LoRA target modules detected: {target_modules}")
    lora_cfg = LoraConfig(
        r=8, lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.1, task_type="CAUSAL_LM"
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
        learning_rate=3e-5,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.1
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
    target_lang = "hin"
    for d in [f"eng_{target_lang}", f"{target_lang}_eng"]:
        preds[d], refs[d] = [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    print("\n🔍 Starting batched evaluation (ENG<->HIN only)...\n")

    for split in tqdm(splits, desc="Evaluating language pairs"):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if not ((sl=="eng" and tl==target_lang) or (sl==target_lang and tl=="eng")):
            continue
        lang = tl if sl=="eng" else sl
        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        batch_prompts, batch_prompts_raw, batch_refs, batch_dirs, count = [], [], [], [], 0

        for row in ds:
            if max_samples_per_split and count >= max_samples_per_split: break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)

            p_eng2hin = eval_prompt(eng, "eng", lang)
            p_hin2eng = eval_prompt(indic, lang, "eng")

            batch_prompts += [p_eng2hin, p_hin2eng]
            batch_prompts_raw += [p_eng2hin, p_hin2eng]
            batch_refs += [indic.strip(), eng.strip()]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]
            count += 1

            if len(batch_prompts) >= batch_size:
                enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
                decs = tok.batch_decode(out, skip_special_tokens=True)
                for dirn, pred_raw, ref, prompt in zip(batch_dirs, decs, batch_refs, batch_prompts_raw):
                    clean_pred = extract_answer(pred_raw, prompt)
                    preds[dirn].append(clean_pred)
                    refs[dirn].append(ref.strip())
                batch_prompts, batch_prompts_raw, batch_refs, batch_dirs = [], [], [], []

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

    bleu_scores, chrf_scores, comet_scores = {}, {}, {}
    for d in preds:
        if not preds[d]: continue
        try:
            bleu_scores[d] = sacrebleu.corpus_bleu(preds[d],[refs[d]]).score
        except: bleu_scores[d] = 0.0
        try:
            chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r] for r in refs[d]]).score
        except: chrf_scores[d] = 0.0

    for metric, scores in [("BLEU",bleu_scores),("chrF",chrf_scores)]:
        plt.figure(figsize=(10,5))
        langs, vals = list(scores.keys()), [scores[k] for k in scores]
        plt.bar(langs,vals)
        plt.title(f"{metric} Scores per Direction")
        plt.xticks(rotation=45,ha="right"); plt.tight_layout()
        plt.savefig(OUTPUT_DIR / f"{metric.lower()}_scores.png"); plt.close()
    print("📈 Metrics plots saved.")

    print("\n🔠 Sample Translations (Top 10 per direction):\n")
    for d in preds.keys():
        display(Markdown(f"### {d.upper()}"))
        for i in range(min(10,len(preds[d]))):
            display(Markdown(f"**Ref:** {refs[d][i]}  \n**Pred:** {preds[d][i]}"))

    return bleu_scores, chrf_scores, comet_scores

# ------------------------------ TRAIN CURVE
def plot_training(trainer):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]
    plt.figure(figsize=(8,4))
    plt.plot(steps,losses)
    plt.xlabel("Step"); plt.ylabel("Loss")
    plt.title("Training Loss Trend")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png")
    print("📉 Training loss curve saved.")

# ------------------------------ MAIN
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"]="1"
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES

    # 1️⃣ Train
    model, tok, trainer = train_model(max_samples=max_samples)

    # 2️⃣ Evaluate
    bleu, chrf, comet = evaluate_model(
        model, tok,
        max_samples_per_split=None if FULL_DATASET else 200,
        batch_size=EVAL_BATCH_SIZE
    )

    # 3️⃣ Plot training curve
    plot_training(trainer)

# ------------------------------ Enhanced Training Plots
logs = trainer.state.log_history
df = pd.DataFrame(logs)
df["loss_smooth"] = df["loss"].rolling(window=10, min_periods=1).mean()

plt.figure(figsize=(8, 4))
plt.plot(df["step"], df["loss"], label="Raw Loss", alpha=0.5, color="gray")
plt.plot(df["step"], df["loss_smooth"], label="Smoothed Loss", color="blue", linewidth=2)
plt.xlabel("Step"); plt.ylabel("Loss"); plt.title("Training Loss Over Steps")
plt.legend(); plt.tight_layout()
plt.savefig(OUTPUT_DIR / "training_loss_smooth.png"); plt.close()

if "learning_rate" in df.columns:
    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["learning_rate"], label="Learning Rate", color="orange")
    plt.xlabel("Step"); plt.ylabel("LR"); plt.title("Learning Rate Schedule")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "learning_rate_trend.png"); plt.close()

if "epoch" in df.columns:
    plt.figure(figsize=(8, 4))
    plt.scatter(df["epoch"], df["loss"], color="green", s=20, alpha=0.6, label="Raw Loss per Epoch")
    epoch_means = df.groupby("epoch")["loss"].mean()
    plt.plot(epoch_means.index, epoch_means.values, color="red", linewidth=2, label="Mean Loss per Epoch")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("Loss per Epoch")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "epoch_loss_trend.png"); plt.close()

if len(df) > 5:
    df["loss_derivative"] = np.gradient(df["loss_smooth"])
    plt.figure(figsize=(8, 4))
    plt.plot(df["step"], df["loss_derivative"], color="purple", label="d(Loss)/d(Step)")
    plt.axhline(0, color="black", linestyle="--", alpha=0.5)
    plt.xlabel("Step"); plt.ylabel("Loss Change"); plt.title("Loss Change Rate")
    plt.legend(); plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "loss_derivative_curve.png"); plt.close()

# ------------------------------ BLEU/chrF Metric Plots
plot_dir = OUTPUT_DIR / "metric_plots"
plot_dir.mkdir(exist_ok=True, parents=True)

def plot_metric(metric_name, scores_dict):
    if not scores_dict: return
    langs, vals = list(scores_dict.keys()), [scores_dict[k] for k in scores_dict]
    plt.figure(figsize=(12,6))
    plt.bar(langs, vals, color='skyblue')
    plt.title(f"{metric_name} Scores per Direction", fontsize=16)
    plt.xlabel("Language Direction", fontsize=12)
    plt.ylabel(metric_name, fontsize=12)
    plt.xticks(rotation=45, ha="right")
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
    plt.savefig(plot_dir / f"{metric_name.lower()}_per_direction.png"); plt.close()

plot_metric("BLEU", bleu)
plot_metric("chrF", chrf)
plot_metric("COMET", comet)
