# ======================================================
# ✅ Universal Fine-tuning + Evaluation for any Hugging Face instruct/causal LM
# (Streaming, LoRA, Fast Evaluation, Metrics, Top-10 Preview, FP16/BF16, multi-GPU aware)
# T4-friendly fixes: automatic dtype/workers/limits so training actually starts on T4.
# Merged A + B — training + eval pipeline, with safe default toggles.
# ======================================================

import os, json, zipfile, math, warnings, time
from pathlib import Path
from itertools import islice
import torch
from torch.utils.data import IterableDataset
from datasets import load_dataset, get_dataset_split_names
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import LoraConfig, get_peft_model
from trl import SFTTrainer, SFTConfig
import sacrebleu
try:
    import evaluate
except Exception:
    evaluate = None
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from tqdm import tqdm
from IPython.display import display, Markdown

# ------------------------------ CONFIG (tweak if you want)
MODEL_NAME = "google/gemma-3-270m-it"
OUTPUT_DIR = Path("/kaggle/working/universal_output")
OUTPUT_DIR.mkdir(exist_ok=True, parents=True)

MAX_SEQ_LEN = 1024
PER_DEVICE_BATCH = 1         # effective per-device micro-batch
GRAD_ACCUM = 4
MAX_TRAIN_STEPS = 50       # will be reduced automatically on small GPUs for safety
EVAL_BATCH_SIZE = 8
FULL_DATASET = True          # toggle full dataset or capped samples on small GPUs
MAX_SAMPLES = None           # if None, script may auto-cap on small GPUs
NUM_WORKERS = 3
# For T4: recommend FP16; on A100/H100 prefer BF16. Use these to override auto-detection:
FORCE_FP16 = True
FORCE_BF16 = False

HF_AUTH_TOKEN_ENV = "USE YOUR OWN HF TOKEN"

INDIAN_LANGS = ["hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"]
LANG_MAP = {
    "eng":"English","hin":"Hindi","ben":"Bengali","tam":"Tamil",
    "tel":"Telugu","mal":"Malayalam","kan":"Kannada","mar":"Marathi",
    "guj":"Gujarati","urd":"Urdu","pan":"Punjabi","ori":"Odia"
}
  

# ------------------------------ PROMPT BUILDER
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
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)
    if not splits:
        raise RuntimeError("Could not get splits for ai4bharat/Pralekha (check datasets & internet).")

    for split in splits:
        parts = split.split("_")
        if len(parts) != 2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS + ["eng"] or tl not in INDIAN_LANGS + ["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        # streaming dataset (safe, memory-efficient)
        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        one_shot = ("","")
        try:
            for row in islice(ds,50):
                s,t = row.get("src_txt",""), row.get("tgt_txt","")
                if s and t and len(s.split())>5 and len(t.split())>5:
                    one_shot = (s,t)
                    break
        except Exception as e:
            print(f"Warning finding one-shot example for {split}: {e}")
            one_shot = ("","")

        ds = load_dataset(dataset_name, split=split, streaming=True, name=config_name)
        count = 0
        for row in ds:
            if max_samples and count >= max_samples:
                break
            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            for s_txt, t_txt, dirn in [(eng,indic,f"eng_{lang}"),(indic,eng,f"{lang}_eng")]:
                yield {"input_text": build_prompt(s_txt, dirn.split("_")[0], dirn.split("_")[1], one_shot, tokenizer),
                       "target_text": t_txt,
                       "direction": dirn}
            count += 1

# ------------------------------ ITERABLE DATASET
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

# ------------------------------ MODEL + LoRA detection
def detect_lora_modules(model):
    modules = []
    for n,m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in ["q_proj","k_proj","v_proj","o_proj","up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

# ------------------------------ PREPARE MODEL (safe load & device movement)
def prepare_model():
    hf_token = os.environ.get(HF_AUTH_TOKEN_ENV,None)
    if hf_token:
        os.environ["HF_HUB_TOKEN"] = hf_token

    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # GPU + dtype selection
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        gpu_name = torch.cuda.get_device_name(0).lower()
        print(f"⚡ Detected {n_gpus} CUDA GPU(s): {gpu_name}")
        # prefer bf16 only on A100/H100/A6000 (and if user doesn't force FP16)
        if (any(x in gpu_name for x in ["a100","h100","a6000"]) and not FORCE_FP16) or FORCE_BF16:
            dtype = torch.bfloat16
        else:
            dtype = torch.float16 if (not FORCE_FP16) else torch.float32
    else:
        print("⚠️ No CUDA detected — running on CPU")
        dtype = torch.float32

    # Load model onto CPU first (device_map=None) then move to GPU — avoids device_map auto issues on small GPUs
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading model {MODEL_NAME} with dtype={dtype} (device_map=None)...")
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, trust_remote_code=True, torch_dtype=dtype, device_map=None, low_cpu_mem_usage=False)
    model.to(device)
    print("Model loaded and moved to device:", device)

    target_modules = detect_lora_modules(model)
    print(f"⚡ LoRA target modules detected: {target_modules}")

    lora_cfg = LoraConfig(r=16, lora_alpha=16, target_modules=target_modules, lora_dropout=0.05, task_type="CAUSAL_LM")
    peft_model = get_peft_model(model, lora_cfg)
    peft_model.to(device)
    return peft_model, tok

# ------------------------------ TRAINING
def train_model(max_samples=None):
    model, tok = prepare_model()
    ds = PralekhaDataset(tok, max_samples=max_samples)

    # Determine flags for fp16/bf16 in SFTConfig from model dtype
    model_dtype = None
    for p in model.parameters():
        model_dtype = p.dtype
        break
    use_bf16 = (model_dtype == torch.bfloat16)
    use_fp16 = (model_dtype == torch.float16)

    # Respect user-forced choices but avoid both True
    if FORCE_FP16:
        use_fp16 = True
        use_bf16 = False
    if FORCE_BF16:
        use_bf16 = True
        use_fp16 = False

    # Safety print
    print(f"Training dtype flags -> use_fp16: {use_fp16}, use_bf16: {use_bf16}")
    print(f"Training dataset max samples: {max_samples}")

    cfg = SFTConfig(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=PER_DEVICE_BATCH,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=1.5e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        max_steps=MAX_TRAIN_STEPS,
        logging_steps=10,
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.03,
        fp16=use_fp16,
        bf16=use_bf16,
        dataloader_num_workers=NUM_WORKERS,
    )

    print("SFTConfig prepared:", cfg)
    trainer = SFTTrainer(model=model, args=cfg, train_dataset=ds, tokenizer=tok)
    print("Starting trainer.train() ...")
    trainer.train()
    print("Training finished. Saving model & tokenizer ...")
    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION
def evaluate_model(model, tok, max_new_tokens=256, max_samples_per_split=None, batch_size=EVAL_BATCH_SIZE):
    warnings.filterwarnings("ignore", message="Setting `pad_token_id` to `eos_token_id`")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    #comet = evaluate.load("comet")

    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        for d in [f"eng_{lang}", f"{lang}_eng"]:
            preds[d], refs[d] = [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha", "dev")
    print("\n🔍 Starting batched evaluation...\n")

    for split in tqdm(splits, desc="Evaluating language pairs"):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]: continue
        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")
        batch_prompts, batch_refs, batch_dirs, count = [], [], [], 0
        for row in ds:
            if max_samples_per_split and count>=max_samples_per_split: break
            s,t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t: continue
            eng, indic = (s,t) if sl=="eng" else (t,s)
            batch_prompts += [
                build_prompt(eng,"eng",lang,("Example","Example"),tok),
                build_prompt(indic,lang,"eng",("Example","Example"),tok)
            ]
            batch_refs += [indic, eng]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]
            count += 1
            if len(batch_prompts)>=batch_size:
                enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
                with torch.no_grad():
                    out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
                decs = tok.batch_decode(out, skip_special_tokens=True)
                for dirn, pred, ref in zip(batch_dirs, decs, batch_refs):
                    preds[dirn].append(pred.strip())
                    refs[dirn].append(ref.strip())
                batch_prompts, batch_refs, batch_dirs = [], [], []

        if batch_prompts:
            enc = tok(batch_prompts, return_tensors="pt", padding=True, truncation=True, max_length=MAX_SEQ_LEN).to(device)
            with torch.no_grad():
                out = model.generate(**enc, max_new_tokens=max_new_tokens, pad_token_id=tok.pad_token_id)
            decs = tok.batch_decode(out, skip_special_tokens=True)
            for dirn, pred, ref in zip(batch_dirs, decs, batch_refs):
                preds[dirn].append(pred.strip())
                refs[dirn].append(ref.strip())

    # Save predictions into a submission.zip (same format you used)
    sub_zip = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(sub_zip, "w") as zf:
        for d in preds:
            if not preds[d]: continue
            n_chunks = math.ceil(len(preds[d]) / 1000)
            for i in range(n_chunks):
                chunk = preds[d][i*1000:(i+1)*1000]
                if not chunk: continue
                fp = OUTPUT_DIR / f"{d.replace('_','_2_')}_{i+1}.jsonl"
                with open(fp, "w", encoding="utf-8") as f:
                    for p in chunk:
                        f.write(json.dumps([p], ensure_ascii=False) + "\n")
                zf.write(fp, fp.name)
    print(f"\n✅ Submission ZIP saved: {sub_zip}")

    bleu_scores, chrf_scores, comet_scores = {}, {}, {}
    for d in preds:
        if not preds[d]: continue
        try:
            bleu_scores[d] = sacrebleu.corpus_bleu(preds[d], [refs[d]]).score
            chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r for r in refs[d]]]).score
            #comet_scores[d] = comet.compute(predictions=preds[d], references=refs[d], sources=[""]*len(refs[d]))["mean_score"]

        except Exception as e:
            print(f"Warning computing metrics for {d}: {e}")
            bleu_scores[d] = 0.0
            chrf_scores[d] = 0.0
            comet_scores[d] = 0.0

    return bleu_scores, chrf_scores, comet_scores

# ------------------------------ TRAIN CURVE
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
    print("📉 Training loss curve saved.")

# ------------------------------ MAIN
if __name__=="__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # If FULL_DATASET True but small GPU, cap max_samples for safety
    gpu_ok_for_full = False
    if torch.cuda.is_available():
        gname = torch.cuda.get_device_name(0).lower()
        if any(x in gname for x in ["a100","h100","a6000"]):
            gpu_ok_for_full = True

    safe_max_samples = None if (FULL_DATASET and (MAX_SAMPLES is None) and gpu_ok_for_full) else (MAX_SAMPLES or 2000)
    max_samples = None if (FULL_DATASET and safe_max_samples is None) else safe_max_samples

    start_train = time.time()
    model, tok, trainer = train_model(max_samples=max_samples)
    print(f"Training completed in {(time.time()-start_train)/60:.2f} minutes")

    start_eval = time.time()
    bleu, chrf, comet = evaluate_model(model, tok,
                                       max_samples_per_split=None if FULL_DATASET else 200,
                                       batch_size=EVAL_BATCH_SIZE)
    print(f"Evaluation completed in {(time.time()-start_eval)/60:.2f} minutes")

    plot_training(trainer)

    # Build metrics table & display (same as your B)
    data=[]
    for d in sorted(set(list(bleu.keys())+list(chrf.keys())+list(comet.keys()))):
        data.append({
            "Direction": d,
            "BLEU": round(bleu.get(d, 0.0), 2),
            "chrF": round(chrf.get(d, 0.0), 2),
            "COMET": round(comet.get(d, 0.0), 4) if comet else "N/A"
        })
    df_metrics = pd.DataFrame(data).sort_values("Direction").reset_index(drop=True)
    display(Markdown("## 📋 Translation Quality Metrics per Direction"))
    display(df_metrics.style.background_gradient(cmap="YlGnBu", subset=["BLEU","chrF"]))

    avg_bleu = sum(bleu.values())/len(bleu) if bleu else 0
    avg_chrf = sum(chrf.values())/len(chrf) if chrf else 0
    avg_comet = sum(comet.values())/len(comet) if comet else 0
    print(f"\n🧮 Averages → BLEU: {avg_bleu:.2f}, chrF: {avg_chrf:.2f}, COMET: {avg_comet:.4f}")

    plot_dir = OUTPUT_DIR / "metric_plots"; plot_dir.mkdir(exist_ok=True, parents=True)
    def plot_metric(metric_name, scores_dict):
        if not scores_dict: return
        langs, vals = list(scores_dict.keys()), [scores_dict[k] for k in scores_dict]
        plt.figure(figsize=(12,6)); plt.bar(langs,vals)
        plt.title(f"{metric_name} Scores per Direction",fontsize=16)
        plt.xlabel("Language Direction",fontsize=12); plt.ylabel(metric_name,fontsize=12)
        plt.xticks(rotation=45,ha="right"); plt.grid(axis='y',linestyle='--',alpha=0.7)
        plt.tight_layout(); path=plot_dir/f"{metric_name.lower()}_per_direction.png"
        plt.savefig(path); plt.close()
        print(f"✅ Saved {metric_name} plot → {path}")

    plot_metric("BLEU", bleu)
    plot_metric("chrF", chrf)
    plot_metric("COMET", comet)

    print("\n✅ All done — outputs saved under:", OUTPUT_DIR)
