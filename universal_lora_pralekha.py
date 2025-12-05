# ======================================================
# ✅ Universal Fine-tuning + Evaluation for any HF causal LM
# (Streaming, LoRA, FP16, Top-10 Preview, BLEU/chrF/COMET-ready)
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
from IPython.display import display, Markdown

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

INDIAN_LANGS = [
    "hin","ben","tam","tel","mal","kan","mar","guj","urd","pan","ori"
]

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
            {"role":"user",
             "content":f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{ex_src}"},
            {"role":"assistant","content":ex_tgt},
            {"role":"user",
             "content":f"Now translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"},
            {"role":"assistant","content":""}
        ]
        return tokenizer.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)

    return (
        f"Example translation ({LANG_MAP[src_lang]} → {LANG_MAP[tgt_lang]}):\n"
        f"{ex_src} → {ex_tgt}\n\n"
        f"Translate this {LANG_MAP[src_lang]} text to {LANG_MAP[tgt_lang]}:\n{src}"
    )

# ------------------------------ STREAMING DATASET
def stream_examples(tokenizer, max_samples=None):
    dataset_name = "ai4bharat/Pralekha"
    config_name = "train"
    splits = get_dataset_split_names(dataset_name, config_name)

    for split in splits:
        parts = split.split("_")
        if len(parts)!=2: continue

        sl, tl = parts
        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]:
            continue

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

            for s_txt, t_txt, dirn in [
                (eng, indic, f"eng_{lang}"),
                (indic, eng, f"{lang}_eng")
            ]:
                yield {
                    "input_text": build_prompt(
                        s_txt, dirn.split("_")[0], dirn.split("_")[1],
                        one_shot, tokenizer
                    ),
                    "target_text": t_txt,
                    "direction": dirn
                }
            count += 1

# ------------------------------ ITERABLE WRAPPER
class PralekhaDataset(IterableDataset):
    def __init__(self, tokenizer, max_samples=None):
        self.tok = tokenizer
        self.max_samples = max_samples

    def __iter__(self):
        for ex in stream_examples(self.tok, self.max_samples):
            s_enc = self.tok(
                ex["input_text"], truncation=True, max_length=MAX_SEQ_LEN,
                add_special_tokens=False
            )
            t_enc = self.tok(
                ex["target_text"], truncation=True, max_length=MAX_SEQ_LEN,
                add_special_tokens=True
            )

            inp = (s_enc["input_ids"] + t_enc["input_ids"])[:MAX_SEQ_LEN]
            lbl = (
                [-100]*len(s_enc["input_ids"]) +
                [min(i, self.tok.vocab_size-1) for i in t_enc["input_ids"]]
            )[:MAX_SEQ_LEN]

            yield {
                "input_ids": inp,
                "attention_mask": [1]*len(inp),
                "labels": lbl
            }

# ------------------------------ MODEL PREP
def detect_lora_modules(model):
    modules=[]
    for n,m in model.named_modules():
        n_lower = n.lower()
        if any(x in n_lower for x in [
            "q_proj","k_proj","v_proj","gate_proj","o_proj",
            "up_proj","down_proj","attn.wq","attn.wk","attn.wv","attn.wo"
        ]):
            modules.append(n.split(".")[-1])
    return list(set(modules))

def prepare_model():
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,     # FIXED (FP16)
        device_map="auto"
    )

    target_modules = detect_lora_modules(model)
    print("⚡ LoRA target modules:", target_modules)

    lora_cfg = LoraConfig(
        r=16,
        lora_alpha=16,
        target_modules=target_modules,
        lora_dropout=0.05,
        task_type="CAUSAL_LM"
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
        logging_steps=5,          # FIXED
        save_strategy="no",
        report_to="none",
        warmup_ratio=0.03         # FIXED
    )

    trainer = SFTTrainer(
        model=model, args=cfg,
        train_dataset=ds,
        tokenizer=tok
    )
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tok.save_pretrained(OUTPUT_DIR)
    return model, tok, trainer

# ------------------------------ EVALUATION
def evaluate_model(model, tok, max_new_tokens=256, max_samples_per_split=None, batch_size=EVAL_BATCH_SIZE):
    warnings.filterwarnings("ignore")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()

    # comet = evaluate.load("comet")

    preds, refs = {}, {}
    for lang in INDIAN_LANGS:
        for d in [f"eng_{lang}", f"{lang}_eng"]:
            preds[d], refs[d] = [], []

    splits = get_dataset_split_names("ai4bharat/Pralekha","dev")
    print("\n🔍 Evaluating...\n")

    for split in tqdm(splits):
        parts = split.split("_")
        if len(parts)!=2: continue
        sl, tl = parts

        if sl not in INDIAN_LANGS+["eng"] or tl not in INDIAN_LANGS+["eng"]:
            continue

        lang = tl if sl=="eng" else sl
        if lang not in INDIAN_LANGS: continue

        ds = load_dataset("ai4bharat/Pralekha", split=split, streaming=True, name="dev")

        batch_prompts = []
        batch_refs = []
        batch_dirs = []
        count = 0

        for row in ds:
            if max_samples_per_split and count >= max_samples_per_split:
                break

            s, t = row.get("src_txt",""), row.get("tgt_txt","")
            if not s or not t:
                continue

            eng, indic = (s,t) if sl=="eng" else (t,s)

            batch_prompts += [
                build_prompt(eng,"eng",lang,("Ex","Ex"),tok),
                build_prompt(indic,lang,"eng",("Ex","Ex"),tok)
            ]
            batch_refs += [indic, eng]
            batch_dirs += [f"eng_{lang}", f"{lang}_eng"]

            count += 1

            if len(batch_prompts) >= batch_size:
                enc = tok(batch_prompts, return_tensors="pt", padding=True,
                          truncation=True, max_length=MAX_SEQ_LEN).to(device)

                with torch.no_grad():
                    out = model.generate(
                        **enc,
                        max_new_tokens=max_new_tokens,
                        pad_token_id=tok.pad_token_id,
                        eos_token_id=tok.eos_token_id   # FIXED
                    )

                decs = tok.batch_decode(out, skip_special_tokens=True)

                for d, p, r in zip(batch_dirs, decs, batch_refs):
                    preds[d].append(p.strip())
                    refs[d].append(r.strip())

                batch_prompts, batch_refs, batch_dirs = [], [], []

    # ---------------- Save ZIP
    zip_path = OUTPUT_DIR / "submission.zip"
    with zipfile.ZipFile(zip_path,"w") as zf:
        for d in preds:
            chunks = math.ceil(len(preds[d]) / 1000)
            for i in range(chunks):
                sub = preds[d][i*1000:(i+1)*1000]
                if not sub: continue
                fp = OUTPUT_DIR / f"{d.replace('_','_2_')}_{i+1}.jsonl"
                with open(fp,"w",encoding="utf-8") as f:
                    for p in sub:
                        f.write(json.dumps([p],ensure_ascii=False)+"\n")
                zf.write(fp, fp.name)
    print(f"✅ Submission created: {zip_path}")

    # ---------------- Metrics
    bleu_scores = {}
    chrf_scores = {}
    comet_scores = {}

    for d in preds:
        if not preds[d]: continue
        
        bleu_scores[d] = sacrebleu.corpus_bleu(preds[d],[refs[d]]).score
        chrf_scores[d] = sacrebleu.corpus_chrf(preds[d], [[r] for r in refs[d]]).score
        # comet_scores[d] = comet.compute(...)["mean_score"]

    # ---------------- Plots
    plot_dir = OUTPUT_DIR / "metric_plots"
    plot_dir.mkdir(exist_ok=True)

    def plot_metric(name, data):
        if not data: return
        plt.figure(figsize=(12,6))
        plt.bar(list(data.keys()), list(data.values()))
        plt.xticks(rotation=45,ha="right")
        plt.title(f"{name} per direction")
        plt.tight_layout()
        plt.savefig(plot_dir / f"{name.lower()}_plot.png")
        plt.close()

    plot_metric("BLEU", bleu_scores)
    plot_metric("chrF", chrf_scores)
    #plot_metric("COMET", comet_scores)

    # ---------------- Previews
    print("\n🔠 Top-10 samples:\n")
    for d in preds:
        print(f"--- {d.upper()} ---")
        for i in range(min(10, len(preds[d]))):
            print(f"[REF] {refs[d][i]}")
            print(f"[PRED] {preds[d][i]}\n")

    return bleu_scores, chrf_scores, comet_scores

# ------------------------------ TRAIN CURVE
def plot_training(trainer):
    logs = trainer.state.log_history
    steps = [l["step"] for l in logs if "loss" in l]
    losses = [l["loss"] for l in logs if "loss" in l]

    plt.figure(figsize=(8,4))
    plt.plot(steps,losses)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "training_loss.png")
    plt.close()
    print("📉 Training loss saved.")

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

    # 3️⃣ Plot training
    plot_training(trainer)

    print("\n🎉 DONE! All outputs saved to:", OUTPUT_DIR)
