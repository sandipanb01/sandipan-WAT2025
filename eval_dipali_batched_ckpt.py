# ============================================================
# 0. IMPORTS
# ============================================================
import json
import torch
import sacrebleu
import pandas as pd
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer
from peft import AutoPeftModelForCausalLM

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "google/gemma-3-270m-it"

CHECKPOINT_ROOT = Path("./gemma3_outputs/checkpoints")
OUTPUT_ROOT = Path("./checkpoint_eval_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DATASET_NAME = "ai4bharat/Pralekha"
EVAL_SPLIT = "test"

MAX_TGT_LEN = 2400
BATCH_SIZE = 4  # adjust based on GPU memory
DEVICE = "cuda"

# ============================================================
# 2. HELPERS
# ============================================================
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ============================================================
# 3. LOAD TEST SET ONCE
# ============================================================
print("📥 Loading Pralekha TEST split...")
dataset = load_dataset(DATASET_NAME, EVAL_SPLIT, split="eng_hin")

# ============================================================
# 4. DISCOVER CHECKPOINTS
# ============================================================
checkpoints = sorted(
    [p for p in CHECKPOINT_ROOT.iterdir() if p.name.startswith("checkpoint-")],
    key=lambda x: int(x.name.split("-")[-1])
)

assert len(checkpoints) > 0, "❌ No checkpoints found"
print(f"✅ Found {len(checkpoints)} checkpoints")

# ============================================================
# 5. LOAD TOKENIZER
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token
tokenizer.padding_side = "right"

# ============================================================
# 6. PRE-BUILD PROMPTS
# ============================================================
def build_prompts(example):
    return {
        "eng_prompt": (
            f"<start_of_turn>user\nTranslate to HINDI DEVANAGARI:\n"
            f"{example['src_txt']}<end_of_turn>\n<start_of_turn>model\n"
        ),
        "hin_prompt": (
            f"<start_of_turn>user\nTranslate to ENGLISH:\n"
            f"{example['tgt_txt']}<end_of_turn>\n<start_of_turn>model\n"
        )
    }

dataset = dataset.map(build_prompts, desc="Building prompts")

# Pre-tokenize all prompts once
tokenized_eng = tokenizer(dataset["eng_prompt"], return_tensors="pt", padding=True, truncation=True)
tokenized_hin = tokenizer(dataset["hin_prompt"], return_tensors="pt", padding=True, truncation=True)

# ============================================================
# 7. BATCHED GENERATION FUNCTION
# ============================================================
def batched_generate(model, tokenized_inputs):
    preds = []
    for i in tqdm(range(0, len(tokenized_inputs["input_ids"]), BATCH_SIZE)):
        batch = {k: v[i:i+BATCH_SIZE].to(DEVICE) for k, v in tokenized_inputs.items()}
        with torch.inference_mode():
            out = model.generate(
                **batch,
                max_new_tokens=MAX_TGT_LEN,
                do_sample=False,
                temperature=0.1,
                repetition_penalty=1.1
            )
        for j in range(out.size(0)):
            gen = out[j, batch["input_ids"].shape[1]:]
            preds.append(tokenizer.decode(gen, skip_special_tokens=True).strip())
    return preds

# ============================================================
# 8. EVALUATE ALL CHECKPOINTS
# ============================================================
summary_rows = []

for ckpt in checkpoints:
    step = int(ckpt.name.split("-")[-1])
    print(f"\n🚀 Evaluating checkpoint-{step}")

    ckpt_out = OUTPUT_ROOT / ckpt.name
    ckpt_out.mkdir(exist_ok=True)

    # Load model (Auto PEFT)
    model = AutoPeftModelForCausalLM.from_pretrained(
        ckpt,
        torch_dtype=torch.bfloat16,
        device_map="auto"
    )
    model.eval()

    # ENG→HIN
    eng_preds = batched_generate(model, tokenized_eng)
    # HIN→ENG
    hin_preds = batched_generate(model, tokenized_hin)

    # Metrics
    e2h_bleu, e2h_chrf = calc_metrics(eng_preds, dataset["tgt_txt"])
    h2e_bleu, h2e_chrf = calc_metrics(hin_preds, dataset["src_txt"])

    summary_rows.append({
        "step": step,
        "ENG_to_HIN_BLEU": e2h_bleu,
        "ENG_to_HIN_chrF": e2h_chrf,
        "HIN_to_ENG_BLEU": h2e_bleu,
        "HIN_to_ENG_chrF": h2e_chrf
    })

    # Save JSONL per checkpoint
    eng_path = ckpt_out / "eng_to_hin.jsonl"
    hin_path = ckpt_out / "hin_to_eng.jsonl"
    with open(eng_path, "w", encoding="utf-8") as fe, open(hin_path, "w", encoding="utf-8") as fh:
        for src, ref, pred in zip(dataset["src_txt"], dataset["tgt_txt"], eng_preds):
            fe.write(json.dumps({"src": src, "ref": ref, "pred": pred}, ensure_ascii=False)+"\n")
        for src, ref, pred in zip(dataset["tgt_txt"], dataset["src_txt"], hin_preds):
            fh.write(json.dumps({"src": src, "ref": ref, "pred": pred}, ensure_ascii=False)+"\n")

    # Save raw results
    raw_results_path = ckpt_out / "raw_results.json"
    with open(raw_results_path, "w", encoding="utf-8") as f:
        json.dump([
            {"mode":"ENG_to_HIN","src":s,"ref":r,"pred":p} for s,r,p in zip(dataset["src_txt"], dataset["tgt_txt"], eng_preds)
        ] + [
            {"mode":"HIN_to_ENG","src":s,"ref":r,"pred":p} for s,r,p in zip(dataset["tgt_txt"], dataset["src_txt"], hin_preds)
        ], f, ensure_ascii=False, indent=2)

    # Cleanup
    del model
    torch.cuda.empty_cache()

# ============================================================
# 9. SAVE METRICS CSV
# ============================================================
df = pd.DataFrame(summary_rows).sort_values("step")
csv_path = OUTPUT_ROOT / "checkpoint_metrics.csv"
df.to_csv(csv_path, index=False)
print(f"\n📊 Metrics CSV saved → {csv_path}")

# ============================================================
# 10. PLOTS
# ============================================================
import matplotlib.pyplot as plt

plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_BLEU"], label="ENG→HIN BLEU")
plt.plot(df["step"], df["HIN_to_ENG_BLEU"], label="HIN→ENG BLEU")
plt.xlabel("Training Step")
plt.ylabel("BLEU")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "bleu_vs_steps.png")
plt.close()

plt.figure()
plt.plot(df["step"], df["ENG_to_HIN_chrF"], label="ENG→HIN chrF")
plt.plot(df["step"], df["HIN_to_ENG_chrF"], label="HIN→ENG chrF")
plt.xlabel("Training Step")
plt.ylabel("chrF")
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_ROOT / "chrf_vs_steps.png")
plt.close()

print("📈 Saved plots: bleu_vs_steps.png, chrf_vs_steps.png")
print("\n✅ ALL CHECKPOINTS EVALUATED (LoRA MERGED, batched, pre-tokenized)")
