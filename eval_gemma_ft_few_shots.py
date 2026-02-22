# ============================================================
# 0. IMPORTS
# ============================================================
import os
import json
import torch
import sacrebleu
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

# ============================================================
# 1. GLOBAL CONFIG
# ============================================================
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)

BASE_MODEL_ID = "google/gemma-3-270m-it"
DATASET_NAME = "ai4bharat/Pralekha"
EVAL_SPLIT = "test"

CHECKPOINT_ROOT = Path("./gemma3_outputs/checkpoints")
OUTPUT_ROOT = Path("./checkpoint_eval_outputs")
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

BATCH_SIZE = 8
MAX_NEW_TOKENS = 3500
FEW_SHOT_K = 2   # MUST MATCH TRAINING

# ============================================================
# 2. METRICS
# ============================================================
def calc_metrics(preds, refs):
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(bleu, 2), round(chrf, 2)

# ============================================================
# 3. LOAD TEST DATA (RAW, UNTOUCHED)
# ============================================================
print("Loading Pralekha TEST split...")
test_set = load_dataset(DATASET_NAME, EVAL_SPLIT, split="eng_hin")

# ============================================================
# 4. TOKENIZER (IDENTICAL TO TRAINING)
# ============================================================
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
tokenizer.pad_token = tokenizer.eos_token

# ============================================================
# 5. DISCOVER CHECKPOINTS
# ============================================================
checkpoints = sorted(
    [p for p in CHECKPOINT_ROOT.iterdir() if p.name.startswith("checkpoint-")],
    key=lambda x: int(x.name.split("-")[-1]),
)

assert checkpoints, "No checkpoints found"
print(f" Found {len(checkpoints)} checkpoints")

summary_rows = []

# ============================================================
# 6. MAIN CHECKPOINT LOOP
# ============================================================
for ckpt in checkpoints:
    step = int(ckpt.name.split("-")[-1])
    print(f"\n Evaluating checkpoint-{step}")

    ckpt_out = OUTPUT_ROOT / ckpt.name
    ckpt_out.mkdir(exist_ok=True)

    # --------------------------------------------------------
    # 6.1 LOAD BASE + LORA → MERGE
    # --------------------------------------------------------
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL_ID,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
        attn_implementation="sdpa",
    )

    model = PeftModel.from_pretrained(base_model, ckpt)
    model = model.merge_and_unload()
    model.eval()

    all_preds = {"ENG_to_HIN": [], "HIN_to_ENG": []}
    all_refs  = {"ENG_to_HIN": [], "HIN_to_ENG": []}
    jsonl_rows = []

    # --------------------------------------------------------
    # 6.2 FEW-SHOT + BIDIRECTIONAL PROMPT-PARITY INFERENCE
    # --------------------------------------------------------
    for i in tqdm(range(0, len(test_set), BATCH_SIZE), desc="Inference"):
        batch = test_set[i:i + BATCH_SIZE]

        prompts, modes, refs, srcs = [], [], [], []

        for idx in range(len(batch["src_txt"])):

            for mode in ["ENG_to_HIN", "HIN_to_ENG"]:

                if mode == "ENG_to_HIN":
                    src_lang, tgt_lang = "English", "Hindi"
                    src = batch["src_txt"][idx]
                    ref = batch["tgt_txt"][idx]
                    ex_srcs = test_set["src_txt"]
                    ex_tgts = test_set["tgt_txt"]
                else:
                    src_lang, tgt_lang = "Hindi", "English"
                    src = batch["tgt_txt"][idx]
                    ref = batch["src_txt"][idx]
                    ex_srcs = test_set["tgt_txt"]
                    ex_tgts = test_set["src_txt"]

                # ---- build few-shot examples (deterministic) ----
                examples = []
                base_idx = i + idx
                for j in range(1, FEW_SHOT_K + 1):
                    ex_idx = (base_idx - j) % len(test_set)
                    examples.append(
                        f"{src_lang}: {ex_srcs[ex_idx]}\n\n{tgt_lang}: {ex_tgts[ex_idx]}"
                    )

                instruction = (
                    f"Translate the given input document to {tgt_lang}. "
                    "Generate only the translation. Do not generate any other tokens."
                )

                prompt = (
                    f"<start_of_turn>user\n"
                    f"{instruction}\n\n"
                    + "\n\n".join(examples)
                    + f"\n\n{src_lang}: {src}\n\n{tgt_lang}:"
                    f"<end_of_turn>\n"
                    f"<start_of_turn>model\n"
                )

                prompts.append(prompt)
                modes.append(mode)
                refs.append(ref)
                srcs.append(src)

        inputs = tokenizer(
            prompts,
            truncation=True,
            padding=False,
            return_tensors="pt",
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True,
            )

        gen_tokens = outputs[:, inputs.input_ids.shape[1]:]
        decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

        for mode, pred, ref, src in zip(modes, decoded, refs, srcs):
            pred = pred.strip()
            all_preds[mode].append(pred)
            all_refs[mode].append(ref)

            jsonl_rows.append({
                "mode": mode,
                "src": src,
                "ref": ref,
                "pred": pred,
            })

    # --------------------------------------------------------
    # 6.3 METRICS
    # --------------------------------------------------------
    e2h_bleu, e2h_chrf = calc_metrics(
        all_preds["ENG_to_HIN"], all_refs["ENG_to_HIN"]
    )
    h2e_bleu, h2e_chrf = calc_metrics(
        all_preds["HIN_to_ENG"], all_refs["HIN_to_ENG"]
    )

    summary_rows.append({
        "step": step,
        "ENG_to_HIN_BLEU": e2h_bleu,
        "ENG_to_HIN_chrF": e2h_chrf,
        "HIN_to_ENG_BLEU": h2e_bleu,
        "HIN_to_ENG_chrF": h2e_chrf,
    })

    # --------------------------------------------------------
    # 6.4 SAVE JSONL
    # --------------------------------------------------------
    with open(ckpt_out / "eng_to_hin.jsonl", "w", encoding="utf-8") as fe, \
         open(ckpt_out / "hin_to_eng.jsonl", "w", encoding="utf-8") as fh:

        for r in jsonl_rows:
            line = json.dumps(
                {"src": r["src"], "ref": r["ref"], "pred": r["pred"]},
                ensure_ascii=False,
            )
            if r["mode"] == "ENG_to_HIN":
                fe.write(line + "\n")
            else:
                fh.write(line + "\n")

    del model, base_model
    torch.cuda.empty_cache()

# ============================================================
# 7. SAVE METRICS + PLOTS
# ============================================================
df = pd.DataFrame(summary_rows).sort_values("step")
df.to_csv(OUTPUT_ROOT / "checkpoint_metrics.csv", index=False)

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

print("\n ALL CHECKPOINTS EVALUATED (FEW-SHOT + BIDIRECTIONAL)")
print(" Metrics → checkpoint_metrics.csv")
print(" Plots saved")
