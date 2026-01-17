# ============================================================
# BACK-TRANSLATION FOR FINETUNED GEMMA MODEL
# ============================================================

import torch
from tqdm import tqdm
import sacrebleu
import pandas as pd
import numpy as np

# Ensure model and tokenizer are loaded
# model = ... (your fine-tuned Gemma3)
# tokenizer = ... (your fine-tuned tokenizer)
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)
model.eval()

# ============================================================
# GREEDY BACK-TRANSLATION FUNCTION
# ============================================================
def back_translate_audit(results, model, tokenizer):
    audit_results = []

    print("🔁 Back-translation started")
    for r in tqdm(results):
        r["back_translation"] = "N/A"
        r["bt_consistency_chrf"] = 0.0

        if r["mode"] == "ENG_to_HIN":
            # HIN → ENG back-translation
            instr = "Translate to ENGLISH:"
            src = r["prediction"]
            ref = r["source"]
        elif r["mode"] == "HIN_to_ENG":
            # ENG → HIN back-translation
            instr = "Translate to HINDI DEVANAGARI:"
            src = r["prediction"]
            ref = r["source"]
        else:
            audit_results.append(r)
            continue

        # Skip empty predictions
        if not src.strip():
            audit_results.append(r)
            continue

        # Match exactly the fine-tuned training prompt
        prompt = f"<start_of_turn>user\n{instr}\n{src}<end_of_turn>\n<start_of_turn>model\n"

        inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True).to(device)

        with torch.no_grad():
            output = model.generate(
                **inputs,
                max_new_tokens=512,
                temperature=0.1,
                do_sample=False,
                repetition_penalty=1.1
            )

        # Extract only newly generated tokens
        pred_tokens = output[0][inputs.input_ids.shape[-1]:]
        bt_text = tokenizer.decode(pred_tokens, skip_special_tokens=True).strip()

        # Save results
        r["back_translation"] = bt_text
        r["bt_consistency_chrf"] = round(sacrebleu.sentence_chrf(bt_text, [ref]).score, 2)

        audit_results.append(r)

    print("✅ Back-translation completed")
    return audit_results

# ============================================================
# EXECUTE AUDIT
# ============================================================
audit_results = back_translate_audit(results, model, tokenizer)

# ============================================================
# SUMMARY CALCULATION
# ============================================================
summary_rows = []
for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [res for res in audit_results if res["mode"] == mode]
    preds = [s["prediction"] for s in subset]
    refs = [s["reference"] for s in subset]

    doc_bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    doc_chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    bt_scores = [s["bt_consistency_chrf"] for s in subset if s["back_translation"] != "N/A"]
    avg_bt = np.mean(bt_scores) if bt_scores else 0.0

    summary_rows.append({
        "Direction": mode,
        "Doc_BLEU": round(doc_bleu, 2),
        "Doc_chrF": round(doc_chrf, 2),
        "BT_Consistency": round(avg_bt, 2)
    })

print("\n" + "="*50)
print("FINAL AUDIT SUMMARY")
display(pd.DataFrame(summary_rows))
print("="*50)

# ============================================================
# TOP-10 QUALITATIVE CHECK
# ============================================================
df_bt = pd.DataFrame(audit_results)
print("\n🧪 TOP-10 BACK-TRANSLATION SAMPLES\n")
for i in range(min(10, len(df_bt))):
    r = df_bt.iloc[i]
    print(f"[{i+1}] {r['mode']}")
    print("SRC :", r["source"])
    print("REF :", r["reference"])
    print("PRED:", r["prediction"])
    print("BT  :", r["back_translation"])
    print("Consistency chrF:", r["bt_consistency_chrf"])
    print("-"*80)
# ============================================================
# SAVE BACK-TRANSLATION RESULTS: JSONL + ZIP + XLSX
# ============================================================

import json
from pathlib import Path
import shutil
import pandas as pd
import numpy as np
import sacrebleu

# -----------------------------
# 1️⃣ Create output directory
# -----------------------------
out_dir = Path("bt_audit_outputs")
out_dir.mkdir(exist_ok=True)

eng_path = out_dir / "eng_to_hin.jsonl"
hin_path = out_dir / "hin_to_eng.jsonl"

# -----------------------------
# 2️⃣ Write JSONL for both directions
# -----------------------------
eng_count = 0
hin_count = 0

with open(eng_path, "w", encoding="utf-8") as fe, open(hin_path, "w", encoding="utf-8") as fh:
    for r in audit_results:
        line = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"],
            "back_translation": r.get("back_translation", "N/A"),
            "bt_chrF": r.get("bt_consistency_chrf", 0.0)
        }
        if r["mode"] == "ENG_to_HIN":
            fe.write(json.dumps(line, ensure_ascii=False) + "\n")
            eng_count += 1
        elif r["mode"] == "HIN_to_ENG":
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            hin_count += 1

print(f"✅ JSONL saved: ENG→HIN={eng_count}, HIN→ENG={hin_count}")

# -----------------------------
# 3️⃣ Zip JSONL files
# -----------------------------
zip_name = "translation_bt_outputs"
shutil.make_archive(base_name=zip_name, format="zip", root_dir=out_dir)
print(f"✅ JSONL ZIP created: {zip_name}.zip")

# -----------------------------
# 4️⃣ Compute BLEU/chrF/BT-BLEU per direction
# -----------------------------
summary_rows = []
for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [r for r in audit_results if r["mode"] == mode]
    preds = [r["prediction"] for r in subset]
    refs = [r["reference"] for r in subset]
    bt_texts = [r["back_translation"] for r in subset if r.get("back_translation", "N/A") != "N/A"]

    doc_bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    doc_chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    bt_chrF_avg = np.mean([r.get("bt_consistency_chrf", 0.0) for r in subset]) if subset else 0.0
    bt_bleu = sacrebleu.corpus_bleu(bt_texts, [refs]).score if bt_texts else 0.0

    summary_rows.append({
        "Direction": mode,
        "Doc_BLEU": round(doc_bleu, 2),
        "Doc_chrF": round(doc_chrf, 2),
        "Avg_BT_Consistency_chrF": round(bt_chrF_avg, 2),
        "Avg_BT_BLEU": round(bt_bleu, 2)
    })

# -----------------------------
# 5️⃣ Save Excel with summary
# -----------------------------
xlsx_path = Path("bt_audit_summary.xlsx")

# Attempt to use openpyxl (works in Colab & VS Code)
try:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        pd.DataFrame(summary_rows).to_excel(writer, sheet_name="Summary", index=False)
        pd.DataFrame(audit_results).to_excel(writer, sheet_name="Detailed_Samples", index=False)
except ModuleNotFoundError:
    print("⚠️ 'openpyxl' not installed. Run: pip install openpyxl")

print(f"✅ XLSX saved: {xlsx_path.resolve()}")
# ============================================================
# BACK-TRANSLATION METRICS CALCULATION
# ============================================================

import sacrebleu
import numpy as np
import pandas as pd

bt_summary = []

for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [r for r in audit_results if r["mode"] == mode]
    refs = [r["source"] for r in subset]  # original source is reference for BT
    bt_texts = [r["back_translation"] for r in subset if r.get("back_translation", "N/A") != "N/A"]

    if bt_texts:
        bt_bleu = sacrebleu.corpus_bleu(bt_texts, [refs]).score
        bt_chrf = sacrebleu.corpus_chrf(bt_texts, [refs]).score
    else:
        bt_bleu = 0.0
        bt_chrf = 0.0

    bt_summary.append({
        "Direction": mode,
        "BT_BLEU": round(bt_bleu, 2),
        "BT_chrF": round(bt_chrf, 2),
        "Num_Samples": len(bt_texts)
    })

df_bt_metrics = pd.DataFrame(bt_summary)
print("\n📝 BACK-TRANSLATION METRICS (Both Directions)\n")
display(df_bt_metrics)
# ============================================================
# SAVE JSONL FILES & BT METRICS TO ZIP + XLSX
# ============================================================

import json
from pathlib import Path
import shutil

# --- Create output directories ---
out_dir = Path("bt_audit_outputs")
out_dir.mkdir(exist_ok=True)

eng_jsonl_path = out_dir / "eng_to_hin_src_ref_pred.jsonl"
hin_jsonl_path = out_dir / "hin_to_eng_src_ref_pred.jsonl"

# --- Save JSONL files ---
eng_count, hin_count = 0, 0
with open(eng_jsonl_path, "w", encoding="utf-8") as fe, \
     open(hin_jsonl_path, "w", encoding="utf-8") as fh:

    for r in audit_results:
        line = {
            "src": r["source"],
            "ref": r["reference"],
            "pred": r["prediction"],
            "bt": r.get("back_translation", "N/A"),
            "bt_chrF": r.get("bt_consistency_chrf", 0.0)
        }
        if r["mode"] == "ENG_to_HIN":
            fe.write(json.dumps(line, ensure_ascii=False) + "\n")
            eng_count += 1
        elif r["mode"] == "HIN_to_ENG":
            fh.write(json.dumps(line, ensure_ascii=False) + "\n")
            hin_count += 1

print(f"✅ JSONL files saved: ENG→HIN ({eng_count}) | HIN→ENG ({hin_count})")

# --- Zip the JSONL files ---
zip_name = "bt_audit_jsonl_outputs"
shutil.make_archive(zip_name, "zip", root_dir=out_dir)
print(f"📦 JSONL files zipped: {zip_name}.zip")

# --- Combine summary metrics (forward + BT) ---
# Forward translation BLEU/chrF from previous audit_summary
forward_summary_rows = []
for mode in ["ENG_to_HIN", "HIN_to_ENG"]:
    subset = [r for r in audit_results if r["mode"] == mode]
    preds = [r["prediction"] for r in subset]
    refs = [r["reference"] for r in subset]

    doc_bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    doc_chrf = sacrebleu.corpus_chrf(preds, [refs]).score

    bt_scores = [r.get("bt_consistency_chrf", 0.0) for r in subset if r.get("back_translation", "N/A") != "N/A"]
    avg_bt = np.mean(bt_scores) if bt_scores else 0.0

    forward_summary_rows.append({
        "Direction": mode,
        "Doc_BLEU": round(doc_bleu, 2),
        "Doc_chrF": round(doc_chrf, 2),
        "Avg_BT_Consistency_chrF": round(avg_bt, 2),
        "Num_Samples": len(subset)
    })

df_summary = pd.DataFrame(forward_summary_rows)

# --- Save as XLSX ---
try:
    import xlsxwriter
except ModuleNotFoundError:
    !pip install xlsxwriter

xlsx_path = Path("bt_audit_summary.xlsx")
with pd.ExcelWriter(xlsx_path, engine="xlsxwriter") as writer:
    df_summary.to_excel(writer, sheet_name="Summary", index=False)
    pd.DataFrame(audit_results).to_excel(writer, sheet_name="Detailed_Samples", index=False)

print(f"✅ XLSX saved: {xlsx_path.resolve()}")
