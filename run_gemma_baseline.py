"""
ADVISOR-STYLE EVAL: Gemma-4-31B-IT (Google AI Studio API) on NCERT MathDoc-ENHI

Amalgamates:
A) Advisor evaluation structure + metrics (BLEU + CHRF)
B) Google GenAI API inference + masking + generation params

What it does:
- Finds dataset zip/json/jsonl (class11 + class12) under /content or cwd
- Loads + normalizes schema (content_en/content_hi etc.)
- Masks LaTeX + HTML tables into [EQ_k]/[TB_k]
- Calls Gemma via Google GenAI API
- Restores LaTeX/tables verbatim
- Computes BLEU + CHRF + debugging metrics
- Saves:
  - outputs/gemma_eval_api/predictions.jsonl
  - outputs/gemma_eval_api/metrics.json
  - outputs/gemma_eval_api/metrics.csv
  - outputs/gemma_eval_api/length_stats.json

API key:
  export GEMMA_API_KEY="your_key_here"

Run:
  python eval_gemma_api_ncert_mathdoc_enhi.py
"""

# ===============================
# IMPORTS
# ===============================
import os
import re
import json
import time
import math
import shutil
import zipfile
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

# ===============================
# INSTALLS
# ===============================
def pip_install(pkgs: str):
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q"] + pkgs.split())

print("Installing dependencies...")
pip_install("""
google-genai
sacrebleu
tqdm
numpy
pandas
""")
print("✅ Dependencies installed")

import numpy as np
import pandas as pd
import sacrebleu
from tqdm import tqdm
from google import genai
from google.genai import types


# ===============================
# SETTINGS (FILLED — NO GUESSING)
# ===============================

MODEL_ID = "gemma-4-31b-it"

MAX_OUTPUT_TOKENS = 3072  # (B) max_new_tokens equivalent
TEMPERATURE = 1.0
TOP_P = 0.95
TOP_K = 64

SLEEP_BETWEEN_CALLS = 0.5
MAX_RETRIES = 3

OUTPUT_DIR = Path("outputs/gemma_eval_api")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKDIR_CANDIDATES = [Path("/content"), Path.cwd()]  # Colab first, else local


# ===============================
# METRICS (Advisor)
# ===============================
def calc_metrics(preds: List[str], refs: List[str]) -> Tuple[float, float]:
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.corpus_chrf(preds, [refs]).score
    return round(float(bleu), 2), round(float(chrf), 2)

def percentile_stats(arr: List[int]) -> Dict[str, float]:
    a = np.asarray(arr, dtype=np.int64)
    if a.size == 0:
        return {"count": 0}
    return {
        "count": int(a.size),
        "min": int(a.min()),
        "max": int(a.max()),
        "mean": float(np.mean(a)),
        "p50": float(np.percentile(a, 50)),
        "p90": float(np.percentile(a, 90)),
        "p95": float(np.percentile(a, 95)),
        "p99": float(np.percentile(a, 99)),
    }


# ===============================
# MASKING (from B)
# ===============================
_MASK_PATTERNS = [
    (re.compile(r"\$\$[^$]+?\$\$", re.DOTALL), "EQ"),
    (re.compile(r"\\\[.+?\\\]", re.DOTALL), "EQ"),
    (re.compile(
        r"\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}.*?"
        r"\\end\{\1\*?\}", re.DOTALL), "EQ"),
    (re.compile(r"\$[^$\n]+?\$"), "EQ"),
    (re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE), "TB"),
]

_PLACEHOLDER_RE = re.compile(r"\[(EQ|TB)_(\d+)\]")

def mask_spans(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def repl(m, prefix=prefix):
            spans.append(m.group(0))
            return f"[{prefix}_{len(spans) - 1}]"
        masked = pat.sub(repl, masked)
    return masked, spans

def unmask_spans(text: str, spans: List[str]) -> str:
    result = text
    for i, span in enumerate(spans):
        for prefix in ("EQ", "TB"):
            result = result.replace(f"[{prefix}_{i}]", span)
    return result

def extract_placeholders(s: str) -> List[str]:
    return [m.group(0) for m in _PLACEHOLDER_RE.finditer(s)]

def placeholder_ok(masked_input: str, model_output: str) -> Tuple[bool, str]:
    inp = extract_placeholders(masked_input)
    out = extract_placeholders(model_output)
    if inp == out:
        return True, "ok"
    if set(inp) != set(out):
        missing = [p for p in inp if p not in out]
        extra = [p for p in out if p not in inp]
        return False, f"set_mismatch missing={missing[:10]} extra={extra[:10]}"
    return False, "order_mismatch"


# ===============================
# DATA LOADING (zip/json/jsonl bulletproof)
# ===============================
def read_json_any(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))

def read_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows

def normalize_loaded(x: Any) -> List[Dict[str, Any]]:
    if isinstance(x, list):
        return [v for v in x if isinstance(v, dict)]
    if isinstance(x, dict):
        for k in ["data", "instances", "records", "examples"]:
            if k in x and isinstance(x[k], list):
                return [v for v in x[k] if isinstance(v, dict)]
    raise RuntimeError("Unknown dataset schema: expected list or dict containing list.")

def get_src(ex: Dict[str, Any]) -> str:
    for k in ["content_en", "source", "src", "input", "text", "english", "prompt"]:
        if k in ex and ex[k] is not None and str(ex[k]).strip():
            return str(ex[k])
    return ""

def get_tgt(ex: Dict[str, Any]) -> str:
    for k in ["content_hi", "reference", "target", "tgt", "output", "hindi", "completion"]:
        if k in ex and ex[k] is not None and str(ex[k]).strip():
            return str(ex[k])
    return ""

def get_id(ex: Dict[str, Any], fallback_idx: int) -> str:
    for k in ["id", "qid", "uid", "example_id", "idx"]:
        if k in ex and ex[k] is not None and str(ex[k]).strip():
            return str(ex[k])
    return str(fallback_idx)

def find_dataset_files(work: Path, extract_dir: Path) -> Tuple[Optional[Path], Optional[Path]]:
    roots = [work, work / "paperfiles", extract_dir]
    all_files: List[Path] = []
    for root in roots:
        if not root.exists():
            continue
        for p in root.rglob("*"):
            if p.is_file() and p.suffix.lower() in [".json", ".jsonl"]:
                all_files.append(p)

    def rank(p: Path) -> Tuple[int, int]:
        s = str(p).lower()
        pri = 10
        if "class11" in s: pri -= 4
        if "class12" in s: pri -= 4
        if "paperfiles" in s: pri -= 1
        if "dataset_extracted" in s: pri -= 1
        return (pri, len(s))

    ds11 = sorted([p for p in all_files if "class11" in p.name.lower()], key=rank)
    ds12 = sorted([p for p in all_files if "class12" in p.name.lower()], key=rank)

    return (ds11[0] if ds11 else None), (ds12[0] if ds12 else None)

def load_mathdoc_enhi() -> List[Dict[str, Any]]:
    work = None
    for w in WORKDIR_CANDIDATES:
        if w.exists():
            work = w
            break
    if work is None:
        raise RuntimeError("No valid working directory found.")

    extract_dir = work / "dataset_extracted"

    zip_candidates = []
    zip_candidates += list(work.glob("paperfiles*.zip"))
    zip_candidates += list(work.glob("dataset*.zip"))
    zip_candidates += list(work.glob("*.zip"))
    zip_candidates = sorted(set(zip_candidates))

    dataset_zip = zip_candidates[0] if zip_candidates else None

    if dataset_zip is None:
        # If in Colab, prompt upload
        try:
            from google.colab import files
            print("\n⚠️ No ZIP found. Upload your zip now...")
            files.upload()
            zip_candidates = list(work.glob("*.zip"))
            if not zip_candidates:
                raise RuntimeError("No ZIP uploaded.")
            dataset_zip = sorted(zip_candidates)[0]
        except Exception:
            dataset_zip = None

    if dataset_zip is not None:
        print(f"\n✅ Using ZIP: {dataset_zip.name}")
        if extract_dir.exists():
            shutil.rmtree(extract_dir)
        extract_dir.mkdir(parents=True, exist_ok=True)
        print("Extracting ZIP...")
        with zipfile.ZipFile(dataset_zip, "r") as z:
            z.extractall(extract_dir)
        print("✅ Extracted to:", extract_dir)
    else:
        print("\nℹ️ No ZIP used; scanning folders for class11/class12 json/jsonl...")

    ds11, ds12 = find_dataset_files(work, extract_dir)
    if ds11 is None or ds12 is None:
        raise RuntimeError(
            f"Could not locate dataset JSON/JSONL.\n"
            f"class11={ds11}\nclass12={ds12}\n"
            f"Ensure filenames contain 'class11' and 'class12'."
        )

    print("\n✅ DATASETS FOUND")
    print("DS11:", ds11)
    print("DS12:", ds12)

    def load_file(p: Path) -> List[Dict[str, Any]]:
        if p.suffix.lower() == ".jsonl":
            return normalize_loaded(read_jsonl(p))
        return normalize_loaded(read_json_any(p))

    data11 = load_file(ds11)
    data12 = load_file(ds12)
    merged = data11 + data12

    print(f"\n✅ Loaded total records: {len(merged)} (class11={len(data11)}, class12={len(data12)})")

    normalized: List[Dict[str, Any]] = []
    for i, ex in enumerate(merged):
        src = get_src(ex)
        tgt = get_tgt(ex)
        if not src.strip() or not tgt.strip():
            continue
        normalized.append({
            "id": get_id(ex, i),
            "src_txt": src,
            "tgt_txt": tgt,
        })

    if not normalized:
        raise RuntimeError("Dataset is empty after normalization (no src/tgt found).")

    print("✅ Normalized usable records:", len(normalized))
    return normalized


# ===============================
# INFERENCE (Google GenAI API, advisor structure)
# ===============================
SYSTEM_INSTRUCTION = (
    "You are a professional mathematical translator. Translate the following English text into Hindi. "
    "CRITICAL RULE: You will encounter placeholders like [EQ_0], [EQ_1], [TB_0], etc. "
    "Do NOT translate, alter, omit, duplicate, or reorder these placeholders. "
    "Keep them exactly as they are in their correct relative positions in the translated Hindi sentence. "
    "Output ONLY the translated text."
)

def translate_one(client: genai.Client, text: str) -> Tuple[str, Optional[str], bool, str]:
    """
    Returns:
      hyp_text (unmasked), err (or None), placeholder_ok_bool, placeholder_reason
    """
    masked, spans = mask_spans(text)

    last_err = None
    last_out_masked = ""
    last_ph_ok = True
    last_ph_reason = "ok"

    for attempt in range(MAX_RETRIES):
        try:
            resp = client.models.generate_content(
                model=MODEL_ID,
                contents=masked,
                config=types.GenerateContentConfig(
                    system_instruction=SYSTEM_INSTRUCTION,
                    temperature=TEMPERATURE,
                    top_p=TOP_P,
                    top_k=TOP_K,
                    max_output_tokens=MAX_OUTPUT_TOKENS,  # correct name in GenAI SDK
                ),
            )

            out_masked = (resp.text or "").strip()
            last_out_masked = out_masked

            ok, reason = placeholder_ok(masked, out_masked)
            last_ph_ok = ok
            last_ph_reason = reason

            # if placeholders broken, retry
            if not ok:
                last_err = f"PlaceholderIntegrityError: {reason}"
                time.sleep(2 * (attempt + 1))
                continue

            hyp = unmask_spans(out_masked, spans)
            return hyp, None, True, "ok"

        except Exception as e:
            last_err = str(e)
            time.sleep(2 * (attempt + 1))

    # Final fallback: unmask whatever we have (may be broken)
    try:
        hyp = unmask_spans(last_out_masked, spans)
    except Exception as e:
        hyp = last_out_masked
        last_err = last_err or f"UnmaskError: {str(e)}"

    return hyp, last_err, last_ph_ok, last_ph_reason


# ===============================
# MAIN EVAL
# ===============================
def main():
    api_key = os.environ.get("GEMMA_API_KEY", "").strip()
    if not api_key:
        raise SystemExit("Missing GEMMA_API_KEY. Set: export GEMMA_API_KEY=...")

    print("\n==============================")
    print("LOADING DATASET (NCERT MathDoc-ENHI)")
    print("==============================")
    data = load_mathdoc_enhi()

    print("\n==============================")
    print("INITIALIZING GOOGLE GENAI CLIENT")
    print("==============================")
    client = genai.Client(api_key=api_key)

    preds: List[str] = []
    refs: List[str] = []

    out_rows: List[Dict[str, Any]] = []

    # extra debugging metrics
    empty_outputs = 0
    errors = 0
    placeholder_violations = 0
    out_lens = []
    in_lens = []

    print("\n==============================")
    print("RUNNING INFERENCE")
    print("==============================")
    for ex in tqdm(data):
        src = ex["src_txt"]
        ref = ex["tgt_txt"]
        ex_id = ex["id"]

        hyp, err, ph_ok, ph_reason = translate_one(client, src)

        preds.append(hyp)
        refs.append(ref)

        in_lens.append(len(src))
        out_lens.append(len(hyp))

        if not hyp.strip():
            empty_outputs += 1
        if err:
            errors += 1
        if not ph_ok:
            placeholder_violations += 1

        row = {
            "id": ex_id,
            "source": src,
            "reference": ref,
            "hypothesis": hyp,
            "placeholder_ok": ph_ok,
            "placeholder_reason": ph_reason,
        }
        if err:
            row["error"] = err
        out_rows.append(row)

        time.sleep(SLEEP_BETWEEN_CALLS)

    print("\n==============================")
    print("COMPUTING METRICS")
    print("==============================")
    bleu, chrf = calc_metrics(preds, refs)

    length_stats = {
        "input_char_lengths": percentile_stats(in_lens),
        "output_char_lengths": percentile_stats(out_lens),
    }

    summary = {
        "model": MODEL_ID,
        "api": "Google AI Studio / Google GenAI SDK",
        "dataset": "NCERT MathDoc-ENHI (class11+class12)",
        "num_records": len(data),

        # generation params (these were “missing defs” you referenced from code B)
        "generation": {
            "temperature": TEMPERATURE,
            "top_p": TOP_P,
            "top_k": TOP_K,
            "max_output_tokens": MAX_OUTPUT_TOKENS,
        },

        # advisor metrics
        "BLEU": bleu,
        "CHRF": chrf,

        # crucial debugging metrics to avoid “all zeros” confusion
        "debug": {
            "empty_output_rate_pct": round(100.0 * empty_outputs / max(1, len(data)), 4),
            "error_rate_pct": round(100.0 * errors / max(1, len(data)), 4),
            "placeholder_violation_rate_pct": round(100.0 * placeholder_violations / max(1, len(data)), 4),
        }
    }

    # ===============================
    # SAVE OUTPUTS
    # ===============================
    pred_path = OUTPUT_DIR / "predictions.jsonl"
    metrics_path = OUTPUT_DIR / "metrics.json"
    csv_path = OUTPUT_DIR / "metrics.csv"
    length_path = OUTPUT_DIR / "length_stats.json"

    with pred_path.open("w", encoding="utf-8") as f:
        for r in out_rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    metrics_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    length_path.write_text(json.dumps(length_stats, indent=2, ensure_ascii=False), encoding="utf-8")

    pd.DataFrame([{
        "model": summary["model"],
        "BLEU": summary["BLEU"],
        "CHRF": summary["CHRF"],
        "temperature": TEMPERATURE,
        "top_p": TOP_P,
        "top_k": TOP_K,
        "max_output_tokens": MAX_OUTPUT_TOKENS,
        "empty_output_rate_pct": summary["debug"]["empty_output_rate_pct"],
        "error_rate_pct": summary["debug"]["error_rate_pct"],
        "placeholder_violation_rate_pct": summary["debug"]["placeholder_violation_rate_pct"],
        "num_records": summary["num_records"],
    }]).to_csv(csv_path, index=False)

    print("\n============================================================")
    print("✅ EVALUATION COMPLETE")
    print("============================================================")
    print("BLEU:", bleu)
    print("CHRF:", chrf)
    print("\nSaved:")
    print(" -", pred_path)
    print(" -", metrics_path)
    print(" -", csv_path)
    print(" -", length_path)
    print("============================================================\n")


if __name__ == "__main__":
    main()
