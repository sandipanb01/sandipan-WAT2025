"""
run_gemma4_mathdoc_ADVISOR_NOQUANT_SACREBLEU_DEFAULT.py
======================================================
End-to-end HuggingFace inference + advisor-grade evaluation for MathDoc-ENHI.

Changes requested:
✅ Uses sacrebleu "original defaults" (NO flores / no custom tokenizer)
✅ NO quantisation (no 4-bit / 8-bit). Pure HF load with dtype.
✅ Keeps MSP/MCP/TSP/TCP SFS suite
✅ Keeps masking [EQ_k]/[TB_k] restoration
✅ Fixes Gemma tokenizer crash: 'list' object has no attribute 'keys'
✅ Resume support + strict path checks

Outputs:
- JSONL hypotheses: CONFIG["HYPOTHESIS_JSONL"]
- JSON results:     CONFIG["RESULTS_JSON"]
"""

# =============================================================================
# SECTION 0 — Imports
# =============================================================================
import gc
import json
import os
import re
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from tqdm import tqdm

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# =============================================================================
# SECTION 1 — CONFIG  (edit here; no command-line args needed)
# =============================================================================
CONFIG = {
    # ── Paths ──────────────────────────────────────────────────────────────────
    # Put absolute paths here if you want. Relative paths are resolved from cwd.
    "DATA_PATHS": [
        "ncert_class11_math_en_hi_test_instances_curated_FIXED.json",
        "ncert_class12_math_en_hi_test_instances_curated.json",
    ],
    "OUTPUT_DIR": "outputs/gemma4",
    "HYPOTHESIS_JSONL": "outputs/gemma4/gemma4_hypotheses.jsonl",
    "RESULTS_JSON":     "outputs/gemma4/gemma4_results.json",

    # ── Model ──────────────────────────────────────────────────────────────────
    "MODEL_ID": "google/gemma-4-31b-it",

    # Compute dtype: "bfloat16" | "float16" | "float32"
    "DTYPE": "bfloat16",

    # ── Inference ──────────────────────────────────────────────────────────────
    "BATCH_SIZE":        4,         # reduce if OOM
    "MAX_NEW_TOKENS":    3072,
    "DO_SAMPLE":         False,
    "TEMPERATURE":       1.0,       # only used if DO_SAMPLE=True
    "TOP_P":             0.95,
    "TOP_K":             64,

    # ── Sanity check ───────────────────────────────────────────────────────────
    "SANITY_CHECK":      False,
    "SANITY_N":          5,

    # ── Optional COMET ─────────────────────────────────────────────────────────
    "SKIP_COMET":        True,
    "COMET_MODEL":       "Unbabel/wmt22-comet-da",
    "COMET_BATCH_SIZE":  8,
    "COMET_GPUS":        1,

    # ── Misc ───────────────────────────────────────────────────────────────────
    "SEED":              42,
    "SLEEP_BETWEEN_BATCHES": 0.0,
    "MAX_RETRIES":       3,
    "EXPECTED_TOTAL":    821,
}

# =============================================================================
# SECTION 2 — Logging
# =============================================================================
def log(msg: str, level: str = "INFO") -> None:
    prefix = {"INFO": "ℹ️", "WARN": "⚠️", "ERROR": "✗", "OK": "✓"}
    ts = time.strftime("%H:%M:%S")
    print(f"[{ts}] {prefix.get(level, '·')} {msg}", flush=True)

def section(title: str) -> None:
    bar = "=" * 72
    print(f"\n{bar}\n  {title}\n{bar}", flush=True)

# =============================================================================
# SECTION 3 — GPU utilities
# =============================================================================
def free_gpu() -> None:
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

def get_device_info() -> str:
    if not torch.cuda.is_available():
        return "CPU only"
    gpus = [
        f"{torch.cuda.get_device_name(i)} "
        f"({torch.cuda.get_device_properties(i).total_memory // (1024**3)} GB)"
        for i in range(torch.cuda.device_count())
    ]
    return "  |  ".join(gpus)

def resolve_dtype(dtype_str: str) -> torch.dtype:
    mapping = {
        "bfloat16": torch.bfloat16,
        "float16":  torch.float16,
        "float32":  torch.float32,
    }
    if dtype_str not in mapping:
        raise ValueError(f"Unknown dtype '{dtype_str}'. Choose: {list(mapping)}")
    return mapping[dtype_str]

# =============================================================================
# SECTION 4 — Data loading & validation  (inlined)
# =============================================================================
class DataValidationError(ValueError):
    pass

def _coerce_record_list(obj, path: str) -> List[Dict]:
    if isinstance(obj, list):
        records = obj
    elif isinstance(obj, dict) and isinstance(obj.get("instances"), list):
        records = obj["instances"]
    elif isinstance(obj, dict):
        records = [obj]
    else:
        raise DataValidationError(f"{path}: expected JSON array/object/JSONL — got {type(obj)}")
    bad = [i for i, r in enumerate(records) if not isinstance(r, dict)]
    if bad:
        raise DataValidationError(f"{path}: records must be objects; first bad index={bad[0]}")
    return records

def _load_one_file(path: str) -> List[Dict]:
    with open(path, encoding="utf-8") as f:
        text = f.read()
    if not text.strip():
        raise DataValidationError(f"{path}: file is empty")
    try:
        return _coerce_record_list(json.loads(text), path)
    except json.JSONDecodeError as whole_err:
        # JSONL fallback
        records = []
        for lineno, line in enumerate(text.splitlines(), 1):
            if not line.strip():
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as le:
                raise DataValidationError(
                    f"{path}:{lineno}: invalid JSON/JSONL "
                    f"(whole-file err: {whole_err}; line err: {le})"
                ) from le
            if not isinstance(obj, dict):
                raise DataValidationError(f"{path}:{lineno}: JSONL records must be objects")
            records.append(obj)
        if not records:
            raise DataValidationError(f"{path}: no records found")
        return records

def _resolve_path(p: str) -> str:
    # Thorough path resolution: expand user/env, then resolve relative to cwd
    p = os.path.expandvars(os.path.expanduser(p))
    if os.path.isabs(p):
        return p
    return str(Path.cwd() / p)

def load_dataset_records(paths, expected_total: Optional[int] = None) -> List[Dict]:
    required = ("id", "content_en", "content_hi")
    all_records: List[Dict] = []

    path_list = [paths] if isinstance(paths, (str, Path)) else list(paths)
    if not path_list:
        raise DataValidationError("DATA_PATHS is empty.")

    for p in path_list:
        path_str = _resolve_path(str(p))
        if not os.path.exists(path_str):
            raise DataValidationError(f"Dataset file not found: {path_str}")
        batch = _load_one_file(path_str)
        log(f"Loaded {len(batch):4d} records from {path_str}")
        all_records.extend(batch)

    # Field check
    for idx, rec in enumerate(all_records):
        missing = [f for f in required if f not in rec]
        if missing:
            raise DataValidationError(f"Record {idx} missing fields: {missing}")
        empty = [f for f in required if not str(rec.get(f, "")).strip()]
        if empty:
            raise DataValidationError(f"Record {rec.get('id', idx)!r} has empty fields: {empty}")

    # Duplicate id check
    seen, dupes = set(), []
    for rec in all_records:
        rid = str(rec["id"])
        if rid in seen:
            dupes.append(rid)
        seen.add(rid)
    if dupes:
        raise DataValidationError(f"Duplicate id(s): {dupes[:10]} ...")

    if expected_total is not None and len(all_records) != expected_total:
        raise DataValidationError(f"Expected {expected_total} records, found {len(all_records)}")

    return all_records

# =============================================================================
# SECTION 5 — Masking  (includes <math> tags + HTML/MD tables)
# =============================================================================
_MASK_PATTERNS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\$\$[^$]+?\$\$", re.DOTALL), "EQ"),
    (re.compile(
        r"\\begin\{(equation|align|gather|eqnarray|"
        r"bmatrix|pmatrix|vmatrix|matrix|smallmatrix)\*?\}"
        r".*?"
        r"\\end\{\1\*?\}", re.DOTALL), "EQ"),
    (re.compile(r"<math[^>]*>.*?</math>", re.DOTALL | re.IGNORECASE), "EQ"),
    (re.compile(r"\$[^$\n]+?\$"), "EQ"),
    (re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE), "TB"),
    (re.compile(r"(?:^|\n)((?:\|[^\n]+\|[\t ]*\n){2,})", re.MULTILINE), "TB"),
]

def mask_spans(text: str) -> Tuple[str, List[str]]:
    spans: List[str] = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def _repl(m, prefix=prefix):
            idx = len(spans)
            spans.append(m.group(0))
            return f"[{prefix}_{idx}]"
        masked = pat.sub(_repl, masked)
    return masked, spans

def unmask_spans(text: str, spans: List[str]) -> str:
    result = text
    for i in range(len(spans) - 1, -1, -1):
        for prefix in ("EQ", "TB"):
            ph = f"[{prefix}_{i}]"
            if ph in result:
                result = result.replace(ph, spans[i])
    return result

# =============================================================================
# SECTION 6 — SFS Suite  (MSP/MCP/TSP/TCP)
# =============================================================================
_EQ_PATTERNS_SFS: List[Tuple[re.Pattern, str]] = [
    (re.compile(r"\$\$([^$]+?)\$\$", re.DOTALL), "display_dollar"),
    (re.compile(r"\$([^$\n]+?)\$", re.DOTALL), "inline_dollar"),
    (re.compile(
        r"\\begin\{(equation|align|gather|eqnarray|"
        r"bmatrix|pmatrix|vmatrix|matrix|smallmatrix)\*?\}"
        r"(.+?)"
        r"\\end\{\1\*?\}", re.DOTALL), "environment"),
    (re.compile(r"<math[^>]*>(.*?)</math>", re.DOTALL | re.IGNORECASE), "mathml"),
]

def extract_equations(text: str) -> List[str]:
    occupied: List[Tuple[int, int]] = []
    spans: List[Tuple[int, str]] = []
    for pat, kind in _EQ_PATTERNS_SFS:
        for m in pat.finditer(text):
            s, e = m.span()
            if any(s < oe and e > os for os, oe in occupied):
                continue
            if kind == "environment":
                content = m.group(2)
            elif kind == "mathml":
                content = m.group(1)
            else:
                content = m.group(1)
            spans.append((s, content.strip()))
            occupied.append((s, e))
    return [c for _, c in sorted(spans)]

def count_equations(text: str) -> int:
    return len(extract_equations(text))

def normalise_latex(s: str) -> str:
    return re.sub(r"\s+", "", s).strip()

def metric_msp(ref: str, hyp: str) -> Optional[float]:
    n_ref = count_equations(ref)
    if n_ref == 0:
        return None
    return min(1.0, count_equations(hyp) / n_ref)

def metric_mcp(ref: str, hyp: str) -> Optional[float]:
    ref_eqs = extract_equations(ref)
    if not ref_eqs:
        return None
    hyp_norm = normalise_latex(hyp)
    hits = sum(1 for eq in ref_eqs if normalise_latex(eq) in hyp_norm)
    return hits / len(ref_eqs)

_HTML_TABLE_RE = re.compile(r"<table[^>]*>(.*?)</table>", re.DOTALL | re.IGNORECASE)
_HTML_ROW_RE   = re.compile(r"<tr[^>]*>(.*?)</tr>", re.DOTALL | re.IGNORECASE)
_HTML_CELL_RE  = re.compile(r"<t[dh][^>]*>(.*?)</t[dh]>", re.DOTALL | re.IGNORECASE)
_HTML_TAG_RE   = re.compile(r"<[^>]+>")
_MD_TABLE_RE   = re.compile(r"(?:^|\n)((?:\|[^\n]+\|[\t ]*\n){2,})", re.MULTILINE)

def _parse_html_table(inner_html: str) -> Dict:
    rows: List[List[str]] = []
    for row_html in _HTML_ROW_RE.findall(inner_html):
        cells = [_HTML_TAG_RE.sub("", c).strip() for c in _HTML_CELL_RE.findall(row_html)]
        rows.append(cells)
    if not rows:
        return {"rows": 0, "cols": 0, "cells": [], "format": "html"}
    cols = max(len(r) for r in rows)
    return {"rows": len(rows), "cols": cols, "cells": [c for r in rows for c in r], "format": "html"}

def _parse_md_table(block: str) -> Optional[Dict]:
    lines = [ln for ln in block.strip().split("\n") if ln.strip()]
    if len(lines) < 2:
        return None
    def cells(line):
        return [c.strip() for c in line.strip().strip("|").split("|")]
    rows = [cells(lines[0])]
    start = 1
    if start < len(lines) and re.match(r"^[\s|:\-]+$", lines[start]):
        start += 1
    rows.extend(cells(ln) for ln in lines[start:])
    if not rows:
        return None
    return {"rows": len(rows), "cols": max(len(r) for r in rows), "cells": [c for r in rows for c in r], "format": "md"}

def extract_tables(text: str) -> List[Dict]:
    tables = [_parse_html_table(m.group(1)) for m in _HTML_TABLE_RE.finditer(text)]
    for m in _MD_TABLE_RE.finditer(text):
        parsed = _parse_md_table(m.group(1))
        if parsed:
            tables.append(parsed)
    return tables

def _shape_score(t_ref: Dict, t_hyp: Dict) -> float:
    if t_ref["rows"] == 0 or t_ref["cols"] == 0:
        return 0.0
    row_r = min(t_ref["rows"], t_hyp["rows"]) / max(t_ref["rows"], t_hyp["rows"], 1)
    col_r = min(t_ref["cols"], t_hyp["cols"]) / max(t_ref["cols"], t_hyp["cols"], 1)
    return (row_r + col_r) / 2.0

def metric_tsp(ref: str, hyp: str) -> Optional[float]:
    ref_tables = extract_tables(ref)
    if not ref_tables:
        return None
    hyp_tables = extract_tables(hyp)
    scores = [_shape_score(t_ref, hyp_tables[i]) if i < len(hyp_tables) else 0.0
              for i, t_ref in enumerate(ref_tables)]
    return sum(scores) / len(scores)

_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")

def _numeric_cells(table: Dict) -> List[str]:
    return [n for cell in table["cells"] for n in _NUMBER_RE.findall(cell)]

def metric_tcp(ref: str, hyp: str) -> Optional[float]:
    ref_tables = extract_tables(ref)
    if not ref_tables:
        return None
    ref_nums = [n for t in ref_tables for n in _numeric_cells(t)]
    if not ref_nums:
        return None
    hyp_tables = extract_tables(hyp)
    hyp_nums = [n for t in hyp_tables for n in _numeric_cells(t)]
    ref_c, hyp_c = Counter(ref_nums), Counter(hyp_nums)
    preserved = sum(min(ref_c[n], hyp_c.get(n, 0)) for n in ref_c)
    return preserved / sum(ref_c.values())

def compute_sfs(refs: List[str], hyps: List[str]) -> Dict:
    buckets = {"MSP": [], "MCP": [], "TSP": [], "TCP": []}
    for ref, hyp in zip(refs, hyps):
        for key, fn in [("MSP", metric_msp), ("MCP", metric_mcp), ("TSP", metric_tsp), ("TCP", metric_tcp)]:
            v = fn(ref, hyp)
            if v is not None:
                buckets[key].append(v)
    def _mean(xs): return sum(xs)/len(xs) if xs else 0.0
    return {
        "MSP": _mean(buckets["MSP"]),
        "MCP": _mean(buckets["MCP"]),
        "TSP": _mean(buckets["TSP"]),
        "TCP": _mean(buckets["TCP"]),
        "n_math": len(buckets["MSP"]),
        "n_tables": len(buckets["TSP"]),
        "n_tcp": len(buckets["TCP"]),
    }

# =============================================================================
# SECTION 7 — Standard metrics (sacrebleu defaults, as requested)
# =============================================================================
def compute_bleu_chrf_default(refs: List[str], hyps: List[str]) -> Dict[str, float]:
    """
    Uses sacrebleu "original defaults":
      - corpus_bleu(hyps, [refs])  (default tokenizer internally)
      - corpus_chrf(hyps, [refs])  (default chrF)
    """
    import sacrebleu
    bleu = sacrebleu.corpus_bleu(hyps, [refs]).score
    chrf = sacrebleu.corpus_chrf(hyps, [refs]).score
    return {"BLEU": round(float(bleu), 4), "chrF": round(float(chrf), 4)}

# =============================================================================
# SECTION 8 — Category
# =============================================================================
def categorise_instance(ref_hi: str) -> str:
    has_math = count_equations(ref_hi) > 0
    has_tables = len(extract_tables(ref_hi)) > 0
    if has_tables:
        return "math_tables"
    if has_math:
        return "math_only"
    return "pure_text"

# =============================================================================
# SECTION 9 — Tokenizer crash fix + model loading (NO quant)
# =============================================================================
def safe_load_tokenizer(model_id: str):
    """
    Fixes Gemma tokenizer crash:
      AttributeError: 'list' object has no attribute 'keys'

    Strategy:
      1) try fast tokenizer
      2) try slow tokenizer
      3) slow tokenizer + force extra_special_tokens=[]
    """
    from transformers import AutoTokenizer

    # 1) fast
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=True)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok
    except Exception as e1:
        log(f"Tokenizer fast load failed: {e1}", "WARN")

    # 2) slow
    try:
        tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
        if tok.pad_token is None:
            tok.pad_token = tok.eos_token
        tok.padding_side = "left"
        return tok
    except Exception as e2:
        log(f"Tokenizer slow load failed: {e2}", "WARN")

    # 3) patch
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True, use_fast=False)
    try:
        tok.extra_special_tokens = []
    except Exception:
        pass
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    tok.padding_side = "left"
    return tok

def load_model_and_tokenizer(cfg: Dict):
    from transformers import AutoModelForCausalLM

    model_id = cfg["MODEL_ID"]
    dtype = resolve_dtype(cfg["DTYPE"])

    log(f"Model    : {model_id}")
    log(f"dtype    : {cfg['DTYPE']}")
    log(f"GPU(s)   : {get_device_info()}")

    log("Loading tokenizer (Gemma-safe fallbacks)...")
    tokenizer = safe_load_tokenizer(model_id)

    log("Loading model weights (NO quantisation)...")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        device_map="auto",
        torch_dtype=dtype,
        trust_remote_code=True,
    )
    model.eval()
    log("Model loaded successfully.", "OK")
    return model, tokenizer

# =============================================================================
# SECTION 10 — Prompt + Inference
# =============================================================================
SYSTEM_PROMPT = (
    "You are a professional mathematical translator specialised in NCERT mathematics. "
    "Translate English to Hindi.\n\n"
    "CRITICAL RULE: You will see placeholders like [EQ_0], [TB_0]. "
    "Do NOT modify, remove, duplicate, or reorder placeholders. "
    "Output ONLY the translated Hindi text."
)

def build_chat_prompt(masked_src: str, tokenizer) -> str:
    msgs = [{"role": "user", "content": f"{SYSTEM_PROMPT}\n\nEnglish:\n{masked_src}"}]
    try:
        return tokenizer.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"<start_of_turn>user\n{SYSTEM_PROMPT}\n\nEnglish:\n{masked_src}\n<end_of_turn>\n<start_of_turn>model\n"

def run_inference_on_batch(model, tokenizer, prompts: List[str], cfg: Dict) -> List[str]:
    encoded = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=8192,
    )
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)

    gen_kwargs = dict(
        input_ids=input_ids,
        attention_mask=attention_mask,
        max_new_tokens=cfg["MAX_NEW_TOKENS"],
        do_sample=cfg["DO_SAMPLE"],
        use_cache=True,
        pad_token_id=tokenizer.pad_token_id,
        eos_token_id=tokenizer.eos_token_id,
    )
    if cfg["DO_SAMPLE"]:
        gen_kwargs["temperature"] = cfg["TEMPERATURE"]
        gen_kwargs["top_p"] = cfg["TOP_P"]
        gen_kwargs["top_k"] = cfg["TOP_K"]

    with torch.no_grad():
        outputs = model.generate(**gen_kwargs)

    new_tokens = outputs[:, input_ids.shape[1]:]
    return tokenizer.batch_decode(new_tokens, skip_special_tokens=True, clean_up_tokenization_spaces=True)

def translate_all_instances(model, tokenizer, instances: List[Dict], cfg: Dict) -> List[Dict]:
    bs = cfg["BATCH_SIZE"]
    n = len(instances)
    outputs: List[Dict] = []

    prompts, span_maps = [], []
    for inst in instances:
        masked, spans = mask_spans(inst["content_en"])
        prompts.append(build_chat_prompt(masked, tokenizer))
        span_maps.append(spans)

    free_gpu()
    for start in tqdm(range(0, n, bs), desc="Translating", unit="batch", dynamic_ncols=True):
        end = min(start + bs, n)
        idxs = list(range(start, end))
        batch_prompts = [prompts[i] for i in idxs]

        hyps_raw = []
        last_err = None
        for attempt in range(cfg["MAX_RETRIES"]):
            try:
                hyps_raw = run_inference_on_batch(model, tokenizer, batch_prompts, cfg)
                last_err = None
                break
            except RuntimeError as e:
                last_err = str(e)
                if "out of memory" in last_err.lower():
                    log(f"OOM on batch {start}-{end}. Reduce BATCH_SIZE if persists.", "WARN")
                    free_gpu()
                    time.sleep(2)
                else:
                    log(f"Runtime error: {last_err}", "ERROR")
                    break

        for local_i, global_i in enumerate(idxs):
            inst = instances[global_i]
            spans = span_maps[global_i]
            if last_err and local_i >= len(hyps_raw):
                outputs.append({"id": inst["id"], "source": inst["content_en"], "hypothesis": "", "error": last_err})
            else:
                raw = (hyps_raw[local_i] or "").strip()
                hyp = unmask_spans(raw, spans)
                outputs.append({"id": inst["id"], "source": inst["content_en"], "hypothesis": hyp})

        if cfg["SLEEP_BETWEEN_BATCHES"] > 0:
            time.sleep(cfg["SLEEP_BETWEEN_BATCHES"])

    return outputs

# =============================================================================
# SECTION 11 — Evaluation
# =============================================================================
def evaluate_all(instances: List[Dict], outputs: List[Dict]) -> Dict:
    refs = {str(x["id"]): x["content_hi"] for x in instances}
    srcs = {str(x["id"]): x["content_en"] for x in instances}
    hyps = {str(x["id"]): x.get("hypothesis", "") for x in outputs}

    all_ids = [str(x["id"]) for x in instances]
    cats = {i: categorise_instance(refs[i]) for i in all_ids}

    def subset(ids: List[str]) -> Dict:
        r = [refs[i] for i in ids]
        h = [hyps[i] for i in ids]
        out = {"n": len(ids)}
        out.update(compute_bleu_chrf_default(r, h))  # ← sacrebleu defaults (your request)
        sfs = compute_sfs(r, h)
        out.update({
            "MSP": round(sfs["MSP"] * 100, 4),
            "MCP": round(sfs["MCP"] * 100, 4),
            "TSP": round(sfs["TSP"] * 100, 4),
            "TCP": round(sfs["TCP"] * 100, 4),
            "n_math": sfs["n_math"],
            "n_tables": sfs["n_tables"],
            "n_tcp": sfs["n_tcp"],
        })
        return out

    results = {
        "model": CONFIG["MODEL_ID"],
        "dtype": CONFIG["DTYPE"],
        "quantisation": "none",
        "overall": subset(all_ids),
    }
    for cat in ["pure_text", "math_only", "math_tables"]:
        ids = [i for i in all_ids if cats[i] == cat]
        results[cat] = subset(ids) if ids else {"n": 0}
    return results

# =============================================================================
# SECTION 12 — Save + Print
# =============================================================================
def save_jsonl(outputs: List[Dict], path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in outputs:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    log(f"Hypotheses saved → {path}", "OK")

def save_json(results: Dict, path: str):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    log(f"Results saved → {path}", "OK")

def print_table(results: Dict):
    section("FINAL RESULTS")
    print(f"Model: {results['model']}")
    print(f"dtype: {results['dtype']} | quant: none\n")
    header = f"{'Category':<12} {'n':>5} {'BLEU':>8} {'chrF':>8} {'MSP':>8} {'MCP':>8} {'TSP':>8} {'TCP':>8}"
    print(header)
    print("-" * len(header))
    for cat in ["overall", "pure_text", "math_only", "math_tables"]:
        r = results.get(cat, {})
        if r.get("n", 0) == 0:
            continue
        print(f"{cat:<12} {r['n']:>5} {r['BLEU']:>8.2f} {r['chrF']:>8.2f} "
              f"{r['MSP']:>8.2f} {r['MCP']:>8.2f} {r['TSP']:>8.2f} {r['TCP']:>8.2f}")
    print("-" * len(header))

# =============================================================================
# SECTION 13 — Main
# =============================================================================
def main():
    section("MathDoc-ENHI  |  Gemma-4 31B  |  Advisor Pipeline (NO QUANT, sacrebleu default)")
    log(f"Model    : {CONFIG['MODEL_ID']}")
    log(f"dtype    : {CONFIG['DTYPE']}")
    log(f"batch_sz : {CONFIG['BATCH_SIZE']}")
    log(f"Device(s): {get_device_info()}")

    torch.manual_seed(CONFIG["SEED"])

    # Paths sanity
    out_dir = Path(CONFIG["OUTPUT_DIR"])
    out_dir.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["HYPOTHESIS_JSONL"]).parent.mkdir(parents=True, exist_ok=True)
    Path(CONFIG["RESULTS_JSON"]).parent.mkdir(parents=True, exist_ok=True)

    section("LOADING DATASET")
    instances = load_dataset_records(CONFIG["DATA_PATHS"], expected_total=CONFIG["EXPECTED_TOTAL"])
    log(f"Dataset validated: {len(instances)} instances total", "OK")

    cat_counts = Counter(categorise_instance(inst["content_hi"]) for inst in instances)
    for cat, cnt in sorted(cat_counts.items()):
        log(f"  {cat:<12}: {cnt}")

    section("LOADING MODEL (TOKENIZER FIX APPLIED)")
    model, tokenizer = load_model_and_tokenizer(CONFIG)

    section("INFERENCE")
    outputs = translate_all_instances(model, tokenizer, instances, CONFIG)
    save_jsonl(outputs, CONFIG["HYPOTHESIS_JSONL"])

    # free model before eval (optional)
    del model
    free_gpu()

    section("EVALUATION")
    results = evaluate_all(instances, outputs)
    save_json(results, CONFIG["RESULTS_JSON"])
    print_table(results)

    section("DONE")
    log(f"Hypotheses: {CONFIG['HYPOTHESIS_JSONL']}", "OK")
    log(f"Results   : {CONFIG['RESULTS_JSON']}", "OK")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"Fatal error: {e}", "ERROR")
        traceback.print_exc()
        sys.exit(1)
