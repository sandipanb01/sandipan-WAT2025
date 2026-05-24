"""
Sarvam 105B (Chat LLM) inference for MathDoc-ENHI translation.

API:   Sarvam OpenAI-compatible Chat Completions
Base:  https://api.sarvam.ai/v1
Model: sarvam-105b

We mask LaTeX spans as [EQ_k] and HTML tables as [TB_k] before sending to the LLM,
then restore them verbatim after translation.

Set the API key:
    export SARVAM_API_KEY=your_key_here

Usage:
    python run_sarvam_105b.py \
        --input  ncert_class11_math_en_hi_test_instances_curated.json \
                 ncert_class12_math_en_hi_test_instances_curated.json \
        --output outputs/sarvam_105b.jsonl \
        --max-new-tokens 3072
"""

import os
import re
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI

from data_io import DataValidationError, load_dataset


# Sarvam OpenAI-compatible base URL (chat completions)
SARVAM_BASE_URL = "https://api.sarvam.ai/v1"

# Patterns to mask before sending to API.
# Order matters: $$...$$ must match before $...$.
_MASK_PATTERNS = [
    (re.compile(r"\$\$[^$]+?\$\$", re.DOTALL), "EQ"),
    (re.compile(r"\\\[.+?\\\]", re.DOTALL), "EQ"),
    (re.compile(
        r"\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}.*?"
        r"\\end\{\1\*?\}", re.DOTALL), "EQ"),
    (re.compile(r"\$[^$\n]+?\$"), "EQ"),
    (re.compile(r"<table[^>]*>.*?</table>", re.DOTALL | re.IGNORECASE), "TB"),
]


def mask_spans(text: str):
    """Mask equations and tables. Returns (masked_text, span_list)."""
    spans = []
    masked = text

    for pat, prefix in _MASK_PATTERNS:
        def repl(m, prefix=prefix):
            spans.append(m.group(0))
            return f"[{prefix}_{len(spans) - 1}]"
        masked = pat.sub(repl, masked)

    return masked, spans


def unmask_spans(text: str, spans):
    """Restore original spans by replacing placeholders."""
    result = text
    for i, span in enumerate(spans):
        for prefix in ("EQ", "TB"):
            result = result.replace(f"[{prefix}_{i}]", span)
    return result


def translate_one(
    client: OpenAI,
    text: str,
    model: str = "sarvam-105b",
    temperature: float = 0.2,
    top_p: float = 1.0,
    max_tokens: int = 3072,
):
    """One translation call using Sarvam-105B via Chat Completions."""
    masked, spans = mask_spans(text)

    system_prompt = (
        "You are an expert English-to-Hindi translator working with technical math materials. "
        "Translate the input English text into accurate, natural Hindi.\n\n"
        "CRITICAL INSTRUCTION: You will see placeholders like [EQ_0], [EQ_1], or [TB_0]. "
        "Do NOT modify, translate, or remove these placeholders. Keep them exactly as they are "
        "and position them correctly within the translated Hindi sentence. "
        "Respond ONLY with the text translation."
    )

    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": masked},
        ],
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens,   # <-- THIS is "max new tokens"
    )

    translated = (resp.choices[0].message.content or "").strip()
    return unmask_spans(translated, spans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--input", nargs="+", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--api-key", default=os.environ.get("SARVAM_API_KEY"))
    p.add_argument("--model", default="sarvam-105b")
    p.add_argument("--temperature", type=float, default=0.2)
    p.add_argument("--top-p", type=float, default=1.0)
    p.add_argument("--max-new-tokens", type=int, default=3072)
    p.add_argument("--sleep", type=float, default=0.3, help="Seconds between API calls")
    p.add_argument("--max-retries", type=int, default=3)
    p.add_argument("--expected-total", type=int,
                   help="Fail unless the loaded dataset has this many records")
    args = p.parse_args()

    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as e:
        raise SystemExit(f"Data validation failed: {e}")
    print(f"Validated {len(instances)} instances.")

    if not args.api_key:
        raise SystemExit("No API key. Set SARVAM_API_KEY env var or pass --api-key.")

    # Sarvam supports OpenAI-compatible auth; Sarvam docs also accept api-subscription-key header.
    # We send both to be safe across setups.
    client = OpenAI(
        base_url=SARVAM_BASE_URL,
        api_key=args.api_key,
        default_headers={"api-subscription-key": args.api_key},
    )

    print(f"Translating {len(instances)} instances via Sarvam 105B ({args.model})...")

    outputs = []
    for inst in tqdm(instances):
        hyp = ""
        err = None

        for attempt in range(args.max_retries):
            try:
                hyp = translate_one(
                    client=client,
                    text=inst["content_en"],
                    model=args.model,
                    temperature=args.temperature,
                    top_p=args.top_p,
                    max_tokens=args.max_new_tokens,
                )
                err = None
                break
            except Exception as e:
                err = str(e)
                time.sleep(2 * (attempt + 1))

        record = {
            "id": inst["id"],
            "source": inst["content_en"],
            "hypothesis": hyp,
        }
        if err:
            record["error"] = err

        outputs.append(record)
        time.sleep(args.sleep)

    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, "w", encoding="utf-8") as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")

    print("Done.")


if __name__ == "__main__":
    main()
