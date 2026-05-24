"""
Llama 3.3 70B Instruct inference on MathDoc-ENHI.

API:   OpenRouter / OpenAI-Compatible Provider
Model: meta-llama/llama-3.3-70b-instruct:free (or your provider's endpoint string)

This script preserves the exact string masking pattern of the original file. 
LaTeX syntax [EQ_k] and HTML tables [TB_k] are isolated, a targeted system prompt 
tells Llama 3.3 to leave them untouched, and the original markup is cleanly restored.

Usage:
    python run_llama.py \
        --input  ncert_class11_math_en_hi_test_instances_curated.json \
                 ncert_class12_math_en_hi_test_instances_curated.json \
        --output outputs/llama.jsonl
"""

import os
import re
import json
import time
import argparse
from tqdm import tqdm
from openai import OpenAI

from data_io import DataValidationError, load_dataset


# Masking patterns matching original processing schema
_MASK_PATTERNS = [
    (re.compile(r'\$\$[^$]+?\$\$', re.DOTALL), 'EQ'),
    (re.compile(r'\\\[.+?\\\]', re.DOTALL), 'EQ'),
    (re.compile(
        r'\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}.*?'
        r'\\end\{\1\*?\}', re.DOTALL), 'EQ'),
    (re.compile(r'\$[^$\n]+?\$'), 'EQ'),
    (re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE), 'TB'),
]


def mask_spans(text):
    """Isolate equations and table tokens so the LLM doesn't alter them."""
    spans = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def repl(m, prefix=prefix):
            spans.append(m.group(0))
            return f'[{prefix}_{len(spans) - 1}]'
        masked = pat.sub(repl, masked)
    return masked, spans


def unmask_spans(text, spans):
    """Reinsert original LaTeX code blocks back into the text string."""
    result = text
    for i, span in enumerate(spans):
        for prefix in ('EQ', 'TB'):
            result = result.replace(f'[{prefix}_{i}]', span)
    return result


def translate_one(client, text, model='meta-llama/llama-3.3-70b-instruct:free'):
    """Sends isolated payload to Llama 3.3 via OpenAI-compatible endpoint."""
    masked, spans = mask_spans(text)

    # Llama 3.3 requires strict system instructions to guarantee structure control
    system_prompt = (
        "You are an expert English-to-Hindi translator working with technical math materials. "
        "Translate the input English text into accurate, natural Hindi.\n\n"
        "CRITICAL INSTRUCTION: You will see placeholders like [EQ_0], [EQ_1], or [TB_0]. "
        "Do NOT modify, translate, or remove these placeholders. Keep them exactly as they are "
        "and position them correctly within the translated Hindi sentence. "
        "Respond ONLY with the text translation."
    )

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": masked}
        ],
        temperature=0.3,   # Lower temperature prevents formatting hallucinations
        max_new_tokens=3072, 
    )
    
    translated = response.choices[0].message.content.strip()
    return unmask_spans(translated, spans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--api-key', default=os.environ.get('LLAMA_API_KEY'))
    # Change base-url if you shift from OpenRouter to another local/cloud endpoint
    p.add_argument('--base-url', default='https://openrouter.ai/api/v1')
    p.add_argument('--model', default='meta-llama/llama-3.3-70b-instruct:free')
    p.add_argument('--sleep', type=float, default=0.2, help='Time buffer between calls')
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--expected-total', type=int)
    args = p.parse_args()

    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as e:
        raise SystemExit(f'Data validation failed: {e}')
    print(f'Validated {len(instances)} instances.')

    if not args.api_key:
        raise SystemExit(
            'Missing API Key. Set LLAMA_API_KEY env var or explicitly use --api-key.'
        )

    # Initialize client pointed at the selected routing platform
    client = OpenAI(
        base_url=args.base_url,
        api_key=args.api_key
    )

    print(f'Translating {len(instances)} text instances via Llama 3.3 70B...')

    outputs = []
    for inst in tqdm(instances):
        hyp = ''
        err = None
        for attempt in range(args.max_retries):
            try:
                hyp = translate_one(client, inst['content_en'], model=args.model)
                err = None
                break
            except Exception as e:
                err = str(e)
                time.sleep(2 * (attempt + 1))  # Backoff delay strategy

        record = {
            'id': inst['id'],
            'source': inst['content_en'],
            'hypothesis': hyp,
        }
        if err:
            record['error'] = err
        outputs.append(record)
        time.sleep(args.sleep)

    # Safely extract and generate output path
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, 'w', encoding='utf-8') as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('Processing complete.')


if __name__ == '__main__':
    main()
