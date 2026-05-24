"""
Gemma 4 31B inference on MathDoc-ENHI.

API:   Google AI Studio API
Model: google/gemma-4-31b-it (or gemini equivalent hosting gemma)

Like the original script, we mask each equation span as [EQ_k] and
HTML tables as [TB_k] before translation and restore them verbatim after.

Set the API key:
    export GEMMA_API_KEY=your_key_here

Usage:
    python run_gemma.py \
        --input  ncert_class11_math_en_hi_test_instances_curated.json \
                 ncert_class12_math_en_hi_test_instances_curated.json \
        --output outputs/gemma.jsonl
"""

import os
import re
import json
import time
import argparse
from tqdm import tqdm
from google import genai
from google.genai import types

from data_io import DataValidationError, load_dataset


# Patterns to mask before sending to API.
# Order matters: $$...$$ must match before $...$.
_MASK_PATTERNS = [
    (re.compile(r'\$\$[^$]+?\$\$', re.DOTALL), 'EQ'),
    (re.compile(r'\\\[.+?\\\]', re.DOTALL), 'EQ'),
    (re.compile(
        r'\\begin\{(equation|align|bmatrix|pmatrix|vmatrix|matrix)\*?\}.*?'
        r'\\end\{\1\*?\}', re.DOTALL), 'EQ'),
    (re.compile(r'\$[^$\n]+?\$'), 'EQ'),
    (re.compile(r'<table[^>]*>.*?</table>', re.DOTALL | re.IGNORECASE),
     'TB'),
]


def mask_spans(text):
    """Mask equations and tables. Returns (masked_text, span_list)."""
    spans = []
    masked = text
    for pat, prefix in _MASK_PATTERNS:
        def repl(m, prefix=prefix):
            spans.append(m.group(0))
            return f'[{prefix}_{len(spans) - 1}]'
        masked = pat.sub(repl, masked)
    return masked, spans


def unmask_spans(text, spans):
    """Restore original spans by replacing placeholders."""
    result = text
    # We replace in reverse to avoid double-replacement edge cases
    for i, span in enumerate(spans):
        for prefix in ('EQ', 'TB'):
            result = result.replace(f'[{prefix}_{i}]', span)
    return result


def translate_one(client, text, model='gemma-4-31b-it'):
    """One API call to Gemma 4 via Google GenAI Client."""
    masked, spans = mask_spans(text)

    # We explicitly instruct the LLM to preserve the masks
    system_instruction = (
        "You are a professional mathematical translator. Translate the following English text into Hindi. "
        "CRITICAL RULE: You will encounter placeholders like [EQ_0], [EQ_1], [TB_0], etc. "
        "Do NOT translate, alter, or omit these placeholders. Keep them exactly as they are in their "
        "correct relative positions in the translated Hindi sentence. Output ONLY the translated text."
    )

    response = client.models.generate_content(
        model=model,
        contents=masked,
        config=types.GenerateContentConfig(
            system_instruction=system_instruction,
            temperature=1.0,  # Standard recommended sampling config for Gemma 4
            top_p=0.95,
            top_k=64,
            max_output_tokens=3072
        ),
    )
    
    translated = response.text.strip()
    return unmask_spans(translated, spans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--api-key',
                   default=os.environ.get('GEMMA_API_KEY'))
    p.add_argument('--model', default='gemma-4-31b-it',
                   help='The specific variant or endpoint of Gemma 4 31B')
    p.add_argument('--sleep', type=float, default=0.5,
                   help='Seconds between API calls')
    p.add_argument('--max-retries', type=int, default=3)
    p.add_argument('--expected-total', type=int,
                   help='Fail unless the loaded dataset has this many records')
    args = p.parse_args()

    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as e:
        raise SystemExit(f'Data validation failed: {e}')
    print(f'Validated {len(instances)} instances.')

    if not args.api_key:
        raise SystemExit(
            'No API key found. Set GEMMA_API_KEY env var or pass --api-key.')

    # Initialize the official Google GenAI client
    client = genai.Client(api_key=args.api_key)

    print(f'Translating {len(instances)} instances via Gemma 4 ({args.model})...')

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
                time.sleep(2 * (attempt + 1))

        record = {
            'id': inst['id'],
            'source': inst['content_en'],
            'hypothesis': hyp,
        }
        if err:
            record['error'] = err
        outputs.append(record)
        time.sleep(args.sleep)

    # Ensure output directory exists
    out_dir = os.path.dirname(args.output)
    if out_dir and not os.path.exists(out_dir):
        os.makedirs(out_dir)

    with open(args.output, 'w', encoding='utf-8') as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('Done.')


if __name__ == '__main__':
    main()
