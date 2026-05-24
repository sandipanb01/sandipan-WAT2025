"""
Sarvam Translate inference on MathDoc-ENHI.

API:   https://api.sarvam.ai/translate
Model: mayura:v1  (Sarvam's Indic-specialised MT model)

Sarvam does not natively preserve LaTeX, so we mask each equation
span as [EQ_k] before translation and restore the original LaTeX
verbatim after translation. The same is done for HTML tables.

Set the API key:
    export SARVAM_API_KEY=your_key_here

Usage:
    python run_sarvam.py \\
        --input  ncert_class11_math_en_hi_test_instances_curated.json \\
                 ncert_class12_math_en_hi_test_instances_curated.json \\
        --output outputs/sarvam.jsonl
"""

import os
import re
import json
import time
import argparse
from tqdm import tqdm
import requests

from data_io import DataValidationError, load_dataset


SARVAM_URL = 'https://api.sarvam.ai/translate'

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
        # Try both EQ and TB prefixes since we used both
        for prefix in ('EQ', 'TB'):
            result = result.replace(f'[{prefix}_{i}]', span)
    return result


def translate_one(text, api_key, mode='formal',
                  model='mayura:v1', timeout=60):
    """One API call to Sarvam Translate."""
    masked, spans = mask_spans(text)

    response = requests.post(
        SARVAM_URL,
        headers={'api-subscription-key': api_key},
        json={
            'input':                 masked,
            'source_language_code':  'en-IN',
            'target_language_code':  'hi-IN',
            'mode':                  mode,
            'model':                 model,
            'enable_preprocessing':  True,
        },
        timeout=timeout,
    )
    response.raise_for_status()
    translated = response.json().get(
        'translated_text', response.json().get('output', ''))
    return unmask_spans(translated, spans)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--api-key',
                   default=os.environ.get('SARVAM_API_KEY'))
    p.add_argument('--mode', default='formal',
                   choices=['formal', 'modern-colloquial',
                            'classic-colloquial', 'code-mixed'])
    p.add_argument('--sleep', type=float, default=0.3,
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
            'No API key. Set SARVAM_API_KEY env var or pass --api-key.')

    print(f'Translating {len(instances)} instances via Sarvam...')

    outputs = []
    for inst in tqdm(instances):
        hyp = ''
        err = None
        for attempt in range(args.max_retries):
            try:
                hyp = translate_one(
                    inst['content_en'], args.api_key,
                    mode=args.mode)
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

    with open(args.output, 'w', encoding='utf-8') as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('Done.')


if __name__ == '__main__':
    main()
