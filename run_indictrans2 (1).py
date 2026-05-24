"""
IndicTrans2 inference on MathDoc-ENHI.

Reference: Gala et al., 2023 (TMLR).
Model:     ai4bharat/indictrans2-en-indic-1B  (or -200M)

Requires:
    pip install transformers torch IndicTransToolkit

The IndicTransToolkit wraps the language-tag preprocessing that
IndicTrans2 expects (eng_Latn -> hin_Deva).

Usage:
    python run_indictrans2.py \\
        --input  ncert_class11_math_en_hi_test_instances_curated.json \\
                 ncert_class12_math_en_hi_test_instances_curated.json \\
        --output outputs/indictrans2.jsonl \\
        --model  ai4bharat/indictrans2-en-indic-1B
"""

import json
import argparse
from tqdm import tqdm

from data_io import DataValidationError, load_dataset


def chunk_long_text(text: str, max_chars: int = 3500) -> list:
    """
    IndicTrans2 has a max input length. For long instances, split by
    paragraph boundaries to keep each chunk under max_chars.
    """
    if len(text) <= max_chars:
        return [text]
    paragraphs = text.split('\n\n')
    chunks, current = [], ''
    for p in paragraphs:
        if len(current) + len(p) + 2 <= max_chars:
            current = (current + '\n\n' + p) if current else p
        else:
            if current:
                chunks.append(current)
            current = p
    if current:
        chunks.append(current)
    return chunks


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', nargs='+', required=True,
                   help='MathDoc-ENHI JSON/JSONL input file(s)')
    p.add_argument('--output', required=True,
                   help='JSONL output with hypotheses')
    p.add_argument('--model',
                   default='ai4bharat/indictrans2-en-indic-1B')
    p.add_argument('--device', default='cuda')
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--max-length', type=int, default=1024)
    p.add_argument('--num-beams', type=int, default=5)
    p.add_argument('--expected-total', type=int,
                   help='Fail unless the loaded dataset has this many records')
    args = p.parse_args()

    try:
        instances = load_dataset(args.input, expected_total=args.expected_total)
    except DataValidationError as e:
        raise SystemExit(f'Data validation failed: {e}')
    print(f'Validated {len(instances)} instances.')

    try:
        import torch
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
        from IndicTransToolkit.processor import IndicProcessor
    except ImportError as e:
        raise SystemExit(
            'Install dependencies with: pip install -r requirements.txt') from e

    print(f'Loading {args.model}...')
    dtype = (torch.float16 if args.device == 'cuda'
             else torch.float32)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, trust_remote_code=True)
    model = AutoModelForSeq2SeqLM.from_pretrained(
        args.model, trust_remote_code=True, torch_dtype=dtype
    ).to(args.device).eval()

    ip = IndicProcessor(inference=True)

    print(f'Translating {len(instances)} instances...')

    outputs = []
    for inst in tqdm(instances):
        src = inst['content_en']
        chunks = chunk_long_text(src, max_chars=3500)

        translated_chunks = []
        for i in range(0, len(chunks), args.batch_size):
            batch = chunks[i:i + args.batch_size]
            batch_pp = ip.preprocess_batch(
                batch, src_lang='eng_Latn', tgt_lang='hin_Deva')
            enc = tokenizer(
                batch_pp, return_tensors='pt', padding=True,
                truncation=True, max_length=args.max_length,
            ).to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc,
                    max_length=args.max_length,
                    num_beams=args.num_beams,
                    num_return_sequences=1,
                )
            decoded = tokenizer.batch_decode(
                gen, skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
            translated_chunks.extend(
                ip.postprocess_batch(decoded, lang='hin_Deva'))

        hypothesis = '\n\n'.join(translated_chunks)
        outputs.append({
            'id': inst['id'],
            'source': src,
            'hypothesis': hypothesis,
        })

    print(f'Writing to {args.output}...')
    with open(args.output, 'w', encoding='utf-8') as f:
        for o in outputs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')
    print('Done.')


if __name__ == '__main__':
    main()
