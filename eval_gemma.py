# ---------------------------------------------------------
# FIXED: Chat-style tokenized prompt for evaluation

def build_eval_prompt_tokenized(example, tokenizer, src_lang, tgt_lang):
    """Create tokenized chat prompt exactly like training."""
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"

    messages = [
        {"role": "user", "content": user_prompt},
        {"role": "assistant", "content": ""}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    return input_ids

# ---------------------------------------------------------
# FIXED: Generate with EOS-aware stopping + safe slicing
# ---------------------------------------------------------
def generate_batch(model, tokenizer, batch_input_ids):
    enc = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            enc,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            eos_token_id=tokenizer.eos_token_id,  # ✅ FIXED
            pad_token_id=tokenizer.pad_token_id   # ✅ FIXED
        )

    results = []
    for i, ids in enumerate(batch_input_ids):
        prompt_len = len(ids)
        gen_ids = out[i][prompt_len:]
        text = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()
        results.append(text)

    return results


# ---------------------------------------------------------
# FIXED: Single-split loader (ALWAYS load eng_hin)
# ---------------------------------------------------------
def load_pralekha_split(lang1, lang2):
    split = "eng_hin"
    print(f"Dataset load info: split='{split}'")
    return load_dataset(
        "ai4bharat/Pralekha",
        name="train",
        split=split,
        streaming=True
    )


# ---------------------------------------------------------
# FIXED: Correct evaluation for both eng→hin and hin→eng
# ---------------------------------------------------------
def evaluate_direction(model, tokenizer, src_lang, tgt_lang, max_samples=200, batch_size=8):
    ds = load_pralekha_split(src_lang, tgt_lang)
    ds_iter = iter(ds)

    preds, refs = [], []
    processed = 0

    pbar = tqdm(total=max_samples, desc=f"Evaluating {src_lang}→{tgt_lang}")

    while processed < max_samples:
        batch_refs = []
        batch_ids = []

        for _ in range(batch_size):
            try:
                ex = next(ds_iter)
            except StopIteration:
                break

            if src_lang == "eng" and tgt_lang == "hin":
                src_text = ex["src_txt"]
                ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]
                ref_text = ex["src_txt"]

            ids = build_eval_prompt_tokenized(
                {"src_txt": src_text},
                tokenizer,
                src_lang,
                tgt_lang
            )

            batch_ids.append(ids)
            batch_refs.append(ref_text.strip())

        if not batch_ids:
            break

        outs = generate_batch(model, tokenizer, batch_ids)

        preds.extend([o.strip() for o in outs])
        refs.extend(batch_refs)

        processed += len(batch_ids)
        pbar.update(len(batch_ids))

    pbar.close()
    print(f"Done: {processed} samples for {src_lang}→{tgt_lang}")

    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf = sacrebleu.metrics.CHRF(word_order=2).corpus_score(preds, [refs]).score

    print(f"BLEU = {bleu:.2f}   chrF = {chrf:.3f}\n")
    return bleu, chrf


# ---------------------------------------------------------
# MAIN LOOP FOR BOTH DIRECTIONS
# ---------------------------------------------------------
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer, trainer = train_model(max_samples=max_samples)

    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(
            model,
            tokenizer,
            src,
            tgt,
            batch_size=EVAL_BATCH_SIZE
        )
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")

