# ---------------------------------------------------------
# FIXED: Chat-style tokenized prompt for evaluation
# ---------------------------------------------------------
def build_eval_prompt_tokenized(example, tokenizer, src_lang, tgt_lang):
    """Create tokenized chat prompt exactly like training."""
    user_prompt = f"Translate this {src_lang} text to {tgt_lang}:\n{example['src_txt']}"

    messages = [
        {"role": "user", "content": user_prompt}
    ]

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True
    )
    return input_ids


# ---------------------------------------------------------
# FIXED: Generate with variable-length slicing
# ---------------------------------------------------------
def generate_batch(model, tokenizer, batch_input_ids, max_new_tokens=256):
    enc = torch.nn.utils.rnn.pad_sequence(
        [torch.tensor(x) for x in batch_input_ids],
        batch_first=True,
        padding_value=tokenizer.pad_token_id,
    ).to(model.device)

    with torch.no_grad():
        out = model.generate(
            enc,
            max_new_tokens=max_new_tokens,
            do_sample=False
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
    """
    Pralekha ONLY has eng_XXX splits.
    - eng→hin  -> load split="eng_hin"
    - hin→eng  -> still load split="eng_hin" but reverse fields
    """
    # Always use eng_hin split for English-Hindi
    split = "eng_hin"
    print(f"Dataset load info: split='{split}'")
    return load_dataset("ai4bharat/Pralekha", name="train", split=split, streaming=True)


# ---------------------------------------------------------
# FIXED: Correct evaluation for both eng→hin and hin→eng
# ---------------------------------------------------------
def evaluate_direction(model, tokenizer, src_lang, tgt_lang, max_samples=150, batch_size=8):
    ds = load_pralekha_split(src_lang, tgt_lang)
    ds_iter = iter(ds)

    preds, refs, srcs = [], [], []
    processed = 0

    pbar = tqdm(total=max_samples, desc=f"Evaluating {src_lang}→{tgt_lang}")

    while processed < max_samples:
        batch_src = []
        batch_refs = []
        batch_ids = []

        for _ in range(batch_size):
            try:
                ex = next(ds_iter)
            except StopIteration:
                break

            # ---------------------------------------------------
            # FIXED: reverse fields for hin→eng
            # ---------------------------------------------------
            if src_lang == "eng" and tgt_lang == "hin":
                src_text = ex["src_txt"]
                ref_text = ex["tgt_txt"]
            else:
                src_text = ex["tgt_txt"]
                ref_text = ex["src_txt"]

            fake_ex = {"src_txt": src_text}
            ids = build_eval_prompt_tokenized(fake_ex, tokenizer, src_lang, tgt_lang)

            batch_src.append(src_text)
            batch_refs.append(ref_text)
            batch_ids.append(ids)

        if not batch_ids:
            break

        outs = generate_batch(model, tokenizer, batch_ids)

        preds.extend(outs)
        refs.extend([r.strip() for r in batch_refs])
        srcs.extend([s.strip() for s in batch_src])

        processed += len(batch_ids)
        pbar.update(len(batch_ids))

    pbar.close()
    print(f"Done: {processed} samples for {src_lang}→{tgt_lang}")

    # Compute BLEU and chrF
    bleu = sacrebleu.corpus_bleu(preds, [refs]).score
    chrf_metric = sacrebleu.metrics.CHRF(word_order=0)
    chrf = chrf_metric.corpus_score(preds, [refs]).score

    print(f"BLEU = {bleu:.2f}   chrF = {chrf:.3f}\n")
    return bleu, chrf


# ---------------------------------------------------------
# MAIN LOOP FOR BOTH DIRECTIONS
# ---------------------------------------------------------
if __name__ == "__main__":
    os.environ["CUDA_LAUNCH_BLOCKING"] = "1"

    # 1️⃣ Train
    max_samples = None if FULL_DATASET else MAX_COLAB_SAMPLES
    model, tokenizer, trainer = train_model(max_samples=max_samples)

    # 2️⃣ Evaluate ENG↔HIN
    results = {}
    for split in DIRECTIONS:
        src, tgt = split.split("_")
        bleu, chrf = evaluate_direction(model, tokenizer, src, tgt)
        results[split] = {"BLEU": bleu, "chrF": chrf}

    print("\n✅ Final Results (ENG↔HIN):")
    for split, scores in results.items():
        print(f"{split}: BLEU={scores['BLEU']:.2f}, chrF={scores['chrF']:.3f}")
