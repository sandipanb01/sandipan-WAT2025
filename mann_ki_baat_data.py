# =========================================================
# RESEARCH-GRADE MULTILINGUAL DATASET PIPELINE
# Mann Ki Baat → Parallel Corpus
# W19-5212 compliant
# =========================================================

import requests
from bs4 import BeautifulSoup
import re
import time
import unicodedata
from tqdm import tqdm
import pandas as pd
from sklearn.model_selection import train_test_split

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# =========================================================
# CONFIG
# =========================================================

BASE_URL = "https://www.pmindia.gov.in/en/tag/mann-ki-baat/"
SITEMAP = "https://www.pmindia.gov.in/sitemap.xml"

HEADERS = {"User-Agent": "Mozilla/5.0"}

MODEL_NAME = "sarvamai/sarvam-translate"

BATCH_SIZE = 16
MIN_SENT_LEN = 15
MAX_SENT_LEN = 600

# =========================================================
# LOAD MODEL (Stable HF loading)
# =========================================================

device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    device_map="auto"
)

model.eval()

# =========================================================
# TEXT CLEANING
# =========================================================

def normalize(text):

    text = unicodedata.normalize("NFC", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# INDIC SAFE SENTENCE SPLITTER
# =========================================================

def split_sentences(text):

    sents = re.split(r'(?<=[.!?।])\s+', text)

    return [s.strip() for s in sents if len(s.strip()) > 0]


# =========================================================
# STEP 1 — CRAWL EPISODE URLS
# =========================================================

def episode_number(url):

    m = re.search(r'(\d+)', url)
    return int(m.group(1)) if m else 9999


def get_urls():

    urls = set()

    # tag pagination
    for i in range(1, 40):

        try:
            r = requests.get(f"{BASE_URL}page/{i}/", headers=HEADERS)
            soup = BeautifulSoup(r.text, "lxml")

            for a in soup.find_all("a", href=True):

                if "mann-ki-baat" in a["href"] and a["href"].startswith("http"):
                    urls.add(a["href"])

        except:
            pass

    # sitemap
    try:
        r = requests.get(SITEMAP)
        soup = BeautifulSoup(r.text, "xml")

        for loc in soup.find_all("loc"):

            if "mann-ki-baat" in loc.text:
                urls.add(loc.text)

    except:
        pass

    urls = sorted(list(urls), key=episode_number)

    return urls


urls = get_urls()

print("URLs collected:", len(urls))


# =========================================================
# STEP 2 — SCRAPE EPISODE CONTENT
# =========================================================

def scrape_episode(url):

    try:

        r = requests.get(url, headers=HEADERS, timeout=30)

        soup = BeautifulSoup(r.text, "lxml")

        article = soup.find("div", {"class": "entry-content"})

        if article is None:
            return None

        paragraphs = []

        for i, p in enumerate(article.find_all("p")):

            txt = normalize(p.get_text())

            if len(txt) > 40:
                paragraphs.append({
                    "para_id": i,
                    "text": txt
                })

        if len(paragraphs) < 3:
            return None

        return paragraphs

    except:

        return None


episodes = []

for url in tqdm(urls):

    data = scrape_episode(url)

    if data:
        episodes.append({
            "url": url,
            "paragraphs": data
        })

print("Episodes scraped:", len(episodes))


# =========================================================
# STEP 3 — SENTENCE SEGMENTATION
# =========================================================

rows = []

for eid, ep in enumerate(episodes):

    for para in ep["paragraphs"]:

        sentences = split_sentences(para["text"])

        for sid, s in enumerate(sentences):

            if len(s) < MIN_SENT_LEN or len(s) > MAX_SENT_LEN:
                continue

            rows.append({

                "episode_id": eid,
                "url": ep["url"],
                "para_id": para["para_id"],
                "sent_id": sid,
                "en": s

            })

print("English segments:", len(rows))


# =========================================================
# STEP 4 — BATCH TRANSLATION
# =========================================================

def translate_batch(sentences, language):

    prompt = (
        f"Translate each line into {language}. "
        "Return translations line-by-line in same order.\n\n"
    )

    prompt += "\n".join(sentences)

    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    with torch.no_grad():

        output = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.01,
            do_sample=False
        )

    decoded = tokenizer.decode(output[0], skip_special_tokens=True)

    lines = decoded.split("\n")

    return lines[-len(sentences):]


# =========================================================
# STEP 5 — MULTILINGUAL TRANSLATION
# =========================================================

for i in tqdm(range(0, len(rows), BATCH_SIZE)):

    batch = rows[i:i+BATCH_SIZE]

    en_sentences = [r["en"] for r in batch]

    hi = translate_batch(en_sentences, "Hindi")
    ta = translate_batch(en_sentences, "Tamil")
    bn = translate_batch(en_sentences, "Bengali")

    for j in range(len(batch)):

        batch[j]["hi"] = hi[j] if j < len(hi) else ""
        batch[j]["ta"] = ta[j] if j < len(ta) else ""
        batch[j]["bn"] = bn[j] if j < len(bn) else ""


# =========================================================
# STEP 6 — DATAFRAME
# =========================================================

df = pd.DataFrame(rows)

print("Final dataset size:", len(df))


# =========================================================
# STEP 7 — TRAIN DEV TEST SPLIT
# =========================================================

train, temp = train_test_split(df, test_size=0.2, random_state=42)

dev, test = train_test_split(temp, test_size=0.5, random_state=42)


# =========================================================
# STEP 8 — SAVE PARQUET (Pralekha style)
# =========================================================

train.to_parquet("train.parquet")
dev.to_parquet("dev.parquet")
test.to_parquet("test.parquet")

df.to_parquet("full_dataset.parquet")

print("\nPipeline complete.")
print("Train:", len(train))
print("Dev:", len(dev))
print("Test:", len(test))
