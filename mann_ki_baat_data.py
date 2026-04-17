#!/usr/bin/env python3
# -*- coding: utf-8 -*-

###############################################################
# Mann Ki Baat Structured Dataset Builder
# Hashimoto-style XML dataset generator
###############################################################

import requests
from bs4 import BeautifulSoup
import re
import os
import json
import unicodedata
import xml.etree.ElementTree as ET
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from urllib.parse import urljoin

###############################################################
# CONFIG
###############################################################

BASE = "https://www.pmindia.gov.in"
TAG_PAGE = "https://www.pmindia.gov.in/en/tag/mann-ki-baat/"

DATA_DIR = "mkb_dataset"

os.makedirs(DATA_DIR, exist_ok=True)

###############################################################
# STEP 1 — DISCOVER ALL EPISODES
###############################################################

def get_episode_links():

    links=set()

    print("Scanning episode pages...")

    for page in range(1,40):

        url=f"{TAG_PAGE}page/{page}/"

        try:
            r=requests.get(url,timeout=20)
        except:
            continue

        soup=BeautifulSoup(r.text,"lxml")

        for a in soup.find_all("a",href=True):

            href=a["href"]

            if "/news_updates/" in href and "mann" in href.lower():

                links.add(href.split("?")[0])

    links=list(links)

    print("TOTAL UNIQUE EPISODES FOUND:",len(links))

    return links


###############################################################
# STEP 2 — PRESERVE HTML INLINE TAGS
###############################################################

def serialize_html(element):

    """Convert HTML element to text while preserving tags"""

    text=""

    for node in element.children:

        if isinstance(node,str):

            text+=node

        else:

            tag=node.name

            inner=serialize_html(node)

            text+=f"<{tag}>{inner}</{tag}>"

    return text.strip()


###############################################################
# STEP 3 — EXTRACT TRANSCRIPT
###############################################################

def extract_segments(url):

    try:
        r=requests.get(url,timeout=30)
        r.raise_for_status()
    except:
        return []

    soup=BeautifulSoup(r.text,"lxml")

    blocks=[

        soup.find("div",class_="news_content detail_content"),
        soup.find("div",class_="td-post-content"),
        soup.find("div",class_="entry-content"),
        soup.find("div",class_="tdb-block-inner"),
        soup.find("div",itemprop="articleBody")

    ]

    block=None

    for b in blocks:

        if b:
            block=b
            break

    if block:
        tags=block.find_all(["p","li"])
    else:
        tags=soup.find_all("p")

    segments=[]

    for tag in tags:

        text=serialize_html(tag)

        if len(text)<40:
            continue

        if "share this" in text.lower():
            continue

        segments.append(text)

    return segments


###############################################################
# STEP 4 — SENTENCE SEGMENTATION
###############################################################

def split_sentences(paragraphs):

    sentences=[]

    for p in paragraphs:

        parts=re.split(r'(?<=[.!?])\s+',p)

        for s in parts:

            s=s.strip()

            if len(s)>15:
                sentences.append(s)

    return sentences


###############################################################
# STEP 5 — NORMALIZATION
###############################################################

def normalize(text):

    text=unicodedata.normalize("NFC",text)

    text=text.replace("\u200c","")
    text=text.replace("\u200d","")

    return text


###############################################################
# STEP 6 — BUILD DATASET
###############################################################

def build_dataset(links):

    dataset=[]

    print("Extracting transcripts...")

    for episode_id,url in enumerate(tqdm(links)):

        paragraphs=extract_segments(url)

        sentences=split_sentences(paragraphs)

        for sid,s in enumerate(sentences):

            s=normalize(s)

            dataset.append({

                "episode":episode_id,
                "url":url,
                "seg_id":sid,
                "source":s,
                "target":s

            })

    print("TOTAL SEGMENTS:",len(dataset))

    return dataset


###############################################################
# STEP 7 — FILTER BAD SEGMENTS
###############################################################

def filter_pairs(data):

    clean=[]

    for x in data:

        src=x["source"]
        tgt=x["target"]

        if len(src)<10:
            continue

        if len(tgt)<10:
            continue

        ratio=len(tgt)/len(src)

        if ratio>4:
            continue

        clean.append(x)

    print("After filtering:",len(clean))

    return clean


###############################################################
# STEP 8 — TRAIN DEV TEST SPLIT
###############################################################

def split_dataset(data):

    train,temp=train_test_split(data,test_size=0.2,random_state=42)

    dev,test=train_test_split(temp,test_size=0.5,random_state=42)

    print("Train:",len(train))
    print("Dev:",len(dev))
    print("Test:",len(test))

    return train,dev,test


###############################################################
# STEP 9 — SAVE JSONL
###############################################################

def save_jsonl(data,path):

    with open(path,"w",encoding="utf-8") as f:

        for x in data:

            f.write(json.dumps(x,ensure_ascii=False)+"\n")


###############################################################
# STEP 10 — HASHIMOTO STYLE XML
###############################################################

def save_xml(data,path,split_name):

    root=ET.Element("dataset")

    doc=ET.SubElement(root,"doc")

    doc.set("id",split_name)

    for x in data:

        seg=ET.SubElement(doc,"seg")

        seg.set("id",str(x["seg_id"]))

        src=ET.SubElement(seg,"source")
        src.text=x["source"]

        tgt=ET.SubElement(seg,"target")
        tgt.text=x["target"]

    tree=ET.ElementTree(root)

    tree.write(path,encoding="utf-8",xml_declaration=True)


###############################################################
# MAIN PIPELINE
###############################################################

def main():

    links=get_episode_links()

    dataset=build_dataset(links)

    dataset=filter_pairs(dataset)

    train,dev,test=split_dataset(dataset)

    ############################################
    # SAVE JSONL
    ############################################

    save_jsonl(train,f"{DATA_DIR}/train.jsonl")
    save_jsonl(dev,f"{DATA_DIR}/dev.jsonl")
    save_jsonl(test,f"{DATA_DIR}/test.jsonl")

    ############################################
    # SAVE XML
    ############################################

    save_xml(train,f"{DATA_DIR}/train.xml","train")
    save_xml(dev,f"{DATA_DIR}/dev.xml","dev")
    save_xml(test,f"{DATA_DIR}/test.xml","test")

    print("DATASET COMPLETE")


###############################################################

if __name__=="__main__":

    main()
    
########### FOR COLAB ONLY####################    

import zipfile
from google.colab import files

zip_path="mkb_dataset.zip"

with zipfile.ZipFile(zip_path,"w") as z:

    z.write("mkb_dataset/train.jsonl")
    z.write("mkb_dataset/dev.jsonl")
    z.write("mkb_dataset/test.jsonl")

    z.write("mkb_dataset/train.xml")
    z.write("mkb_dataset/dev.xml")
    z.write("mkb_dataset/test.xml")

files.download(zip_path)    
    
