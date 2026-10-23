# -*- coding: utf-8 -*-
"""Split train/val/test + augmentasi sinonim hanya pada train (anti-leakage)."""
import random
from collections import Counter
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from clean_dataset import synonym_variant

random.seed(42)
BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
DATA.mkdir(exist_ok=True)

df = pd.read_csv(BASE / "dataset_clean.csv")
df = df[["text_clean", "label", "label_id"]].dropna()

train, temp = train_test_split(df, test_size=0.2, stratify=df["label_id"], random_state=42)
val, test = train_test_split(temp, test_size=0.5, stratify=temp["label_id"], random_state=42)

print("train:", dict(Counter(train["label"])))
print("val  :", dict(Counter(val["label"])))
print("test :", dict(Counter(test["label"])))

# --- augmentasi minoritas di train saja
cnt = Counter(train["label"])
minor = min(cnt, key=cnt.get)
major = max(cnt, key=cnt.get)
need = cnt[major] - cnt[minor]

all_texts = set(df["text_clean"])  # jangan buat varian yang identik dgn data manapun
minority_texts = train.loc[train["label"] == minor, "text_clean"].tolist()
augmented, attempts = [], 0
while len(augmented) < need and attempts < need * 80:
    var = synonym_variant(random.choice(minority_texts))
    attempts += 1
    if var not in all_texts:
        all_texts.add(var)
        augmented.append(var)

aug_df = pd.DataFrame({
    "text_clean": augmented,
    "label": minor,
    "label_id": 1 if minor == "positif" else 0,
})
train_bal = pd.concat([train, aug_df], ignore_index=True)
train_bal = train_bal.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"\naugmentasi train '{minor}': butuh {need}, dihasilkan {len(augmented)}")
print("train seimbang:", dict(Counter(train_bal["label"])))

train_bal.to_csv(DATA / "train.csv", index=False, encoding="utf-8")
val.to_csv(DATA / "val.csv", index=False, encoding="utf-8")
test.to_csv(DATA / "test.csv", index=False, encoding="utf-8")
print(f"\ntersimpan di {DATA}")
