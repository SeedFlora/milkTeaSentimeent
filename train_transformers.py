# -*- coding: utf-8 -*-
"""
Fine-tuning 10 model transformer (2020-2021+) untuk sentimen review milk tea.
GPU: bf16 autocast, batch 16 (fallback 8 jika OOM), 3 epoch, pilih epoch
terbaik berdasar F1-macro validasi. Hasil disimpan inkremental ke JSON.
"""
import gc
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "metrics_transformers.json"
BEST_DIR = BASE / "best_model"

DEVICE = "cuda"
MAX_LEN = 128
EPOCHS = 3
LR = 2e-5
SEED = 42

MODELS = [
    ("IndoBERT-p1",      "indobenchmark/indobert-base-p1"),
    ("IndoBERT-p2",      "indobenchmark/indobert-base-p2"),
    ("IndoBERT-IndoLEM", "indolem/indobert-base-uncased"),
    ("IndoBERTweet",     "indolem/indobertweet-base-uncased"),
    ("IndoRoBERTa",      "flax-community/indonesian-roberta-base"),
    ("DistilBERT-Indo",  "cahya/distilbert-base-indonesian"),
    ("mBERT",            "bert-base-multilingual-cased"),
    ("XLM-RoBERTa",      "xlm-roberta-base"),
    ("mDistilBERT",      "distilbert-base-multilingual-cased"),
    ("mDeBERTa-v3",      "microsoft/mdeberta-v3-base"),
]
# cadangan jika ada model yang gagal diunduh/dimuat
BACKUPS = [
    ("IndoBERT-lite",    "indobenchmark/indobert-lite-base-p1"),
    ("BERT-Indo-cahya",  "cahya/bert-base-indonesian-1.5G"),
    ("RoBERTa-Indo-cahya", "cahya/roberta-base-indonesian-522M"),
]

torch.manual_seed(SEED)
np.random.seed(SEED)


class TextDataset(Dataset):
    def __init__(self, texts, labels, tokenizer):
        self.enc = tokenizer(list(texts), truncation=True, max_length=MAX_LEN,
                             padding=False)
        self.labels = list(labels)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, i):
        item = {k: self.enc[k][i] for k in self.enc}
        item["label"] = self.labels[i]
        return item


def collate(batch, pad_id):
    maxlen = max(len(x["input_ids"]) for x in batch)
    out = {}
    ids, mask, labels = [], [], []
    for x in batch:
        n = maxlen - len(x["input_ids"])
        ids.append(x["input_ids"] + [pad_id] * n)
        mask.append(x["attention_mask"] + [0] * n)
        labels.append(x["label"])
    out["input_ids"] = torch.tensor(ids)
    out["attention_mask"] = torch.tensor(mask)
    out["labels"] = torch.tensor(labels)
    return out


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, trues, losses = [], [], []
    for batch in loader:
        batch = {k: v.to(DEVICE) for k, v in batch.items()}
        with torch.autocast("cuda", dtype=torch.bfloat16):
            out = model(**batch)
        losses.append(out.loss.float().item())
        p = torch.softmax(out.logits.float(), dim=-1)[:, 1]
        probs.extend(p.cpu().tolist())
        trues.extend(batch["labels"].cpu().tolist())
    return np.array(trues), np.array(probs), float(np.mean(losses))


def train_one(name, hf_name, train_df, val_df, test_df, batch_size=16):
    tok = AutoTokenizer.from_pretrained(hf_name)
    model = AutoModelForSequenceClassification.from_pretrained(
        hf_name, num_labels=2).to(DEVICE)
    pad_id = tok.pad_token_id or 0

    tr_ds = TextDataset(train_df["text_clean"], train_df["label_id"], tok)
    va_ds = TextDataset(val_df["text_clean"], val_df["label_id"], tok)
    te_ds = TextDataset(test_df["text_clean"], test_df["label_id"], tok)
    coll = lambda b: collate(b, pad_id)
    tr_dl = DataLoader(tr_ds, batch_size=batch_size, shuffle=True, collate_fn=coll)
    va_dl = DataLoader(va_ds, batch_size=64, collate_fn=coll)
    te_dl = DataLoader(te_ds, batch_size=64, collate_fn=coll)

    total_steps = len(tr_dl) * EPOCHS
    warmup = int(total_steps * 0.1)
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: s / max(1, warmup) if s < warmup
        else max(0.0, (total_steps - s) / max(1, total_steps - warmup)))

    loss_hist, val_f1s, val_losses = [], [], []
    best_f1, best_state = -1.0, None
    t0 = time.time()
    step = 0
    for epoch in range(EPOCHS):
        model.train()
        running = []
        for batch in tr_dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            with torch.autocast("cuda", dtype=torch.bfloat16):
                out = model(**batch)
            loss = out.loss.float()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad()
            step += 1
            running.append(loss.item())
            if step % 20 == 0:
                loss_hist.append([step, float(np.mean(running[-20:]))])
        yv, pv, vloss = evaluate(model, va_dl)
        vf1 = f1_score(yv, (pv >= 0.5).astype(int), average="macro")
        val_f1s.append(float(vf1))
        val_losses.append(vloss)
        print(f"    epoch {epoch+1}/{EPOCHS}  train_loss={np.mean(running):.4f}  "
              f"val_loss={vloss:.4f}  val_f1={vf1:.4f}", flush=True)
        if vf1 > best_f1:
            best_f1 = vf1
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

    model.load_state_dict(best_state)
    yt, pt, _ = evaluate(model, te_dl)
    pred = (pt >= 0.5).astype(int)
    res = {
        "type": "transformer",
        "hf_name": hf_name,
        "accuracy": float(accuracy_score(yt, pred)),
        "precision": float(precision_score(yt, pred, average="macro")),
        "recall": float(recall_score(yt, pred, average="macro")),
        "f1": float(f1_score(yt, pred, average="macro")),
        "auc": float(roc_auc_score(yt, pt)),
        "cm": confusion_matrix(yt, pred).tolist(),
        "y_prob": pt.round(6).tolist(),
        "loss_history": loss_hist,
        "val_f1_per_epoch": val_f1s,
        "val_loss_per_epoch": val_losses,
        "train_time_sec": round(time.time() - t0, 1),
        "params_m": round(sum(p.numel() for p in model.parameters()) / 1e6, 1),
        "batch_size": batch_size,
    }
    return res, model, tok


def main():
    train_df = pd.read_csv(DATA / "train.csv")
    val_df = pd.read_csv(DATA / "val.csv")
    test_df = pd.read_csv(DATA / "test.csv")
    print(f"train={len(train_df)} val={len(val_df)} test={len(test_df)}", flush=True)

    results = {"y_true": test_df["label_id"].tolist(), "models": {}}
    if OUT_JSON.exists():
        results = json.loads(OUT_JSON.read_text(encoding="utf-8"))

    best_overall = (-1.0, None)  # (f1, name)
    queue = list(MODELS)
    backups = list(BACKUPS)
    trained = 0
    while queue and trained < 10:
        name, hf_name = queue.pop(0)
        if name in results["models"]:
            print(f"[skip] {name} sudah ada", flush=True)
            trained += 1
            f1_prev = results["models"][name]["f1"]
            if f1_prev > best_overall[0]:
                best_overall = (f1_prev, name)
            continue
        print(f"\n=== [{trained+1}/10] {name} ({hf_name}) ===", flush=True)
        for bs in (16, 8):
            try:
                res, model, tok = train_one(name, hf_name, train_df, val_df,
                                            test_df, batch_size=bs)
                results["models"][name] = res
                OUT_JSON.write_text(json.dumps(results), encoding="utf-8")
                print(f"  selesai: acc={res['accuracy']:.4f} f1={res['f1']:.4f} "
                      f"auc={res['auc']:.4f} ({res['train_time_sec']}s)", flush=True)
                if res["f1"] > best_overall[0]:
                    best_overall = (res["f1"], name)
                    BEST_DIR.mkdir(exist_ok=True)
                    model.save_pretrained(BEST_DIR)
                    tok.save_pretrained(BEST_DIR)
                    (BEST_DIR / "info.json").write_text(json.dumps(
                        {"name": name, "hf_name": hf_name, "f1": res["f1"]}))
                    print(f"  -> model terbaik sejauh ini, disimpan ke {BEST_DIR}", flush=True)
                del model, tok
                gc.collect()
                torch.cuda.empty_cache()
                trained += 1
                break
            except torch.cuda.OutOfMemoryError:
                print(f"  OOM dengan batch {bs}, coba lebih kecil...", flush=True)
                gc.collect()
                torch.cuda.empty_cache()
            except Exception:
                print(f"  GAGAL memuat/melatih {name}:", flush=True)
                traceback.print_exc()
                if backups:
                    sub = backups.pop(0)
                    print(f"  -> substitusi dengan cadangan {sub[0]}", flush=True)
                    queue.insert(0, sub)
                break

    print(f"\n=== SELESAI. {trained} model. Terbaik: {best_overall[1]} "
          f"(F1={best_overall[0]:.4f}) ===", flush=True)


if __name__ == "__main__":
    main()
