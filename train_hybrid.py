# -*- coding: utf-8 -*-
"""
Model hybrid machine learning untuk sentimen review milk tea:
fitur TF-IDF / Word2Vec / FastText x klasifier LogReg / SVM / RandomForest.
"""
import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from gensim.models import FastText, Word2Vec
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_score, recall_score, roc_auc_score)
from sklearn.svm import LinearSVC

BASE = Path(__file__).resolve().parent
DATA = BASE / "data"
RESULTS = BASE / "results"
RESULTS.mkdir(exist_ok=True)
OUT_JSON = RESULTS / "metrics_hybrid.json"
SEED = 42

train = pd.read_csv(DATA / "train.csv")
test = pd.read_csv(DATA / "test.csv")
ytr, yte = train["label_id"].values, test["label_id"].values
tr_tokens = [t.split() for t in train["text_clean"]]
te_tokens = [t.split() for t in test["text_clean"]]

results = {"y_true": yte.tolist(), "models": {}}


def eval_model(name, feat, clf, Xtr, Xte, t0):
    clf.fit(Xtr, ytr)
    if hasattr(clf, "predict_proba"):
        prob = clf.predict_proba(Xte)[:, 1]
    else:  # LinearSVC: pakai decision_function utk ROC
        s = clf.decision_function(Xte)
        prob = (s - s.min()) / (s.max() - s.min() + 1e-9)
    pred = clf.predict(Xte)
    results["models"][name] = {
        "type": "hybrid",
        "feature": feat,
        "accuracy": float(accuracy_score(yte, pred)),
        "precision": float(precision_score(yte, pred, average="macro")),
        "recall": float(recall_score(yte, pred, average="macro")),
        "f1": float(f1_score(yte, pred, average="macro")),
        "auc": float(roc_auc_score(yte, prob)),
        "cm": confusion_matrix(yte, pred).tolist(),
        "y_prob": np.round(prob, 6).tolist(),
        "train_time_sec": round(time.time() - t0, 1),
    }
    r = results["models"][name]
    print(f"{name:26s} acc={r['accuracy']:.4f} f1={r['f1']:.4f} auc={r['auc']:.4f}")


# ----------------------------------------------------------------- TF-IDF
t0 = time.time()
tfidf = TfidfVectorizer(ngram_range=(1, 2), min_df=2, max_features=50000,
                        sublinear_tf=True)
Xtr_tf = tfidf.fit_transform(train["text_clean"])
Xte_tf = tfidf.transform(test["text_clean"])

logreg_tf = LogisticRegression(C=4, max_iter=2000, random_state=SEED)
eval_model("TF-IDF + LogReg", "tfidf", logreg_tf, Xtr_tf, Xte_tf, t0)
joblib.dump({"vectorizer": tfidf, "clf": logreg_tf},
            RESULTS / "hybrid_predictor.joblib")

eval_model("TF-IDF + SVM", "tfidf",
           LinearSVC(C=1, random_state=SEED), Xtr_tf, Xte_tf, time.time())


# ------------------------------------------------------- Word2Vec / FastText
def doc_vectors(tokens_list, kv, dim):
    out = np.zeros((len(tokens_list), dim), dtype=np.float32)
    for i, toks in enumerate(tokens_list):
        vecs = [kv[w] for w in toks if w in kv]
        if vecs:
            out[i] = np.mean(vecs, axis=0)
    return out


DIM = 200
t0 = time.time()
w2v = Word2Vec(tr_tokens, vector_size=DIM, window=5, min_count=2, sg=1,
               epochs=30, seed=SEED, workers=4)
Xtr_w2v = doc_vectors(tr_tokens, w2v.wv, DIM)
Xte_w2v = doc_vectors(te_tokens, w2v.wv, DIM)
print(f"word2vec dilatih {time.time()-t0:.1f}s, vocab={len(w2v.wv)}")

eval_model("Word2Vec + LogReg", "word2vec",
           LogisticRegression(C=4, max_iter=2000, random_state=SEED),
           Xtr_w2v, Xte_w2v, time.time())
eval_model("Word2Vec + RandomForest", "word2vec",
           RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
           Xtr_w2v, Xte_w2v, time.time())

t0 = time.time()
ft = FastText(tr_tokens, vector_size=DIM, window=5, min_count=2, sg=1,
              epochs=30, min_n=3, max_n=5, seed=SEED, workers=4)
Xtr_ft = doc_vectors(tr_tokens, ft.wv, DIM)
Xte_ft = doc_vectors(te_tokens, ft.wv, DIM)
print(f"fasttext dilatih {time.time()-t0:.1f}s, vocab={len(ft.wv)}")

eval_model("FastText + LogReg", "fasttext",
           LogisticRegression(C=4, max_iter=2000, random_state=SEED),
           Xtr_ft, Xte_ft, time.time())
eval_model("FastText + RandomForest", "fasttext",
           RandomForestClassifier(n_estimators=300, random_state=SEED, n_jobs=-1),
           Xtr_ft, Xte_ft, time.time())

OUT_JSON.write_text(json.dumps(results), encoding="utf-8")
print(f"\ntersimpan: {OUT_JSON}")
