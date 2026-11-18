"""Phase C - Offline classifier + baselines (ICDFA CIP-B107 LAB01).
Usage: python3 notebooks/phase_c_classifier.py originals
TF-IDF (character n-grams) + Logistic Regression, fitted on TRAIN only; evaluated on the supplied TEST file.
"""
import sys, os, re, json, ipaddress, platform
import numpy as np, pandas as pd, sklearn, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (confusion_matrix, classification_report, accuracy_score, precision_score,
                             recall_score, f1_score, roc_auc_score, average_precision_score, ConfusionMatrixDisplay)

SEED = 42
SRC = sys.argv[1] if len(sys.argv) > 1 else "originals"
os.makedirs("outputs/charts", exist_ok=True)
print("python", platform.python_version(), "| pandas", pd.__version__, "| scikit-learn", sklearn.__version__,
      "| numpy", np.__version__, "| seed", SEED)

def load(name):
    rows = [json.loads(l) for l in open(os.path.join(SRC, name))]
    d = pd.DataFrame(rows)
    d["label"] = d["completion"].str.strip()
    d["context"] = d["prompt"].str.rsplit(";", n=1).str[0]
    d["candidate"] = d["prompt"].str.rsplit(";", n=1).str[1].str.replace(" ->", "", regex=False).str.strip()
    d["norm"] = d["prompt"].str.replace(r"\s+", "", regex=True)      # whitespace-insensitive key for near-duplicates
    return d
tr, te = load("dataset_train.jsonl"), load("dataset_test.jsonl")
print("train", len(tr), tr.label.value_counts().to_dict(), "| test", len(te), te.label.value_counts().to_dict())

# ---- C1: leakage check ----
exact = set(tr.prompt) & set(te.prompt)
near = set(tr.norm) & set(te.norm)
leak_mask = te.norm.isin(tr.norm)
print("\n=== C1 LEAKAGE CHECK ===")
print("exact prompts in both train and test:", len(exact), "| whitespace-normalised overlap:", len(near),
      "| test rows affected:", int(leak_mask.sum()), "of", len(te))
print("same candidate IP in both splits:", len(set(tr.candidate) & set(te.candidate)), "candidates")

# ---- C2: rule-based baselines ----
def strict_valid(s):
    try: ipaddress.IPv4Address(s); return True
    except ValueError: return False
def boundary_clean(ctx, cand):
    i = ctx.find(cand)
    if i == -1: return False
    b = ctx[i-1] if i > 0 else ""; a = ctx[i+len(cand)] if i+len(cand) < len(ctx) else ""
    return not (b.isdigit() or b == "." or a.isdigit() or a == ".")
te["base_strict"] = ["positive" if strict_valid(c) else "negative" for c in te.candidate]
te["base_context"] = ["positive" if strict_valid(c) and boundary_clean(x, c) else "negative"
                      for x, c in zip(te.context, te.candidate)]
maj = tr.label.value_counts().idxmax()
te["majority"] = maj

# ---- C3: TF-IDF + Logistic Regression (vectorizer fitted on train only) ----
vec = TfidfVectorizer(analyzer="char", ngram_range=(2, 5), sublinear_tf=True)
Xtr = vec.fit_transform(tr.prompt); Xte = vec.transform(te.prompt)
clf = LogisticRegression(max_iter=2000, random_state=SEED).fit(Xtr, tr.label)
te["ml_pred"] = clf.predict(Xte)
te["ml_score_positive"] = clf.predict_proba(Xte)[:, list(clf.classes_).index("positive")]

def metrics(y, p, score=None):
    tn, fp, fn, tp = confusion_matrix(y, p, labels=["negative", "positive"]).ravel()
    m = dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=accuracy_score(y, p),
             precision=precision_score(y, p, pos_label="positive", zero_division=0),
             recall=recall_score(y, p, pos_label="positive", zero_division=0),
             f1=f1_score(y, p, pos_label="positive", zero_division=0))
    if score is not None:
        m["auroc"] = roc_auc_score((y == "positive").astype(int), score)
        m["auprc"] = average_precision_score((y == "positive").astype(int), score)
    return m
y = te.label
res = {"ML (TF-IDF+LogReg)": metrics(y, te.ml_pred, te.ml_score_positive),
       "Baseline: strict IPv4 only": metrics(y, te.base_strict),
       "Baseline: strict + context": metrics(y, te.base_context),
       "Majority class": metrics(y, te.majority)}
pd.set_option("display.width", 220)
table = pd.DataFrame(res).T.round(4)
print("\n=== C2/C3 PRIMARY EVALUATION: supplied test file (n=%d) ===" % len(te))
print(table.to_string())
print("\nML confusion matrix (rows=actual, cols=predicted; order negative, positive):")
print(confusion_matrix(y, te.ml_pred, labels=["negative", "positive"]))
print("\nClassification report (ML):\n", classification_report(y, te.ml_pred, zero_division=0))

# secondary: leakage-free subset
cl = te[~leak_mask]
print("=== SECONDARY EVALUATION: test rows with NO train overlap (n=%d, %s) ===" % (len(cl), cl.label.value_counts().to_dict()))
sec = {"ML (TF-IDF+LogReg)": metrics(cl.label, cl.ml_pred), "Baseline: strict + context": metrics(cl.label, cl.base_context),
       "Majority class": metrics(cl.label, cl.majority)}
print(pd.DataFrame(sec).T.round(4).to_string())

# ---- save artefacts ----
out = te[["prompt", "candidate", "label", "ml_pred", "ml_score_positive", "base_strict", "base_context", "majority"]].copy()
out.insert(0, "test_row", range(1, len(te) + 1)); out = out.rename(columns={"label": "actual"})
out["outcome_ml"] = [("TP" if a == "positive" and p == "positive" else "TN" if a == "negative" and p == "negative"
                      else "FP" if a == "negative" else "FN") for a, p in zip(out.actual, out.ml_pred)]
out["in_train_overlap"] = leak_mask.values
out.to_csv("outputs/predictions.csv", index=False)
table.to_csv("outputs/metrics_summary.csv")
ConfusionMatrixDisplay(confusion_matrix(y, te.ml_pred, labels=["negative", "positive"]),
                       display_labels=["negative", "positive"]).plot(cmap="Blues")
plt.title("ML classifier - confusion matrix (supplied test set)"); plt.savefig("outputs/charts/confusion_matrix_ml.png", dpi=150, bbox_inches="tight")
print("\nsaved: outputs/predictions.csv, outputs/metrics_summary.csv, outputs/charts/confusion_matrix_ml.png")
print("errors (ML):"); print(out[out.outcome_ml.isin(["FP", "FN"])][["test_row", "candidate", "actual", "ml_pred", "outcome_ml"]].to_string(index=False))
