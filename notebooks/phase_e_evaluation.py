"""Phase E - Evaluation and forensic interpretation (ICDFA CIP-B107 LAB01).
Usage: python3 notebooks/phase_e_evaluation.py predictions.csv
Every metric here is recomputed from the saved actual/predicted records in predictions.csv.
"""
import sys, os
import pandas as pd, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, accuracy_score, precision_score, recall_score, f1_score,
                             roc_auc_score, average_precision_score)

PRED = sys.argv[1] if len(sys.argv) > 1 else "predictions.csv"
os.makedirs("outputs/charts", exist_ok=True)
pd.set_option("display.width", 230); pd.set_option("display.max_colwidth", 60); pd.set_option("display.max_rows", 100)
p = pd.read_csv(PRED)
p["context"] = p["prompt"].str.rsplit(";", n=1).str[0].str.strip()
POS = "positive"

def outcome(a, q):
    return "TP" if a == POS and q == POS else "TN" if a != POS and q != POS else "FP" if a != POS else "FN"
for col, name in [("ml_pred", "ml"), ("base_strict", "strict"), ("base_context", "context_rule"), ("majority", "majority")]:
    p["outcome_" + name] = [outcome(a, q) for a, q in zip(p.actual, p[col])]

# ---- E1: actual vs predicted table (all holdout records) ----
print("=== E1 ACTUAL vs PREDICTED, ALL %d HOLDOUT RECORDS (positive = positive class) ===" % len(p))
tbl = p[["test_row", "candidate", "actual", "ml_pred", "outcome_ml", "base_strict", "outcome_strict", "majority", "outcome_majority"]]
print(tbl.to_string(index=False))
tbl.to_csv("outputs/actual_vs_predicted_table.csv", index=False)

# ---- E2: metrics recomputed from saved records ----
def m(y, q, score=None):
    tn, fp, fn, tp = confusion_matrix(y, q, labels=["negative", POS]).ravel()
    r = dict(TP=tp, FP=fp, FN=fn, TN=tn, accuracy=accuracy_score(y, q),
             precision=precision_score(y, q, pos_label=POS, zero_division=0),
             recall=recall_score(y, q, pos_label=POS, zero_division=0),
             f1=f1_score(y, q, pos_label=POS, zero_division=0),
             specificity=tn / (tn + fp) if tn + fp else float("nan"))
    if score is not None:
        r["auroc"] = roc_auc_score((y == POS).astype(int), score); r["auprc"] = average_precision_score((y == POS).astype(int), score)
    return r
res = pd.DataFrame({"ML (TF-IDF+LogReg)": m(p.actual, p.ml_pred, p.ml_score_positive),
                    "Strict IPv4 rule": m(p.actual, p.base_strict),
                    "Strict + context rule": m(p.actual, p.base_context),
                    "Majority class": m(p.actual, p.majority)}).T
print("\n=== E2 METRICS RECOMPUTED FROM predictions.csv ===")
print(res.round(4).to_string())
print("\nML confusion matrix [rows actual neg,pos | cols predicted neg,pos]:\n", confusion_matrix(p.actual, p.ml_pred, labels=["negative", POS]))
print("\nmajority-class label:", p.majority.iloc[0], "| class balance in holdout:", p.actual.value_counts().to_dict())
res.round(4).to_csv("outputs/evaluation_metrics.csv")
ax = res[["accuracy", "precision", "recall", "f1", "specificity"]].plot(kind="bar", figsize=(9, 4.5), rot=15)
ax.set_ylim(0, 1.05); ax.set_title("Holdout metrics by method (n=%d)" % len(p)); ax.grid(axis="y", alpha=.3)
plt.savefig("outputs/charts/model_comparison.png", dpi=150, bbox_inches="tight"); plt.close()

# ---- E3: analyst review (>=8 records) ----
# DRAFT decisions and rationales: read each against the printed context and REWRITE in your own words.
review = {
 1:  ("reject as IP evidence", "Source negative. 608.80.24.1 is clipped out of the longer string 3608.80.24.1.8; octet 608 is invalid. ML got it right but with the lowest confidence of any record."),
 20: ("reject as IP evidence", "Source negative. 257.0.1.38 has an octet above 255 and sits inside a five-part build-style number. ML false positive: character n-grams cannot check octet range; the strict validator rejects it."),
 22: ("reject as IP evidence", "Source negative. 256.0.1.26 is inside the token tincan_register_device_256.0.1.26.113: invalid octet and an identifier, not a host address. ML false positive at its highest confidence."),
 44: ("low evidential value; no escalation alone", "Source negative. 0.0.0.0 is syntactically valid but is an unspecified placeholder, not a host. Both the ML model and the strict rule call it positive: syntax alone cannot see the intent."),
 12: ("retain as candidate lead; needs corroboration", "Source positive, standalone valid IPv4 in a plist string, not seen in training (no leakage). Public-looking address: record it as a lead only. Ownership and any malicious use are not established."),
 16: ("retain as candidate lead; needs corroboration", "Source positive, standalone valid IPv4, not seen in training. Same handling as above: a lead to corroborate with other artefacts, not a finding."),
 43: ("borderline: valid but reserved address", "Source positive. 255.255.255.255 is valid syntax but is the limited broadcast address, so it does not identify a remote host. Keep the label, flag it as reserved/low value."),
 11: ("borderline: likely version string", "Source positive. 135.0.0.22 is the first four parts of 135.0.0.22.118, which reads like an app build number. ML true positive but possibly for the wrong reason; the boundary rule rejects it."),
 9:  ("borderline: likely software version", "Source positive. 10.5.0.1 is a path segment (js/10.5.0.1.6.137/) in a vendor script URL: probably a library version. The label is doubtful; do not treat as network evidence."),
 33: ("borderline: truncated by regex", "Source positive. 15.21.2.145 is cut from 15.21.2.14594, so the candidate is an artefact of the extractor's 3-digit octet limit. Prefer a manual check of the original string."),
}
rv = p[p.test_row.isin(review)].copy()
rv["analyst_decision"] = rv.test_row.map(lambda r: review[r][0]); rv["analyst_rationale"] = rv.test_row.map(lambda r: review[r][1])
rv["ml_score"] = rv.ml_score_positive.round(3)
cols = ["test_row", "context", "candidate", "actual", "ml_pred", "ml_score", "outcome_ml", "base_strict", "in_train_overlap",
        "analyst_decision", "analyst_rationale"]
rv[cols].to_csv("outputs/analyst_review.csv", index=False)
print("\n=== E3 ANALYST REVIEW: %d records (%d source-negative, %d source-positive) ===" % (len(rv), (rv.actual == "negative").sum(), (rv.actual == POS).sum()))
for _, r in rv.sort_values("test_row").iterrows():
    print(f"\nrow {r.test_row:>2} | {r.candidate} | source={r.actual} | ML={r.ml_pred} ({r.ml_score}) {r.outcome_ml} | strict={r.base_strict} | seen_in_train={r.in_train_overlap}")
    print("   context :", r.context[:100])
    print("   decision:", r.analyst_decision); print("   why     :", r.analyst_rationale)

# ---- E4: imbalance counterexample ----
print("\n=== E4 WHY HIGH ACCURACY CAN STILL BE OPERATIONALLY POOR ===")
mj = res.loc["Majority class"]
print(f"REAL DATA: always answering 'positive' scores accuracy {mj.accuracy:.1%} on this holdout ({int((p.actual==POS).sum())} positive / {int((p.actual!=POS).sum())} negative) "
      f"yet rejects 0 of {int((p.actual!=POS).sum())} negatives (specificity {mj.specificity:.0%}).")
print(f"ML: accuracy {res.loc['ML (TF-IDF+LogReg)','accuracy']:.1%} but specificity only {res.loc['ML (TF-IDF+LogReg)','specificity']:.1%} - it clears just 4 of 12 non-IP strings.")
N, P = 1000, 10
print(f"\nHYPOTHETICAL ARITHMETIC (illustration only, not data from this lab): {N} recovered strings, {P} genuine IP evidence items. "
      f"A model that always says 'negative' is {(N-P)/N:.0%} accurate, recall = 0/{P} = 0%, and every evidential IP is missed.")
