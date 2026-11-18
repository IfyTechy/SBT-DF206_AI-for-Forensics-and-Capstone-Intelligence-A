"""Phase A - Preserve and profile the evidence (ICDFA CIP-B107 LAB01).
Usage:  python3 notebooks/phase_a_preserve_profile.py originals
Reads files from the given folder (read-only) and writes outputs to ./outputs
"""
import sys, os, json, hashlib
from datetime import datetime, timezone, timedelta
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "originals"
OUT = "outputs"
OPERATOR = "Nebeuwa Ifeanyichukwu Raphael"
SOURCE = "ICDFA E-Campus assignment package (code.zip)"
FILES = ["dataset_small.csv", "dataset_train.jsonl", "dataset_test.jsonl",
         "result.csv", "fine_tuning_steps.txt"]
os.makedirs(OUT, exist_ok=True)

# ---- A1: chain-of-custody manifest ----
WAT = timezone(timedelta(hours=1))
rows = []
for f in FILES:
    p = os.path.join(SRC, f)
    h = hashlib.sha256(open(p, "rb").read()).hexdigest()
    now = datetime.now(timezone.utc)
    rows.append({"filename": f, "sha256": h, "size_bytes": os.path.getsize(p),
                 "collected_utc": now.strftime("%Y-%m-%d %H:%M:%S"),
                 "collected_wat": now.astimezone(WAT).strftime("%Y-%m-%d %H:%M:%S"),
                 "transfer_source": SOURCE, "examiner": OPERATOR})
man = pd.DataFrame(rows)
man.to_csv(f"{OUT}/evidence_manifest.csv", index=False)
print("=== A1 EVIDENCE MANIFEST ===")
print(man[["filename", "sha256", "size_bytes", "collected_wat"]].to_string(index=False))

# ---- A2: profile dataset_small.csv ----
df = pd.read_csv(os.path.join(SRC, "dataset_small.csv"))
print("\n=== A2 PROFILE dataset_small.csv ===")
print("records:", len(df), "| columns:", list(df.columns))
print("class distribution:\n", df["class"].value_counts().to_string())
print("positive (class != N):", (df["class"] != "N").sum(), "| negative (class == N):", (df["class"] == "N").sum())
print("null counts:", df.isna().sum().to_dict())
print("fully duplicated rows:", df.duplicated().sum(), "| unique ip values:", df["ip"].nunique())
print("rows whose ip is not 4 dot-separated parts:", (df["ip"].str.count(r"\.") != 3).sum())
print("negative examples:\n", df[df["class"] == "N"].drop_duplicates().head(10).to_string())

# ---- A3: validate JSONL files ----
def load(name):
    recs, bad = [], 0
    for i, line in enumerate(open(os.path.join(SRC, name)), 1):
        try:
            r = json.loads(line); assert set(r) == {"prompt", "completion"}; recs.append(r)
        except Exception:
            bad += 1; print("  invalid line", i, "in", name)
    return recs, bad
print("\n=== A3 JSONL VALIDATION ===")
tr, b1 = load("dataset_train.jsonl"); te, b2 = load("dataset_test.jsonl")
for n, r, b in [("train", tr, b1), ("test", te, b2)]:
    print(n, "records:", len(r), "| invalid lines:", b, "|",
          pd.Series([x["completion"] for x in r]).value_counts().to_dict())
print("split is 183/46:", (len(tr), len(te)) == (183, 46))
overlap = {x["prompt"] for x in tr} & {x["prompt"] for x in te}
print("exact prompts appearing in BOTH train and test (leakage risk):", len(overlap))

# ---- A4: CSV vs prompt/completion (5 examples, incl. negatives) ----
print("\n=== A4 CSV -> PROMPT/COMPLETION EXAMPLES ===")
sample = pd.concat([df[df["class"] == "N"].drop_duplicates().head(2),
                    df[df["class"] != "N"].drop_duplicates().sample(3, random_state=42)])
for _, r in sample.iterrows():
    print("CSV row     :", dict(r))
    print("prompt      :", repr(r["text"] + ";" + r["ip"] + " ->"))
    print("completion  :", repr(" negative" if r["class"] == "N" else " positive"), "\n")
