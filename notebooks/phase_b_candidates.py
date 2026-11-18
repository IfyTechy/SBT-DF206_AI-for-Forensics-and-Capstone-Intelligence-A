"""Phase B - IP candidate extraction and labelling (ICDFA CIP-B107 LAB01).
Usage: python3 notebooks/phase_b_candidates.py originals
"""
import sys, os, re, ipaddress
import pandas as pd

SRC = sys.argv[1] if len(sys.argv) > 1 else "originals"
LOOSE = re.compile(r"\d{1,3}(?:\.\d{1,3}){3}")          # extractor: matches inside longer strings too

def strict_valid(s):
    """Independent validator: strict IPv4 syntax (Python rejects leading zeros / octets > 255)."""
    try:
        ipaddress.IPv4Address(s); return True
    except ValueError:
        return False

def classify(context, cand, valid):
    i = context.find(cand)
    before = context[i-1] if i > 0 else ""
    after = context[i+len(cand)] if i + len(cand) < len(context) else ""
    if i == -1:
        return "candidate_not_found_in_context"
    if not valid:
        return "invalid_ipv4_syntax"
    if before.isdigit() or before == "." or after.isdigit() or after == ".":
        return "prefix_suffix_contamination"
    if re.match(r"<(string|key)>$", context[:i]) and re.match(r"</(string|key)>$", context[i+len(cand):]):
        return "clean_standalone"
    plain = re.sub(r"</?(string|key)>", "", context)       # ignore the plist tags themselves
    if after == ":":
        return "ip_with_port"
    if "http" in plain or "/" in plain:
        return "embedded_in_url_or_path"
    return "embedded_in_token_or_version"

df = pd.read_csv(os.path.join(SRC, "dataset_small.csv"))
rows = []
for n, r in df.iterrows():
    ctx = str(r["text"]).strip()
    cand = str(r["ip"])
    found = LOOSE.findall(ctx)
    regex_match = cand in found                 # does the extractor reproduce the candidate from its context?
    valid = strict_valid(cand)
    label = "negative" if r["class"] == "N" else "positive"
    cat = classify(ctx, cand, valid)
    note = f"[{cat}] regex={'Y' if regex_match else 'N'}, strictIPv4={'Y' if valid else 'N'}, source={label}"
    if label == "positive" and not valid:
        note += " | MISMATCH: source positive but NOT valid IPv4 - preserved, not relabelled"
    elif label == "negative" and valid:
        note += " | MISMATCH: source negative but syntactically valid IPv4 - context decides"
    elif label == "positive" and cat == "prefix_suffix_contamination":
        note += " | CHECK: valid IPv4 text but may be part of a longer version number"
    rows.append({"source_row": n + 2, "context": ctx, "candidate": cand, "regex_match": regex_match,
                 "ipv4_valid": valid, "source_label": label, "analyst_note": note, "_cat": cat})
rev = pd.DataFrame(rows)
os.makedirs("outputs", exist_ok=True)
rev.drop(columns="_cat").to_csv("outputs/candidate_review.csv", index=False)

print("=== B1 CATEGORY COUNTS ===")
print(rev["_cat"].value_counts().to_string())
print("\n=== B2 THREE CONCEPTS COMPARED (regex vs strict IPv4 vs source label) ===")
print(pd.crosstab([rev["regex_match"], rev["ipv4_valid"]], rev["source_label"]).to_string())
mm = rev[((rev.source_label == "positive") & ~rev.ipv4_valid) | ((rev.source_label == "negative") & rev.ipv4_valid)]
print("\nsource-label vs strict-validity mismatches:", len(mm), "of", len(rev))
print("\n=== B3 DEMO SAMPLES (>=10, one or two per category) ===")
demo = rev.drop_duplicates(subset=["context", "candidate"]).groupby("_cat").head(3)
pd.set_option("display.width", 250); pd.set_option("display.max_colwidth", 70)
print(demo[["source_row", "context", "candidate", "regex_match", "ipv4_valid", "source_label", "_cat"]].to_string(index=False))
print("\nsamples shown:", len(demo), "| saved: outputs/candidate_review.csv (", len(rev), "rows )")
