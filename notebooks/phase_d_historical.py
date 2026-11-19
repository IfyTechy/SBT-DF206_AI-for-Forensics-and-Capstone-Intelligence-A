"""Phase D - Historical fine-tuning evidence (ICDFA CIP-B107 LAB01).
Usage: python3 notebooks/phase_d_historical.py originals
These are SUPPLIED historical logs. They are NOT a model run by the student.
"""
import sys, os
import pandas as pd, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SRC = sys.argv[1] if len(sys.argv) > 1 else "originals"
os.makedirs("outputs/charts", exist_ok=True)
pd.set_option("display.width", 220); pd.set_option("display.max_columns", 30)
st = pd.read_csv(os.path.join(SRC, "fine_tuning_steps.txt"))
rs = pd.read_csv(os.path.join(SRC, "result.csv"))

print("=== D1 STRUCTURE AND MISSING VALUES (historical files) ===")
print("fine_tuning_steps.txt:", st.shape, "| steps", st.step.min(), "-", st.step.max(), "| columns:", list(st.columns))
print("result.csv           :", rs.shape, "| steps", rs.step.min(), "-", rs.step.max())
miss = rs.isna().sum(); print("result.csv non-null counts for sparse columns:")
print(rs[["validation_loss", "validation_sequence_accuracy", "validation_token_accuracy",
          "classification/accuracy", "classification/precision", "classification/recall",
          "classification/auroc", "classification/auprc", "classification/f1.0"]].notna().sum().to_string())
print("training_loss identical in both files?", bool((st.training_loss.values[:len(rs)] == rs.training_loss.values).all()),
      "(different values = two different logged runs)")

print("\n=== D2 TRAINING LOSS TREND ===")
for name, d in [("fine_tuning_steps.txt", st), ("result.csv", rs)]:
    print(f"{name}: first-10 mean {d.training_loss.head(10).mean():.4f} | last-10 mean {d.training_loss.tail(10).mean():.4f} | "
          f"min {d.training_loss.min():.4f} (step {int(d.loc[d.training_loss.idxmin(), 'step'])}) | max {d.training_loss.max():.4f}")
    late = d[d.step > 100]
    print(f"   after step 100: mean {late.training_loss.mean():.4f}, steps with loss > 0.10: {int((late.training_loss > 0.10).sum())}")
    first1 = d[d.training_sequence_accuracy == 1.0].step.min()
    print(f"   first step with training sequence accuracy = 1.0: {int(first1)}; share of steps at 1.0: {(d.training_sequence_accuracy == 1.0).mean():.1%}")

val = rs.dropna(subset=["validation_loss"])
n = len(val); a, b = val.head(n // 3), val.tail(n // 3)
print(f"\nvalidation_loss logged at {n} steps: first third mean {a.validation_loss.mean():.4f} | last third mean {b.validation_loss.mean():.4f} | "
      f"max {val.validation_loss.max():.4f} (step {int(val.loc[val.validation_loss.idxmax(), 'step'])})")

print("\n=== D3 CLASSIFICATION METRICS RECORDED IN result.csv (non-null rows only) ===")
cols = ["step", "classification/accuracy", "classification/precision", "classification/recall", "classification/f1.0",
        "classification/auroc", "classification/auprc"]
m = rs.dropna(subset=["classification/accuracy"])[cols]
print(m.round(4).to_string(index=False))
print("rows with a value: %d of %d (%.1f%% of steps have no classification metric)" % (len(m), len(rs), 100 * (1 - len(m) / len(rs))))
fin = m.iloc[-1]
print("\nFINAL non-null recorded metrics (step %d) - HISTORICAL, not a new run:" % int(fin.step))
for c in cols[1:]: print("  %-26s %.6f" % (c, fin[c]))

# ---- charts ----
w = 20
fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(st.step, st.training_loss, alpha=.25, label="fine_tuning_steps.txt (raw)")
ax.plot(st.step, st.training_loss.rolling(w, min_periods=1).mean(), label=f"fine_tuning_steps.txt ({w}-step mean)")
ax.plot(rs.step, rs.training_loss.rolling(w, min_periods=1).mean(), label=f"result.csv ({w}-step mean)")
ax.set_xlabel("step"); ax.set_ylabel("training loss"); ax.set_title("Historical training loss vs step"); ax.legend(); ax.grid(alpha=.3)
fig.savefig("outputs/charts/training_loss_vs_step.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(rs.step, rs.training_loss.rolling(w, min_periods=1).mean(), label=f"training loss ({w}-step mean)")
ax.plot(val.step, val.validation_loss, "o-", ms=3, alpha=.7, label="validation loss (logged steps only)")
ax.set_xlabel("step"); ax.set_ylabel("loss"); ax.set_title("result.csv: training vs validation loss"); ax.legend(); ax.grid(alpha=.3)
fig.savefig("outputs/charts/training_vs_validation_loss.png", dpi=150, bbox_inches="tight"); plt.close(fig)

fig, ax = plt.subplots(figsize=(9, 4.5))
for c in cols[1:5]: ax.plot(m.step, m[c], "o-", label=c.replace("classification/", ""))
ax.set_ylim(0.9, 1.01); ax.set_xlabel("step (only 4 checkpoints recorded)"); ax.set_ylabel("score")
ax.set_title("Historical classification metrics vs step (result.csv)"); ax.legend(); ax.grid(alpha=.3)
fig.savefig("outputs/charts/classification_metrics_vs_step.png", dpi=150, bbox_inches="tight"); plt.close(fig)
print("\nsaved: outputs/charts/training_loss_vs_step.png, training_vs_validation_loss.png, classification_metrics_vs_step.png")
