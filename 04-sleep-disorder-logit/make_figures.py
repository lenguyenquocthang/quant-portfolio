"""Figures for the sleep-disorder logit (reads results/sleep_results.json)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = json.load(open("results/sleep_results.json")); plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
cv = o["cv"]; lab = [c["model"].split(" ")[0] for c in cv]; x = np.arange(len(cv)); fig, ax = plt.subplots(figsize=(6.2, 3.3))
for dx, k, c, n in [(-.27, "auc_in", "#bdbdbd", "In-sample"), (0, "auc_cvnaive", "#9ecae1", "Naive 5-fold CV (leaky)"), (.27, "auc_cvgroup", "#08519c", "Grouped CV (no duplicate profiles)")]:
    ax.bar(x+dx, [c_[k] for c_ in cv], .26, color=c, label=n)
ax.set_xticks(x); ax.set_xticklabels(lab); ax.set_ylim(.5, 1); ax.set_ylabel("AUC"); ax.legend(frameon=False, fontsize=7, loc="lower right")
fig.tight_layout(); fig.savefig("results/figures/fig1_auc.png", dpi=200)
