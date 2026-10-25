"""Figures for the practicum re-estimation (reads results/thnn_results.json)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = json.load(open("results/thnn_results.json")); plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
# within (FE) vs between estimates: SIZE and CAP effects are cross-sectional, not within-bank
fig, ax = plt.subplots(figsize=(6.2, 3.2)); ks = ["SIZE", "CAP", "NPL"]; x = np.arange(3)
ax.bar(x-.18, [o["between"][k][0] for k in ks], .36, color="#08519c", label="Between (bank means)")
ax.bar(x+.18, [o["tab"]["fe"][k][0] for k in ks], .36, color="#d94801", label="Within (fixed effects)")
ax.axhline(0, color="k", lw=.8); ax.set_xticks(x); ax.set_xticklabels(ks); ax.set_ylabel("Coefficient on NIM"); ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig("results/figures/fig1_within_between.png", dpi=200)
