"""Figures for the single-index model re-estimation (reads results/sim_results.json)."""
import json, numpy as np, matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
o = json.load(open("results/sim_results.json")); plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False})
S = o["sim"]; b = [r["bank"] for r in S]; be = np.array([r["beta"] for r in S]); lo = np.array([r["lo"] for r in S]); hi = np.array([r["hi"] for r in S])
i = np.argsort(be); fig, ax = plt.subplots(figsize=(6.2, 3.4))
ax.errorbar(np.arange(12), be[i], yerr=[be[i]-lo[i], hi[i]-be[i]], fmt="o", color="#08519c", capsize=3)
ax.axhline(1, color="r", ls="--", lw=1); ax.set_xticks(range(12)); ax.set_xticklabels([b[j] for j in i], fontsize=8)
ax.set_ylabel(r"$\beta$ vs VN30 (95% HAC CI)"); fig.tight_layout(); fig.savefig("results/figures/fig1_betas.png", dpi=200)
e = o["egp"]; fig, ax = plt.subplots(figsize=(4.6, 3.2)); v = [e["egp_sharpe"], e["ew_sharpe"], e["vn30_sharpe"]]
ax.bar(["EGP optimal", "Equal weight", "VN30"], v, color=["#d94801", "#9ecae1", "#08519c"]); ax.set_ylabel("Out-of-sample Sharpe (2023-24)")
fig.tight_layout(); fig.savefig("results/figures/fig2_sharpe.png", dpi=200)
