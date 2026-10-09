"""Compare the control error of all configurations stored in results/ (one figure).

Run scripts/run_convergence.py for the four combinations of --lam {1, 0.01} and
--control {DG0, P1} first.

Usage:
    python scripts/compare_rates.py
"""

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CASES = [("lam1_DG0", r"DG0, $\lambda=1$ (bounds inactive)", "o-"),
         ("lam0.01_DG0", r"DG0, $\lambda=0.01$ (bounds active)", "s--"),
         ("lam1_P1", r"P1, $\lambda=1$ (bounds inactive)", "^-"),
         ("lam0.01_P1", r"P1, $\lambda=0.01$ (bounds active)", "v--")]


def main():
    fig, ax = plt.subplots(figsize=(7, 4.8))
    for tag, label, style in CASES:
        path = ROOT / "results" / f"convergence_{tag}.csv"
        if not path.exists():
            sys.exit(f"missing {path}; run scripts/run_convergence.py first")
        rows = list(csv.DictReader(open(path)))
        h = np.array([float(r["h_max"]) for r in rows])
        e = np.array([float(r["err_u_L2"]) for r in rows])
        rate = np.polyfit(np.log(h), np.log(e), 1)[0]
        ax.loglog(h, e, style, label=f"{label}: rate {rate:.2f}")
    ax.set_xlabel(r"$h_{\max}$")
    ax.set_ylabel(r"$\|u - u_h\|_{L^2(\Omega)}$")
    ax.set_title("Control error for different discretisations")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(ROOT / "figures" / "control_rates.png", dpi=150)
    print("saved figures/control_rates.png")


if __name__ == "__main__":
    main()
