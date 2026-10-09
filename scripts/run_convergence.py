"""Convergence study of the PDAS method for the ADR control problem.

Reproduces Table 6.2 of the thesis with the default settings (lambda = 1,
piecewise constant control). Use --lam 0.01 for a test case in which the
control bounds are active on about 60 % of the domain, and --control P1 for
continuous piecewise linear controls (the discretisation used by cashocs).

Usage:
    python scripts/run_convergence.py [--lam 1.0] [--control DG0|P1] [--refinements 5]
"""

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import matplotlib  # noqa: E402
matplotlib.use("Agg")
from dolfin import LogLevel, set_log_level  # noqa: E402

from pdas_adr import ADRProblem, convergence_study, rates  # noqa: E402
from pdas_adr.plotting import plot_convergence, plot_solution  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lam", type=float, default=1.0, help="control cost parameter lambda")
    parser.add_argument("--control", choices=["DG0", "P1"], default="DG0")
    parser.add_argument("--refinements", type=int, default=5, help="meshes 16x16, 32x32, ...")
    args = parser.parse_args()
    set_log_level(LogLevel.ERROR)

    problem = ADRProblem(lam=args.lam)
    tag = f"lam{args.lam:g}_{args.control}"
    print(f"lambda = {args.lam}, control = {args.control}")
    rows, finest = convergence_study(problem, args.control, args.refinements)

    h = [r["h_max"] for r in rows]
    fitted = {k: rates(h, [r[k] for r in rows]) for k in ("err_u_L2", "err_y_H1", "err_p_H1")}
    print("rates:", {k: round(v, 3) for k, v in fitted.items()})
    print(f"area where a bound is active: {rows[-1]['active_measure']:.3f}")

    (ROOT / "results").mkdir(exist_ok=True)
    with open(ROOT / "results" / f"convergence_{tag}.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    (ROOT / "figures").mkdir(exist_ok=True)
    plot_convergence(rows, fitted, ROOT / "figures" / f"convergence_{tag}.png",
                     title=f"PDAS, $\\lambda$ = {args.lam:g}, {args.control} control")
    plot_solution(finest, ROOT / "figures" / f"solution_{tag}.png",
                  title=f"$\\lambda$ = {args.lam:g}, {args.control} control, {rows[-1]['n']}x{rows[-1]['n']} mesh")
    print("saved results/ and figures/")


if __name__ == "__main__":
    main()
