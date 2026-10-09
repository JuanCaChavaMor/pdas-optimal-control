"""Solve the same ADR control problem with the PDAS solver of cashocs 1.3 (Table 6.3 of the thesis).

cashocs discretises control, state and adjoint with continuous P1 elements and
uses the adjoint sign convention of Hinze et al., i.e. its adjoint equals -p.
PDAS was removed from cashocs in later versions, so cashocs 1.3.x is required.

Usage:
    python scripts/run_cashocs.py [--lam 1.0] [--refinements 5]
"""

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "scripts"))

import _ufl_compat  # noqa: E402,F401  (must come before fenics/cashocs)
import numpy as np  # noqa: E402
import cashocs  # noqa: E402
from fenics import (Constant, Expression, Function, FunctionSpace, errornorm, grad, inner,  # noqa: E402
                    interpolate, set_log_level, LogLevel)

from pdas_adr import ADRProblem, rates  # noqa: E402
from pdas_adr.problem import Y_EXACT  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--lam", type=float, default=1.0)
    parser.add_argument("--refinements", type=int, default=5)
    args = parser.parse_args()
    sys.argv = sys.argv[:1]          # cashocs 1.3 parses sys.argv itself (remeshing option)
    set_log_level(LogLevel.ERROR)

    problem = ADRProblem(lam=args.lam)
    config = cashocs.load_config(str(ROOT / "config" / "config_box.ini"))
    eps, kappa, b = Constant(problem.eps), Constant(problem.kappa), Constant(problem.b)
    p_exact = Expression(f"-({Y_EXACT})", degree=4)      # cashocs adjoint = -p

    h, errs = [], {"u_L2": [], "y_H1": [], "p_H1": []}
    for j in range(args.refinements):
        n = 16 * 2 ** j
        mesh, subdomains, boundaries, dx, ds, dS = cashocs.regular_mesh(n)
        V = FunctionSpace(mesh, "CG", 1)
        y, p, u = Function(V), Function(V), Function(V)
        u_a = interpolate(Constant(problem.ua), V)
        u_b = interpolate(Constant(problem.ub), V)

        g, yd = problem.source(), problem.desired_state()
        e = (eps * inner(grad(y), grad(p)) + inner(b, grad(y)) * p + kappa * y * p - u * p - g * p) * dx
        bcs = cashocs.create_bcs_list(V, Constant(0), boundaries, [1, 2, 3, 4])
        J = Constant(0.5) * (y - yd) * (y - yd) * dx + Constant(0.5 * problem.lam) * u * u * dx

        ocp = cashocs.OptimalControlProblem(e, bcs, J, y, u, p, config, control_constraints=[u_a, u_b])
        ocp.solve("pdas", 1e-9, 0.0, 100)
        assert np.all(u_a.vector()[:] <= u.vector()[:]) and np.all(u.vector()[:] <= u_b.vector()[:])

        h.append(mesh.hmax())
        errs["u_L2"].append(errornorm(problem.u_exact(), u, "L2", degree_rise=2))
        errs["y_H1"].append(errornorm(problem.y_exact(), y, "H1", degree_rise=2))
        errs["p_H1"].append(errornorm(p_exact, p, "H1", degree_rise=2))
        print(f"n={n:4d}  |u-u_h|_L2={errs['u_L2'][-1]:.4e}  |y-y_h|_H1={errs['y_H1'][-1]:.4e}  "
              f"|p-p_h|_H1={errs['p_H1'][-1]:.4e}", flush=True)

    print("rates:", {k: round(rates(h, v), 3) for k, v in errs.items()})


if __name__ == "__main__":
    main()
