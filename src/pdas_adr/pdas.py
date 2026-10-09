"""Primal-dual active set (PDAS) strategy for the box-constrained ADR control problem.

With mu = -(p + lambda u) / lambda, the projection formula u = P_[ua,ub](-p/lambda)
is equivalent to choosing the active sets

    A_a = { u + mu < u_a } = { -p/lambda < u_a },   A_b = { u + mu > u_b } = { -p/lambda > u_b }.

Each iteration solves the optimality system for the current active sets and
updates them; the method stops when the active sets no longer change.

Note: the implementation in the thesis used mu = p/lambda - u (sign of p flipped).
This has no effect on the thesis test problem, where the bounds are never active,
but gives wrong active sets as soon as a bound becomes active.
"""

from dataclasses import dataclass

import numpy as np
from dolfin import (FiniteElement, Function, FunctionSpace, UnitSquareMesh, assemble, dx,
                    errornorm, interpolate, Constant)

from .active_set import indicator_lower, indicator_upper, to_control_space
from .solver import solve_optimality_system


@dataclass
class PDASResult:
    u: Function
    y: Function
    p: Function
    iterations: int
    converged: bool
    active_measure: float      # |A_a| + |A_b|, area where a bound is active


def pdas(problem, mesh, control="DG0", tol=1e-9, maxiter=100, u0=-100.0, mu0=-100.0, verbose=False):
    """Solve the discrete control problem on ``mesh``; control = "DG0" or "P1"."""
    family, degree = {"DG0": ("DG", 0), "P1": ("Lagrange", 1)}[control]
    PU = FiniteElement(family, mesh.ufl_cell(), degree)
    PY = FiniteElement("Lagrange", mesh.ufl_cell(), 1)
    U, Y = FunctionSpace(mesh, PU), FunctionSpace(mesh, PY)

    yd = interpolate(problem.desired_state(), Y)
    g = interpolate(problem.source(), Y)
    ua = interpolate(Constant(problem.ua), U)
    ub = interpolate(Constant(problem.ub), U)
    one = interpolate(Constant(1.0), U)

    # initial active sets from the initial guess u0 + mu0
    z = interpolate(Constant(u0 + mu0), U)
    chi_a, chi_b = indicator_lower(z, U, ua), indicator_upper(z, U, ub)

    for it in range(1, maxiter + 1):
        beta = (one - chi_a - chi_b) / problem.lam
        phi = ua * chi_a + ub * chi_b
        u, y, p = solve_optimality_system(mesh, PU, PY, beta, phi, yd, g, problem)

        # u + mu = -p / lambda, represented in the control space
        z.vector()[:] = -to_control_space(p, U).vector()[:] / problem.lam
        new_a, new_b = indicator_lower(z, U, ua), indicator_upper(z, U, ub)
        change = np.sqrt(assemble((chi_a - new_a) ** 2 * dx)) + np.sqrt(assemble((chi_b - new_b) ** 2 * dx))
        chi_a, chi_b = new_a, new_b
        if verbose:
            print(f"  iteration {it}: change of active sets = {change:.3e}")
        if change < tol:
            return PDASResult(u, y, p, it, True, assemble((chi_a + chi_b) * dx))
    return PDASResult(u, y, p, maxiter, False, assemble((chi_a + chi_b) * dx))


def convergence_study(problem, control="DG0", refinements=5, n0=16, verbose=True):
    """Solve on n0 x n0, 2 n0 x 2 n0, ... meshes and record the errors.

    Returns a list of dicts with the mesh parameter, iterations and the errors
    ||u - u_h||_L2, ||y - y_h||_H1, ||p - p_h||_H1.
    """
    rows, last = [], None   # last: solution on the finest mesh
    for j in range(refinements):
        n = n0 * 2 ** j
        mesh = UnitSquareMesh(n, n)
        res = pdas(problem, mesh, control)
        rows.append({
            "n": n,
            "h_max": mesh.hmax(),
            "iterations": res.iterations,
            "active_measure": res.active_measure,
            "err_u_L2": errornorm(problem.u_exact(), res.u, "L2"),
            "err_y_H1": errornorm(problem.y_exact(), res.y, "H1"),
            "err_p_H1": errornorm(problem.p_exact(), res.p, "H1"),
        })
        last = res
        if verbose:
            r = rows[-1]
            print(f"n={n:4d}  iters={r['iterations']}  |u-u_h|_L2={r['err_u_L2']:.4e}  "
                  f"|y-y_h|_H1={r['err_y_H1']:.4e}  |p-p_h|_H1={r['err_p_H1']:.4e}", flush=True)
    return rows, last


def rates(h, err):
    """Least-squares slope of log(err) against log(h)."""
    return float(np.polyfit(np.log(h), np.log(err), 1)[0])
