"""Active-set indicator functions and the projection of p onto the control space."""

import numpy as np
from dolfin import Function, FunctionSpace, TestFunction, TrialFunction, dx, interpolate, solve


def indicator_lower(f, U, ua):
    """chi_a: 1 where f < u_a (lower bound active), 0 elsewhere (dof-wise)."""
    chi = Function(U)
    chi.vector()[:] = np.where(f.vector()[:] < ua.vector()[:], 1.0, 0.0)
    return chi


def indicator_upper(f, U, ub):
    """chi_b: 1 where f > u_b (upper bound active), 0 elsewhere (dof-wise)."""
    chi = Function(U)
    chi.vector()[:] = np.where(f.vector()[:] > ub.vector()[:], 1.0, 0.0)
    return chi


def to_control_space(p, U):
    """Represent p in the control space U.

    For discontinuous controls this is the L2 projection (cell averages for DG0);
    for continuous P1 controls U coincides with the state space and p is copied.
    """
    if U.ufl_element().family() == "Discontinuous Lagrange":
        mesh = U.mesh()
        V = FunctionSpace(mesh, "DG", U.ufl_element().degree())
        u, v = TrialFunction(V), TestFunction(V)
        Pp = Function(V)
        solve(u * v * dx == p * v * dx, Pp)
        return Pp
    return interpolate(p, U)
