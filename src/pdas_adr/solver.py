"""Mixed formulation of the optimality system for fixed active sets.

For given indicator functions chi_a, chi_b of the active sets, find (u, y, p) with

    B(y, v)  - (u, v)                    = (g, v)            state equation
    B*(p, q) - (y, q)                    = -(y_Omega, q)     adjoint equation
    (u, w) + (beta p, w)                 = (phi, w)          control equation

where beta = (1 - chi_a - chi_b) / lambda and phi = u_a chi_a + u_b chi_b, i.e.
u = -p/lambda on the inactive set and u = u_a (u_b) on the active sets.
All three equations are solved as one linear system.
"""

from dolfin import (Constant, DirichletBC, Function, FunctionSpace, MixedElement, TestFunctions,
                    TrialFunctions, assign, dx, grad, inner, solve)


def solve_optimality_system(mesh, PU, PY, beta, phi, yd, g, problem):
    """Return (u_h, y_h, p_h) for the active sets encoded in beta and phi."""
    eps, kappa, b = Constant(problem.eps), Constant(problem.kappa), Constant(problem.b)

    X = FunctionSpace(mesh, MixedElement([PU, PY, PY]))          # (u, y, p)
    zero = Constant(0.0)
    bcs = [DirichletBC(X.sub(1), zero, "on_boundary"),           # homogeneous Dirichlet for y
           DirichletBC(X.sub(2), zero, "on_boundary")]           # ... and for p

    u, y, p = TrialFunctions(X)
    w, v, q = TestFunctions(X)
    a_control = (beta * p * w + u * w) * dx
    a_state = (eps * inner(grad(y), grad(v)) + inner(b, grad(y)) * v + kappa * y * v - u * v) * dx
    a_adjoint = (eps * inner(grad(p), grad(q)) - inner(b, grad(p)) * q + kappa * p * q - y * q) * dx
    L = (phi * w + g * v - yd * q) * dx

    sol = Function(X)
    solve(a_control + a_state + a_adjoint == L, sol, bcs)

    uh, yh, ph = Function(FunctionSpace(mesh, PU)), Function(FunctionSpace(mesh, PY)), Function(FunctionSpace(mesh, PY))
    assign(uh, sol.sub(0))
    assign(yh, sol.sub(1))
    assign(ph, sol.sub(2))
    return uh, yh, ph
