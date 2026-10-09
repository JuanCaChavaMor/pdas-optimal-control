"""Model problem: distributed optimal control of the stationary ADR equation.

    min  J(y, u) = 1/2 ||y - y_Omega||^2 + lambda/2 ||u||^2
    s.t. -eps Lap y + b . grad y + kappa y = u + g   in Omega = (0, 1)^2,  y = 0 on the boundary,
         u_a <= u <= u_b  a.e. in Omega.

The data y_Omega and g are manufactured so that the exact solution is known:

    y = p = x1 x2 (1 - x1)(1 - x2),      u = min(u_b, max(u_a, -p / lambda)).

They were verified symbolically for eps = kappa = 1, b = (3, 4); the expressions
below are valid only for these coefficients (lambda, u_a, u_b are free).
"""

from dataclasses import dataclass

from dolfin import Expression

# Exact state and adjoint state (identical by construction)
Y_EXACT = "x[0]*x[1]*(1-x[0])*(1-x[1])"

# y_Omega = y - (-Lap p - b.grad p + p) with y = p
DESIRED_STATE = ("2*x[0] - 2*pow(x[0],2) + x[1] - 14*x[0]*x[1] + 8*x[1]*pow(x[0],2)"
                 " - pow(x[1],2) + 6*x[0]*pow(x[1],2)")

# -Lap y + b.grad y + y  (the source term is g = this - u)
STATE_OPERATOR_OF_Y = ("6*x[0] - 6*pow(x[0],2) + 5*x[1] - 13*x[0]*x[1] + 7*pow(x[0],2)*x[1]"
                       " - 5*pow(x[1],2) + 5*x[0]*pow(x[1],2) + pow(x[0],2)*pow(x[1],2)")


@dataclass
class ADRProblem:
    eps: float = 1.0           # diffusion
    b: tuple = (3.0, 4.0)      # divergence-free velocity field
    kappa: float = 1.0         # reaction
    lam: float = 1.0           # control cost (Tikhonov) parameter
    ua: float = -2.0           # lower control bound
    ub: float = 2.0            # upper control bound

    def __post_init__(self):
        if (self.eps, tuple(self.b), self.kappa) != (1.0, (3.0, 4.0), 1.0):
            raise ValueError("the manufactured data are only valid for eps = kappa = 1, b = (3, 4)")

    def u_exact_code(self):
        return f"min({self.ub}, max({self.ua}, -({Y_EXACT})/{self.lam}))"

    # --- expressions (degree = quadrature degree used by FEniCS)
    def y_exact(self, degree=4):
        return Expression(Y_EXACT, degree=degree)

    def p_exact(self, degree=4):
        return Expression(Y_EXACT, degree=degree)

    def u_exact(self, degree=3):
        return Expression(self.u_exact_code(), degree=degree)

    def desired_state(self, degree=1):
        return Expression(DESIRED_STATE, degree=degree)

    def source(self, degree=1):
        return Expression(f"{STATE_OPERATOR_OF_Y} - {self.u_exact_code()}", degree=degree)
