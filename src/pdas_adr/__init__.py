"""Primal-dual active set strategy for optimal control of the ADR equation (legacy FEniCS)."""

from .problem import ADRProblem
from .pdas import PDASResult, convergence_study, pdas, rates
