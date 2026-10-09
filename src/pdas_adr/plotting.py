"""Plots: solution fields and convergence rates."""

import matplotlib.pyplot as plt
import numpy as np
from dolfin import plot


def plot_solution(res, path, title=""):
    """Control, state and adjoint state side by side."""
    fig = plt.figure(figsize=(15, 4.5))
    for k, (f, name) in enumerate([(res.u, "Control u"), (res.y, "State y"), (res.p, "Adjoint state p")]):
        plt.subplot(1, 3, k + 1)
        c = plot(f)
        plt.colorbar(c, fraction=0.046, pad=0.04)
        plt.title(name)
    if title:
        fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def slope_triangle(ax, x0, y0, slope, size=0.25, below=True):
    """Draw a small slope triangle with lower-left corner (x0, y0) on log-log axes."""
    x1 = x0 * 10 ** size
    y1 = y0 * (x1 / x0) ** slope
    ax.fill([x0, x1, x1], [y0, y0, y1], color="0.85", zorder=0)
    ax.text(x1 * 1.05, np.sqrt(y0 * y1), f"{slope:g}", va="center", fontsize=9)


def plot_convergence(rows, rates, path, title="Convergence"):
    """Errors vs. h_max on log-log axes with the fitted rates in the legend."""
    h = np.array([r["h_max"] for r in rows])
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    for key, label, marker in [("err_u_L2", r"$\|u-u_h\|_{L^2}$", "o"),
                               ("err_y_H1", r"$\|y-y_h\|_{H^1}$", "D"),
                               ("err_p_H1", r"$\|p-p_h\|_{H^1}$", "x")]:
        e = np.array([r[key] for r in rows])
        ax.loglog(h, e, f"--{marker}", label=f"{label}  (rate {rates[key]:.2f})")
    e_u = np.array([r["err_u_L2"] for r in rows])
    slope_triangle(ax, h[-2], e_u[-2] * 0.5, round(rates["err_u_L2"]))
    ax.set_xlabel(r"$h_{\max}$")
    ax.set_ylabel("Error")
    ax.set_title(title)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)
