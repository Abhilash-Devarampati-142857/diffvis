import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import cumulative_trapezoid, trapezoid
from .parser import make_evaluator


def integral(
    expr_or_func,
    x_range=(0.0, 2.0),
    points: int = 300,
    antiderivative: bool = False,
    save: str | None = None,
    show: bool = True,
):
    """
    Interactive plot of f(x), shaded area over x_range,
    and optional cumulative antiderivative F(x).
    """
    f = make_evaluator(expr_or_func, ["x"])
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.asarray(f(x), dtype=float)
    area = float(trapezoid(y, x))

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(x, y, linewidth=2, label="f(x)")
    ax.fill_between(x, y, 0, alpha=0.25, label=f"Area ≈ {area:.4f}")

    F = None
    if antiderivative:
        F = cumulative_trapezoid(y, x, initial=0.0)
        ax.plot(x, F, "--", linewidth=1.8, label="Antiderivative F(x) (F(a)=0)")

    ax.axhline(0, color="gray", linewidth=0.7)
    ax.set_xlabel("x")
    ax.set_ylabel("Value")
    ax.set_title(f"Integral on [{x_range[0]}, {x_range[1]}]")
    ax.legend()
    plt.tight_layout()

    if save:
        plt.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return {"x": x, "y": y, "area": area, "antiderivative": F}
