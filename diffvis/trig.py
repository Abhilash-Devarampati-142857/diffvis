import matplotlib.pyplot as plt
import numpy as np
from .parser import clean_math_expr
from .ui_controls import attach_interactive_ui, get_palette

TRIG_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "sec": lambda x: 1.0 / np.cos(x),
    "csc": lambda x: 1.0 / np.sin(x),
    "cot": lambda x: np.cos(x) / np.sin(x),
    "asin": np.arcsin,
    "acos": np.arccos,
    "atan": np.arctan,
    "sinh": np.sinh,
    "cosh": np.cosh,
    "tanh": np.tanh,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
}


def _make_trig_evaluator(expr_or_func):
    if callable(expr_or_func):
        return expr_or_func
    code = compile(clean_math_expr(str(expr_or_func)), "<diffvis_trig>", "eval")

    def eval_trig(x):
        val = eval(code, TRIG_GLOBALS, {"x": x})
        if np.ndim(val) == 0:
            return np.full_like(np.asarray(x, dtype=float), float(val), dtype=float)
        return np.asarray(val, dtype=float)

    return eval_trig


def _mask_discontinuities(y: np.ndarray, threshold: float = 25.0) -> np.ndarray:
    y_clean = y.copy()
    jumps = np.abs(np.diff(y_clean)) > threshold
    y_clean[1:][jumps] = np.nan
    return y_clean


def trig(
    expr_or_func="sin(x)",
    x_range=(-2.0 * np.pi, 2.0 * np.pi),
    dim: str = "2d",
    color_mode: str = "Classic",
    points: int = 400,
    show_slope: bool = True,
    tangent_at: float | None = None,
    pi_ticks: bool = True,
    save: str | None = None,
    show: bool = True,
):
    f = _make_trig_evaluator(expr_or_func)
    x = np.linspace(x_range[0], x_range[1], points)
    y = f(x)
    dy_dx = np.gradient(y, x)
    y_plot = _mask_discontinuities(y)
    dy_plot = _mask_discontinuities(dy_dx)
    label_expr = expr_or_func if isinstance(expr_or_func, str) else "trig(x)"

    def redraw(fig, target_dim: str, active_color: str):
        pal = get_palette(active_color)
        for ax in list(fig.axes):
            if ax not in getattr(fig, "_diffvis_ctrl_axes", []):
                ax.remove()

        if target_dim == "2d":
            ax = fig.add_subplot(111)
            ax.plot(x, y_plot, color=pal["line1"], linewidth=2.2, label=f"y = {label_expr}")
            if show_slope:
                ax.plot(x, dy_plot, "--", color=pal["line2"], linewidth=1.7, label="Slope: dy/dx")
            if tangent_at is not None and np.isfinite(f(np.array([tangent_at], dtype=float))[0]):
                y0 = float(f(np.array([tangent_at], dtype=float))[0])
                h = 1e-5
                m = float(
                    (
                        f(np.array([tangent_at + h], dtype=float))[0]
                        - f(np.array([tangent_at - h], dtype=float))[0]
                    )
                    / (2.0 * h)
                )
                span = min(1.5, 0.2 * (x_range[1] - x_range[0]))
                xt = np.linspace(tangent_at - span, tangent_at + span, 50)
                yt = y0 + m * (xt - tangent_at)
                ax.plot(
                    xt,
                    yt,
                    ":",
                    color=pal["accent"],
                    linewidth=2.2,
                    label=f"Tangent @ x={tangent_at:.2f} (m={m:.3f})",
                )
                ax.scatter([tangent_at], [y0], color=pal["accent"], zorder=5)
            if pi_ticks:
                step = np.pi / 2.0
                k_start = int(np.floor(x_range[0] / step))
                k_end = int(np.ceil(x_range[1] / step))
                ticks = np.arange(k_start, k_end + 1) * step
                tick_labels = []
                for k in range(k_start, k_end + 1):
                    if k == 0:
                        tick_labels.append("0")
                    elif k % 2 == 0:
                        half = k // 2
                        tick_labels.append(
                            "π" if half == 1 else ("-π" if half == -1 else f"{half}π")
                        )
                    else:
                        tick_labels.append(
                            "π/2" if k == 1 else ("-π/2" if k == -1 else f"{k}π/2")
                        )
                ax.set_xticks(ticks)
                ax.set_xticklabels(tick_labels)
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_xlabel("x (radians)")
            ax.set_ylabel("Value")
            ax.set_title(f"[2D Trig] {label_expr}  [{active_color}]")
            ax.legend(fontsize=8.5)
        else:
            ax = fig.add_subplot(111, projection="3d")
            slope_vals = dy_plot if show_slope else np.zeros_like(x)
            ax.plot(x, slope_vals, y_plot, color=pal["line1"], linewidth=2.2, label="Curve: (x, dy/dx, y)")
            ax.plot(
                x,
                slope_vals,
                np.zeros_like(x),
                "--",
                color=pal["line2"],
                alpha=0.45,
                linewidth=1.1,
                label="z = 0 projection",
            )
            ax.set_xlabel("x")
            ax.set_ylabel("Slope: dy/dx" if show_slope else "y (0)")
            ax.set_zlabel("y")
            ax.set_title(f"[3D Trig] {label_expr}  [{active_color}] — Drag to rotate")
            ax.legend(loc="upper left", fontsize=8)

        fig.subplots_adjust(left=0.24, right=0.97, top=0.90, bottom=0.12)
        if show and hasattr(fig, "_diffvis_build_controls"):
            fig._diffvis_build_controls()

    fig = plt.figure(figsize=(8.8, 5.4))
    start_dim = "3d" if str(dim).lower().strip() == "3d" else "2d"
    if show:
        attach_interactive_ui(fig, start_dim, color_mode, redraw)
    redraw(fig, start_dim, color_mode)

    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return {"x": x, "y": y, "dy_dx": dy_dx, "dim": start_dim}
