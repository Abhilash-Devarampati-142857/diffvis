import matplotlib.pyplot as plt
import numpy as np
from .parser import make_evaluator
from .ui_controls import attach_interactive_ui, get_palette


def slope_at(expr_or_func, x0: float, h: float = 1e-5) -> float:
    f = make_evaluator(expr_or_func, ["x"])
    x_arr = np.array([x0 - h, x0 + h], dtype=float)
    y_arr = np.asarray(f(x_arr), dtype=float)
    return float((y_arr[1] - y_arr[0]) / (2.0 * h))


def normal(
    expr_or_func,
    x_range=(-5.0, 5.0),
    dim: str = "2d",
    color_mode: str = "Classic",
    points: int = 300,
    show_slope: bool = True,
    tangent_at: float | None = None,
    save: str | None = None,
    show: bool = True,
):
    f = make_evaluator(expr_or_func, ["x"])
    x = np.linspace(x_range[0], x_range[1], points)
    y = np.asarray(f(x), dtype=float)
    dy_dx = np.gradient(y, x)
    label_expr = expr_or_func if isinstance(expr_or_func, str) else "f(x)"

    y0 = m0 = None
    if tangent_at is not None:
        y0 = float(np.asarray(f(np.array([tangent_at], dtype=float)))[0])
        m0 = slope_at(f, tangent_at)

    def redraw(fig, target_dim: str, active_color: str):
        pal = get_palette(active_color)
        for ax in list(fig.axes):
            if ax not in getattr(fig, "_diffvis_ctrl_axes", []):
                ax.remove()

        if target_dim == "2d":
            ax = fig.add_subplot(111)
            ax.plot(x, y, color=pal["line1"], label="f(x)", linewidth=2.2)
            if show_slope:
                ax.plot(x, dy_dx, "--", color=pal["line2"], label="Slope: df/dx", linewidth=1.7)
            if tangent_at is not None and y0 is not None and m0 is not None:
                x_tan = np.linspace(x_range[0], x_range[1], 100)
                y_tan = y0 + m0 * (x_tan - tangent_at)
                span = 0.25 * (x_range[1] - x_range[0])
                mask = np.abs(x_tan - tangent_at) <= span
                ax.plot(
                    x_tan[mask],
                    y_tan[mask],
                    ":",
                    color=pal["accent"],
                    linewidth=2.2,
                    label=f"Tangent @ x={tangent_at} (m={m0:.3f})",
                )
                ax.scatter([tangent_at], [y0], color=pal["accent"], zorder=5)
            ax.axhline(0, color="gray", linewidth=0.7)
            ax.axvline(0, color="gray", linewidth=0.7)
            ax.set_xlabel("x")
            ax.set_ylabel("Value")
            ax.set_title(f"[2D Normal] {label_expr}  [{active_color}]")
            ax.legend(fontsize=8.5)
        else:
            ax = fig.add_subplot(111, projection="3d")
            y_axis_vals = dy_dx if show_slope else np.zeros_like(x)
            ax.plot(x, y_axis_vals, y, color=pal["line1"], linewidth=2.2, label="Curve: (x, df/dx, f(x))")
            ax.plot(
                x,
                y_axis_vals,
                np.zeros_like(x),
                "--",
                color=pal["line2"],
                alpha=0.5,
                linewidth=1.2,
                label="z = 0 projection",
            )
            if tangent_at is not None and y0 is not None and m0 is not None:
                ax.scatter(
                    [tangent_at],
                    [m0],
                    [y0],
                    color=pal["accent"],
                    s=45,
                    zorder=6,
                    label=f"Point @ x={tangent_at} (m={m0:.3f})",
                )
            ax.set_xlabel("x")
            ax.set_ylabel("Slope: df/dx" if show_slope else "y (0)")
            ax.set_zlabel("f(x)")
            ax.set_title(f"[3D Normal] {label_expr}  [{active_color}] — Drag to rotate")
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
