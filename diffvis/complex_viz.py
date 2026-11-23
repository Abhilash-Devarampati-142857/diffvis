import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
from .parser import clean_math_expr
from .ui_controls import attach_interactive_ui, get_palette

COMPLEX_GLOBALS = {
    "__builtins__": {},
    "np": np,
    "sin": np.sin,
    "cos": np.cos,
    "tan": np.tan,
    "exp": np.exp,
    "log": np.log,
    "sqrt": np.sqrt,
    "abs": np.abs,
    "pi": np.pi,
    "e": np.e,
    "j": 1j,
    "i": 1j,
}


def complex_plot(
    expr_or_func="exp(i*pi*z) + 1",
    x_range=(-2.0, 2.0),
    y_range=(-2.0, 2.0),
    dim: str = "3d",
    color_mode: str = "Classic",
    grid_size: int = 65,
    save: str | None = None,
    show: bool = True,
):
    xs = np.linspace(x_range[0], x_range[1], grid_size)
    ys = np.linspace(y_range[0], y_range[1], grid_size)
    X, Y = np.meshgrid(xs, ys)
    Z = X + 1j * Y

    if callable(expr_or_func):
        W = expr_or_func(Z)
    else:
        expr_str = clean_math_expr(str(expr_or_func))
        code = compile(expr_str, "<diffvis_complex>", "eval")
        W = eval(code, COMPLEX_GLOBALS, {"z": Z})

    W = np.asarray(W, dtype=complex)
    mag = np.abs(W)
    phase = np.angle(W)
    norm_phase = (phase + np.pi) / (2.0 * np.pi)
    label_expr = expr_or_func if isinstance(expr_or_func, str) else "f(z)"

    theta = np.linspace(0, 2.0 * np.pi, 200)
    unit_x = np.cos(theta)
    unit_y = np.sin(theta)

    def redraw(fig, target_dim: str, active_color: str):
        pal = get_palette(active_color)
        for ax in list(fig.axes):
            if ax not in getattr(fig, "_diffvis_ctrl_axes", []):
                ax.remove()

        if target_dim == "2d":
            ax = fig.add_subplot(111)
            cf = ax.contourf(X, Y, mag, levels=30, cmap=pal["cmap"], alpha=0.88)
            ax.contour(X, Y, phase, levels=12, colors="white", linewidths=0.7, alpha=0.6)
            ax.plot(unit_x, unit_y, "--", color="cyan", linewidth=1.4, label="Unit circle |z|=1")
            ax.scatter([1.0], [0.0], color=pal["accent"], s=55, zorder=6, label="z = 1 -> e^(i*pi) + 1 = 0")
            ax.set_xlabel("Re(z)")
            ax.set_ylabel("Im(z)")
            ax.set_title(f"[2D Complex] {label_expr}  [{active_color}]")
            ax.legend(loc="upper right", fontsize=8)
            fig.colorbar(cf, ax=ax, shrink=0.85, pad=0.06, label="|f(z)|")
        else:
            ax = fig.add_subplot(111, projection="3d")
            surf_colors = cm.hsv(norm_phase) if active_color == "Classic" else plt.get_cmap(pal["cmap"])(norm_phase)
            ax.plot_surface(
                X, Y, mag, facecolors=surf_colors, rstride=1, cstride=1, antialiased=False
            )
            ax.plot(unit_x, unit_y, np.zeros_like(theta), "--", color="black", linewidth=1.2, label="Unit circle (z=0 plane)")
            ax.scatter([1.0], [0.0], [0.0], color=pal["accent"], s=60, zorder=8, label="z = 1: |f(1)| = 0")
            ax.set_xlabel("Re(z)")
            ax.set_ylabel("Im(z)")
            ax.set_zlabel("|f(z)|")
            ax.set_title(f"[3D Complex] {label_expr}  [{active_color}] — Drag to rotate")
            ax.legend(loc="upper left", fontsize=8)

        fig.subplots_adjust(left=0.24, right=0.95, top=0.90, bottom=0.12)
        if show and hasattr(fig, "_diffvis_build_controls"):
            fig._diffvis_build_controls()

    fig = plt.figure(figsize=(9.0, 5.5))
    start_dim = "2d" if str(dim).lower().strip() == "2d" else "3d"
    if show:
        attach_interactive_ui(fig, start_dim, color_mode, redraw)
    redraw(fig, start_dim, color_mode)

    if save:
        fig.savefig(save, dpi=150)
    if show:
        plt.show()
    plt.close(fig)
    return {"X": X, "Y": Y, "Z": Z, "W": W, "magnitude": mag, "phase": phase, "dim": start_dim}
