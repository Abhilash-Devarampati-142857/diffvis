import matplotlib.pyplot as plt
import numpy as np
from scipy.integrate import solve_ivp
from .parser import clean_math_expr, make_evaluator
from .ui_controls import attach_interactive_ui, get_palette


def _parse_ode_rhs(ode_str: str):
    s = clean_math_expr(ode_str).replace(" ", "")
    if "=" not in s:
        return make_evaluator(s, ["x", "y"])

    lhs, rhs = s.split("=", 1)
    if lhs == "dy/dx":
        return make_evaluator(rhs, ["x", "y"])
    if lhs.startswith("dy/dx+"):
        extra = lhs[len("dy/dx+") :]
        return make_evaluator(f"({rhs}) - ({extra})", ["x", "y"])
    if lhs.startswith("dy/dx-"):
        extra = lhs[len("dy/dx-") :]
        return make_evaluator(f"({rhs}) + ({extra})", ["x", "y"])
    raise ValueError(
        f"Unsupported ODE format: '{ode_str}'. Try 'dy/dx + y = x' or 'dy/dx = -2*y'."
    )


def ode(
    ode_expr_or_func,
    x_range=(0.0, 5.0),
    y0: float = 1.0,
    y0_list: list[float] | None = None,
    dim: str = "2d",
    color_mode: str = "Classic",
    points: int = 250,
    save: str | None = None,
    show: bool = True,
):
    rhs_fn = (
        _parse_ode_rhs(ode_expr_or_func)
        if isinstance(ode_expr_or_func, str)
        else ode_expr_or_func
    )
    x_eval = np.linspace(x_range[0], x_range[1], points)
    initials = y0_list if y0_list is not None else [float(y0)]
    label_expr = ode_expr_or_func if isinstance(ode_expr_or_func, str) else "dy/dx"

    solutions = []
    for init_y in initials:
        sol = solve_ivp(
            fun=lambda x, y: [float(rhs_fn(x, y[0]))],
            t_span=(x_range[0], x_range[1]),
            y0=[float(init_y)],
            t_eval=x_eval,
        )
        slope_vals = np.array(
            [float(rhs_fn(xv, yv)) for xv, yv in zip(sol.t, sol.y[0])],
            dtype=float,
        )
        solutions.append((init_y, sol.t, sol.y[0], slope_vals))

    def redraw(fig, target_dim: str, active_color: str):
        pal = get_palette(active_color)
        cmap = plt.get_cmap(pal["cmap"])
        for ax in list(fig.axes):
            if ax not in getattr(fig, "_diffvis_ctrl_axes", []):
                ax.remove()

        n_sol = max(len(solutions), 1)
        if target_dim == "2d":
            ax = fig.add_subplot(111)
            for idx, (init_y, xs, ys, _) in enumerate(solutions):
                c = pal["line1"] if n_sol == 1 else cmap(0.2 + 0.7 * (idx / max(n_sol - 1, 1)))
                ax.plot(xs, ys, color=c, linewidth=2.1, label=f"y({x_range[0]})={init_y}")
            ax.set_xlabel("x")
            ax.set_ylabel("y(x)")
            ax.set_title(f"[2D ODE] {label_expr}  [{active_color}]")
            ax.legend(fontsize=8.5)
        else:
            ax = fig.add_subplot(111, projection="3d")
            for idx, (init_y, xs, ys, slopes) in enumerate(solutions):
                c = pal["line1"] if n_sol == 1 else cmap(0.2 + 0.7 * (idx / max(n_sol - 1, 1)))
                ax.plot(xs, slopes, ys, color=c, linewidth=2.2, label=f"y0={init_y}: (x, dy/dx, y)")
                ax.plot(
                    xs, slopes, np.full_like(xs, init_y), "--", color=c, alpha=0.35, linewidth=1.0
                )
            ax.set_xlabel("x")
            ax.set_ylabel("Slope: dy/dx")
            ax.set_zlabel("y(x)")
            ax.set_title(f"[3D ODE] {label_expr}  [{active_color}] — Drag to rotate")
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

    _, main_x, main_y, main_slope = solutions[0]
    return {"x": main_x, "y": main_y, "dy_dx": main_slope, "dim": start_dim}
