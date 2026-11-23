import re
import matplotlib.pyplot as plt
import numpy as np
from .parser import normalize_partials
from .ui_controls import attach_interactive_ui, get_palette


def _extract_alpha(pde_str: str, default_alpha: float) -> float:
    norm = normalize_partials(pde_str).replace(" ", "")
    match = re.search(r"([0-9]*\.?[0-9]+)\*d2u_dx2", norm)
    if match:
        return float(match.group(1))
    return default_alpha


def pde(
    pde_expr: str = ">u/>t = 0.25 * (>>u/>>x)",
    x_range=(0.0, 1.0),
    t_range=(0.0, 0.5),
    alpha: float = 0.25,
    dim: str = "3d",
    color_mode: str = "Classic",
    nx: int = 35,
    nt: int = 90,
    cmap: str | None = None,
    save: str | None = None,
    show: bool = True,
):
    a = _extract_alpha(pde_expr, alpha)
    x = np.linspace(x_range[0], x_range[1], nx)
    t = np.linspace(t_range[0], t_range[1], nt)
    dx = x[1] - x[0]
    dt = t[1] - t[0]
    r = a * dt / (dx**2)

    u = np.zeros((nt, nx), dtype=float)
    u[0, :] = np.sin(np.pi * (x - x_range[0]) / (x_range[1] - x_range[0]))

    for n in range(nt - 1):
        u[n + 1, 1:-1] = u[n, 1:-1] + r * (u[n, 2:] - 2.0 * u[n, 1:-1] + u[n, :-2])

    X, T = np.meshgrid(x, t)

    def redraw(fig, target_dim: str, active_color: str):
        pal = get_palette(active_color)
        active_cmap = cmap if cmap else pal["cmap"]
        line_cmap = plt.get_cmap(active_cmap)

        for ax in list(fig.axes):
            if ax not in getattr(fig, "_diffvis_ctrl_axes", []):
                ax.remove()

        if target_dim == "2d":
            ax = fig.add_subplot(111)
            slice_indices = [0, nt // 4, nt // 2, 3 * nt // 4, nt - 1]
            for pos, idx in enumerate(slice_indices):
                c = line_cmap(0.15 + 0.75 * (pos / max(len(slice_indices) - 1, 1)))
                ax.plot(x, u[idx], color=c, linewidth=1.9, label=f"t = {t[idx]:.3f}")
            ax.set_xlabel("Space x")
            ax.set_ylabel("u(x, t)")
            ax.set_title(f"[2D PDE Slices] {pde_expr}  [{active_color}]")
            ax.legend(fontsize=8.5)
        else:
            ax = fig.add_subplot(111, projection="3d")
            surf = ax.plot_surface(
                X, T, u, cmap=active_cmap, edgecolor="none", rstride=1, cstride=2
            )
            ax.set_xlabel("Space x")
            ax.set_ylabel("Time t")
            ax.set_zlabel("u(x, t)")
            ax.set_title(f"[3D PDE Surface] {pde_expr} (r={r:.3f})  [{active_color}]")
            fig.colorbar(surf, ax=ax, shrink=0.52, pad=0.08, label="u(x, t)")

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
    return {"x": x, "t": t, "u": u, "r": r, "alpha": a, "dim": start_dim}
