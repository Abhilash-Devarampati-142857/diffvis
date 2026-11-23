import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons

COLOR_PAIRS = {
    "HD Neon Cyan": {
        "bg": "#070b14",
        "panel": "#101828",
        "text": "#f0f9ff",
        "muted": "#64748b",
        "fg": "#00f5d4",
        "line2": "#00bbf9",
        "accent": "#ff2a85",
        "cmap": "cool",
    },
    "HD Royal Ink": {
        "bg": "#fcfbf7",
        "panel": "#f3f0e6",
        "text": "#0f172a",
        "muted": "#475569",
        "fg": "#1d4ed8",
        "line2": "#059669",
        "accent": "#e11d48",
        "cmap": "viridis",
    },
    "HD Solar Gold": {
        "bg": "#0e0c07",
        "panel": "#1b170e",
        "text": "#fffbeb",
        "muted": "#78716c",
        "fg": "#ffd000",
        "line2": "#ff7b00",
        "accent": "#38bdf8",
        "cmap": "inferno",
    },
    "HD Glacier Teal": {
        "bg": "#e8f4f8",
        "panel": "#d6ebf2",
        "text": "#082f49",
        "muted": "#334155",
        "fg": "#006d77",
        "line2": "#e29578",
        "accent": "#c1121f",
        "cmap": "winter",
    },
    "HD Hyper Magenta": {
        "bg": "#12071f",
        "panel": "#1f0d36",
        "text": "#fdf4ff",
        "muted": "#7e22ce",
        "fg": "#ff4ecd",
        "line2": "#9b5de5",
        "accent": "#00f5d4",
        "cmap": "magma",
    },
}


def get_palette(pair_name: str) -> dict:
    return COLOR_PAIRS.get(pair_name, COLOR_PAIRS["HD Neon Cyan"])


def apply_hd_figure(fig, pal: dict, dpi: int = 120):
    """Set crisp figure background and HD display DPI."""
    fig.set_dpi(dpi)
    fig.patch.set_facecolor(pal["bg"])


def style_axes(ax, pal: dict):
    """Apply paired HD background and sharp axis/grid styling to 2D or 3D axes."""
    ax.set_facecolor(pal["bg"])
    if hasattr(ax, "xaxis") and hasattr(ax.xaxis, "pane"):
        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.pane.set_facecolor(pal["panel"])
            axis.pane.set_edgecolor(pal["muted"])
            axis.pane.set_alpha(0.45)
            axis._axinfo["grid"]["color"] = pal["muted"]
            axis._axinfo["grid"]["alpha"] = 0.30
        ax.tick_params(colors=pal["text"], labelsize=8.5)
        ax.xaxis.label.set_color(pal["text"])
        ax.yaxis.label.set_color(pal["text"])
        ax.zaxis.label.set_color(pal["text"])
    else:
        ax.tick_params(colors=pal["text"], labelsize=9)
        ax.xaxis.label.set_color(pal["text"])
        ax.yaxis.label.set_color(pal["text"])
        for spine in ax.spines.values():
            spine.set_color(pal["muted"])
            spine.set_linewidth(1.1)
        ax.grid(True, linestyle=":", alpha=0.28, color=pal["muted"])
    ax.title.set_color(pal["text"])
    ax.title.set_fontweight("bold")


def style_legend(ax, pal: dict, loc: str = "best", fontsize: int = 8.5):
    leg = ax.legend(loc=loc, fontsize=fontsize, frameon=True)
    if leg is not None:
        frame = leg.get_frame()
        frame.set_facecolor(pal["panel"])
        frame.set_edgecolor(pal["muted"])
        frame.set_alpha(0.92)
        for text in leg.get_texts():
            text.set_color(pal["text"])
    return leg


def _dim_button_label(dim: str) -> str:
    return "3D -> 2D" if dim == "3d" else "2D -> 3D"


def attach_interactive_ui(fig, initial_dim: str, initial_pair: str, redraw_fn):
    """
    Attach a '2D <-> 3D' button and 5 HD paired Background+Graph color themes
    with always-visible labels synced across redraws.
    """
    default_pair = "HD Neon Cyan"
    state = {
        "dim": "3d" if str(initial_dim).lower().strip() == "3d" else "2d",
        "color": initial_pair if initial_pair in COLOR_PAIRS else default_pair,
    }

    toolbar_btn = None
    canvas_dim_btn = None
    radio_widget = None

    def _sync_labels():
        txt = _dim_button_label(state["dim"])
        if canvas_dim_btn is not None:
            canvas_dim_btn.label.set_text(txt)
            try:
                canvas_dim_btn.ax.figure.canvas.draw_idle()
            except Exception:
                pass
        if toolbar_btn is not None:
            try:
                toolbar_btn.config(text=txt)
                toolbar_btn.update_idletasks()
            except Exception:
                pass

    def _apply_redraw():
        pal = get_palette(state["color"])
        apply_hd_figure(fig, pal)
        redraw_fn(fig, state["dim"], state["color"])
        _sync_labels()
        fig.canvas.draw()

    def _on_toggle_dim(_event=None):
        state["dim"] = "2d" if state["dim"] == "3d" else "3d"
        _apply_redraw()

    def _on_color_change(label):
        if label and label in COLOR_PAIRS and label != state["color"]:
            state["color"] = label
            _apply_redraw()

    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None) if manager is not None else None
    tk_parent = getattr(toolbar, "window", None) or getattr(toolbar, "_tkcanvas", None)
    master = getattr(tk_parent, "master", None) if tk_parent is not None else None

    if master is not None:
        try:
            import tkinter as tk

            btn = tk.Button(
                master,
                text=_dim_button_label(state["dim"]),
                command=_on_toggle_dim,
                padx=6,
                pady=1,
            )
            btn.pack(side=tk.LEFT, padx=4)
            toolbar_btn = btn
        except Exception:
            toolbar_btn = None

    def _build_controls():
        nonlocal canvas_dim_btn, radio_widget
        pal = get_palette(state["color"])
        apply_hd_figure(fig, pal)

        for old_ax in getattr(fig, "_diffvis_ctrl_axes", []):
            if old_ax in fig.axes:
                old_ax.remove()

        dim_ax = fig.add_axes([0.02, 0.80, 0.18, 0.08])
        canvas_dim_btn = Button(
            dim_ax,
            _dim_button_label(state["dim"]),
            color=pal["panel"],
            hovercolor=pal["bg"],
        )
        canvas_dim_btn.label.set_fontsize(8.5)
        canvas_dim_btn.label.set_fontweight("bold")
        canvas_dim_btn.label.set_color(pal["fg"])
        canvas_dim_btn.on_clicked(_on_toggle_dim)

        radio_ax = fig.add_axes([0.02, 0.28, 0.18, 0.45])
        radio_ax.set_facecolor(pal["panel"])
        radio_ax.set_title(
            "HD Color Pairs",
            fontsize=8.5,
            fontweight="bold",
            pad=5,
            color=pal["text"],
        )
        pair_names = list(COLOR_PAIRS.keys())
        radio_widget = RadioButtons(
            radio_ax,
            pair_names,
            active=pair_names.index(state["color"]),
            activecolor=pal["fg"],
        )
        for lbl in radio_widget.labels:
            lbl.set_fontsize(8)
            lbl.set_color(pal["text"])
        radio_widget.on_clicked(_on_color_change)

        fig._diffvis_ctrl_axes = [dim_ax, radio_ax]
        fig._diffvis_widgets = [canvas_dim_btn, radio_widget]
        _sync_labels()

    fig._diffvis_build_controls = _build_controls
    fig._diffvis_state = state
    return state
