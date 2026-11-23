import matplotlib.pyplot as plt
from matplotlib.widgets import Button


def attach_2d3d_toolbar(fig, initial_dim: str, render_fn):
    """
    Attach a '2D / 3D' toggle button to the figure's top toolbar (with a figure-button fallback).

    render_fn(fig, target_dim) must clear/recreate axes on `fig` for '2d' or '3d'.
    """
    state = {"dim": "3d" if str(initial_dim).lower().strip() == "3d" else "2d"}

    def _label_for(dim: str) -> str:
        return "2D <-> 3D [3D]" if dim == "3d" else "2D <-> 3D [2D]"

    def _toggle(_event=None):
        state["dim"] = "2d" if state["dim"] == "3d" else "3d"
        render_fn(fig, state["dim"])
        if toolbar_btn is not None:
            toolbar_btn.config(text=_label_for(state["dim"]))
        if canvas_btn is not None:
            canvas_btn.label.set_text(_label_for(state["dim"]))
        fig.canvas.draw_idle()

    toolbar_btn = None
    canvas_btn = None

    manager = getattr(fig.canvas, "manager", None)
    toolbar = getattr(manager, "toolbar", None) if manager is not None else None

    # TkAgg toolbar uses a tk.Frame / tk widget hierarchy
    tk_parent = getattr(toolbar, "window", None) or getattr(toolbar, "_tkcanvas", None)
    master = getattr(tk_parent, "master", None) if tk_parent is not None else None

    added_to_toolbar = False
    if master is not None:
        try:
            import tkinter as tk

            btn = tk.Button(
                master,
                text=_label_for(state["dim"]),
                command=_toggle,
                padx=6,
                pady=1,
            )
            btn.pack(side=tk.LEFT, padx=4)
            toolbar_btn = btn
            added_to_toolbar = True
        except Exception:
            added_to_toolbar = False

    if not added_to_toolbar:
        # Fallback button inside the figure if not using a Tk toolbar
        btn_ax = fig.add_axes([0.78, 0.01, 0.20, 0.045])
        canvas_btn = Button(btn_ax, _label_for(state["dim"]))
        canvas_btn.on_clicked(_toggle)
        fig._diffvis_2d3d_btn = canvas_btn  # keep reference alive

    fig._diffvis_toggle_2d3d = _toggle
    return state
