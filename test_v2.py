import diffvis

diffvis.normal(
    "sin(x) * exp(-0.2*x)", x_range=(-5, 5), show_slope=True, tangent_at=1.0
)
diffvis.integral("x**2", x_range=(0, 3), antiderivative=True)
diffvis.ode("dy/dx + y = x", x_range=(0, 5), y0=1.0)
diffvis.pde(">u/>t = 0.25 * (>>u/>>x)", x_range=(0, 1), t_range=(0, 0.5))
diffvis.complex("z**2 + 1", x_range=(-2, 2), y_range=(-2, 2))
