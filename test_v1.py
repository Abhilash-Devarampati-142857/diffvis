import diffvis

diffvis.normal("sin(x) * exp(-0.2*x)", x_range=(-5, 5), show_slope=True, tangent_at=1.0, save="v1_normal.png", show=False)
diffvis.integral("x**2", x_range=(0, 3), antiderivative=True, save="v1_integral.png", show=False)
diffvis.ode("dy/dx + y = x", x_range=(0, 5), y0=1.0, save="v1_ode.png", show=False)
diffvis.pde(">u/>t = 0.25 * (>>u/>>x)", x_range=(0, 1), t_range=(0, 0.5), save="v1_pde.png", show=False)
diffvis.complex("z**2 + 1", x_range=(-2, 2), y_range=(-2, 2), save="v1_complex.png", show=False)

print("v1 smoke test passed! PNGs saved in current folder.")
