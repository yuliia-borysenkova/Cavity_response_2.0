from scipy.integrate import solve_ivp

def driven_mode_rhs(t, y, omega, Q, RHS_fn):
    u, uD = y
    rhs_val = RHS_fn(t)
    return [
        uD,
        -omega / Q * uD - omega**2 * u - omega**2 * rhs_val
    ]

def solve_mode_amplitude(ts, RHS_fn, omega, Q, y0=(0.0, 0.0), method="Radau", rtol=1e-8, atol=1e-15):

    sol = solve_ivp(
        lambda t, y: driven_mode_rhs(t, y, omega, Q, RHS_fn),
        (ts[0], ts[-1]),
        y0=y0, t_eval=ts,
        method=method,
        rtol=rtol, atol=atol,
    )

    if not sol.success:
        print(f"[WARN] ODE solver failed: {sol.message}")

    u, uD = sol.y
    
    return {
        "t": ts,
        "u": u,
        "uD": uD,
        "c": u / omega**2,  # Convert back to physical amplitude c(t)
        "cD": uD / omega**2,
    }
