import numpy as np
import vegas
import scipy.integrate as integrate

def overlap_nquad(E1, E2, geometry, *, epsabs=1e-8, epsrel=1e-6, limit=80, complex_value=True):

    def vdot_times_jac(*vars_):
        Y = geometry.make_Y(*vars_)
        return geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))

    opts = [{'limit': limit, 'epsabs': epsabs, 'epsrel': epsrel}] * len(geometry.bounds)

    def integrand_real(*vars_):
        return vdot_times_jac(*vars_).real
    
    if not complex_value:
        real_val, _ = integrate.nquad(integrand_real, geometry.bounds, opts=opts)
        return float(real_val)

    def integrand_imag(*vars_):
        return vdot_times_jac(*vars_).imag

    real_val, _ = integrate.nquad(integrand_real, geometry.bounds, opts=opts)
    imag_val, _ = integrate.nquad(integrand_imag, geometry.bounds, opts=opts)
    return real_val + 1j * imag_val

def overlap_vegas(E1, E2, geometry, *, neval=10_000, nitn_warmup=5, nitn_main=10, complex_value=True):

    integ = vegas.Integrator(geometry.bounds)

    def integrand_real(x):
        Y = geometry.make_Y(*x)
        val = geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))
        return val.real

    if not complex_value:
        integ(integrand_real, nitn=nitn_warmup, neval=max(1, neval // 3))
        res_re = integ(integrand_real, nitn=nitn_main, neval=neval)
        return float(res_re.mean)

    def integrand_imag(x):
        Y = geometry.make_Y(*x)
        val = geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))
        return val.imag

    integ(integrand_real, nitn=nitn_warmup, neval=max(1, neval // 3))
    integ(integrand_imag, nitn=nitn_warmup, neval=max(1, neval // 3))

    res_re = integ(integrand_real, nitn=nitn_main, neval=neval)
    res_im = integ(integrand_imag, nitn=nitn_main, neval=neval)
    return res_re.mean + 1j * res_im.mean

_METHODS = {
    "nquad": overlap_nquad,
    "vegas": overlap_vegas,
}

def overlap_integral(E1, E2, geometry, *, method="nquad", complex_value=True, **opts):
    if method not in _METHODS:
        raise ValueError(f"Unknown method '{method}'. Available: {list(_METHODS.keys())}")
    return _METHODS[method](E1, E2, geometry, complex_value=complex_value, **opts)


#as an optional bonus
#NOT TESTED PROPERLY!!!
#if you use many integrals with the same geometry, you might want to cache the VEGAS grid. Here's a simple wrapper class for that:
class OverlapIntegrator:
    def __init__(self, geometry, *, method="nquad", complex_value=True, **default_opts):
        self.geometry = geometry
        self.method = method
        self.complex_value = complex_value
        self.default_opts = dict(default_opts)

        self._vegas_integ = None
        self._vegas_warmed = False  # whether we've done warmup already

        if self.method == "vegas":
            self._vegas_integ = vegas.Integrator(self.geometry.bounds)

    # -------- core integrand helper --------
    def _vdot_times_jac(self, E1, E2, *vars_):
        Y = self.geometry.make_Y(*vars_)
        return self.geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))

    # -------- nquad implementation --------
    def _call_nquad(self, E1, E2, *, complex_value, epsabs=1e-8, epsrel=1e-6, limit=80):
        opts = [{'limit': limit, 'epsabs': epsabs, 'epsrel': epsrel}] * len(self.geometry.bounds)

        def integrand_real(*vars_):
            return self._vdot_times_jac(E1, E2, *vars_).real

        if not complex_value:
            real_val, _ = integrate.nquad(integrand_real, self.geometry.bounds, opts=opts)
            return float(real_val)

        def integrand_imag(*vars_):
            return self._vdot_times_jac(E1, E2, *vars_).imag

        real_val, _ = integrate.nquad(integrand_real, self.geometry.bounds, opts=opts)
        imag_val, _ = integrate.nquad(integrand_imag, self.geometry.bounds, opts=opts)
        return real_val + 1j * imag_val

    # -------- vegas implementation (cached integrator) --------
    def _call_vegas(
        self, E1, E2, *,
        complex_value,
        neval=10_000,
        nitn_warmup=5,
        nitn_main=10,
        warmup_each_call=False,
    ):
        integ = self._vegas_integ
        if integ is None:
            raise RuntimeError("VEGAS integrator not initialized.")

        def integrand_real(x):
            Y = self.geometry.make_Y(*x)
            val = self.geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))
            return val.real

        def integrand_imag(x):
            Y = self.geometry.make_Y(*x)
            val = self.geometry.jacobian(Y) * np.vdot(E1(Y), E2(Y))
            return val.imag

        # Warm-up strategy:
        # - If you do MANY integrals with similar structure, warming once is usually best.
        # - If the integrand changes wildly every call, set warmup_each_call=True.
        do_warmup = warmup_each_call or (not self._vegas_warmed)

        if do_warmup:
            integ(integrand_real, nitn=nitn_warmup, neval=max(1, neval // 3))
            if complex_value:
                integ(integrand_imag, nitn=nitn_warmup, neval=max(1, neval // 3))
            self._vegas_warmed = True

        if not complex_value:
            res_re = integ(integrand_real, nitn=nitn_main, neval=neval)
            return float(res_re.mean)

        res_re = integ(integrand_real, nitn=nitn_main, neval=neval)
        res_im = integ(integrand_imag, nitn=nitn_main, neval=neval)
        return res_re.mean + 1j * res_im.mean

    # -------- public call --------
    def __call__(self, E1, E2, *, complex_value=None, **opts):
        if complex_value is None:
            complex_value = self.complex_value

        # merge opts: call-time overrides defaults
        cfg = dict(self.default_opts)
        cfg.update(opts)

        if self.method == "nquad":
            return self._call_nquad(E1, E2, complex_value=complex_value, **cfg)
        elif self.method == "vegas":
            return self._call_vegas(E1, E2, complex_value=complex_value, **cfg)
        else:
            raise ValueError(f"Unknown method '{self.method}'. Use 'nquad' or 'vegas'.")

    def reset_vegas_grid(self):
        """If you want to discard the adapted grid and re-warm from scratch."""
        if self.method != "vegas":
            return
        self._vegas_integ = vegas.Integrator(self.geometry.bounds)
        self._vegas_warmed = False