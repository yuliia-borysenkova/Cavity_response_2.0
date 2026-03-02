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