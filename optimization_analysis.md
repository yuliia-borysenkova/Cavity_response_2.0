# Optimization Analysis: Cavity Response 2.0

## Summary

The codebase computes cavity mode amplitudes driven by gravitational waves. The main bottlenecks and improvement opportunities span three areas: **correctness bugs**, **numerical/computational performance**, and **code quality/DRY violations**.

---

## 🐛 Bugs / Correctness Issues

### 1. `ode/utils.py` — `compute_full_fourier` uses `freqs` before assignment

**File**: [utils.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/ode/utils.py#L179)

```python
# Line 179 — uses `freqs` before it is defined on line 187!
c_hat_num = np.fft.fft(c_padded) * dt * np.exp(-1j*freqs*ts_ext[0])

# ...
freqs = omegas / (2 * np.pi)  # defined here, AFTER use
```

**Fix**: Move `freqs` (and `omegas`) computation to before line 179:
```python
omegas = np.fft.fftfreq(n, d=dt) * 2 * np.pi
freqs  = omegas / (2 * np.pi)
```

---

### 2. `rhs_num.py` — `find_chirp_match_time` called with `f_cavity=omega` (angular frequency)

**File**: [rhs_num.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs_num.py#L146)

```python
# Line 146/166 — passes omega (rad/s), but the function likely expects Hz
t_match, _ = find_chirp_match_time(ts=ts, f_cavity=omega, ...)
# Compare with rhs.py line 86 which correctly uses f_cavity (Hz)
```

**Fix**: Pass `omega / (2*np.pi)` (i.e., `f_cavity_GHz * 1e9`) consistently.

---

### 3. `rhs.py` line 96 — redundant `== True` comparison

```python
if args.pre_RHS == True:   # ← anti-pattern
```
Should be `if args.pre_RHS:`.

---

## ⚡ Performance: The Biggest Win — Vectorise `compute_num_rhs`

**File**: [num_rhs_integration.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs/num_rhs_integration.py#L7-L31)

The two Python loops `[overlap_sum1d(t) for t in ts]` and `[err_sum1d(t) for t in ts]` call `hplus(t - xps/c)` and `hcross(t - xps/c)` for every time step individually. This is **O(Nt × Ns)** in pure Python.

### Vectorised replacement

```python
def compute_num_rhs(xpar, xps, ts, num, V, Efield, B_plusdir, B_crossdir, hplus_arr, hcross_arr, t_data, ell):
    """
    Fully vectorised version.
    hplus_arr, hcross_arr are raw arrays (not lambda interp wrappers).
    t_data is the original time axis they are defined on.
    """
    xp_num = len(xps)

    # --- E-field slice averages (unchanged, already vectorised) ---
    EE = V * np.mean(
        stats.norm.pdf(xpar[:, None], loc=xps[None, :], scale=ell)[:, :, None]
        * np.conj(Efield[:, None, :]),
        axis=0,
    )  # shape (Ns, 3)

    E_plus  = EE @ np.conj(B_plusdir)   # shape (Ns,)
    E_cross = EE @ np.conj(B_crossdir)  # shape (Ns,)

    # --- Vectorised retarded-time interpolation ---
    # t_ret[i_t, i_x] = ts[i_t] - xps[i_x] / c
    t_ret = ts[:, None] - xps[None, :] / c_cnst   # shape (Nt, Ns)

    hp  = np.interp(t_ret.ravel(), t_data, hplus_arr,  left=0.0, right=0.0).reshape(t_ret.shape)
    hc  = np.interp(t_ret.ravel(), t_data, hcross_arr, left=0.0, right=0.0).reshape(t_ret.shape)

    # --- Overlap integral ---
    # shape (Nt,)
    overlap = (xps[-1] - xps[0]) / xp_num * np.sum(
        hp * E_plus[None, :] + hc * E_cross[None, :], axis=1
    )

    # --- Error (vectorised) ---
    EEsDD   = (EE[2:] - 2*EE[1:-1] + EE[:-2]) / (xps[1] - xps[0])**2
    EEsqr   = V * np.mean(
        stats.norm.pdf(xpar[:, None], loc=xps[None, :], scale=ell)[:, :, None]**2
        * np.abs(Efield[:, None, :])**2,
        axis=0,
    )
    EE_sys  = np.abs(EEsDD) * ell**2 / 2
    EE_stat = np.sqrt((V * EEsqr - np.abs(EE)**2) / num)
    EE_errs = np.sqrt(EE_sys**2 + EE_stat[1:-1]**2)

    hp_mid = np.interp((ts[:, None] - xps[1:-1][None, :] / c_cnst).ravel(),
                       t_data, hplus_arr, left=0.0, right=0.0).reshape(len(ts), -1)
    hc_mid = np.interp((ts[:, None] - xps[1:-1][None, :] / c_cnst).ravel(),
                       t_data, hcross_arr, left=0.0, right=0.0).reshape(len(ts), -1)

    integrand_err = (hp_mid[:, :, None] * B_plusdir[None, None, :]
                   + hc_mid[:, :, None] * B_crossdir[None, None, :])  # (Nt, Ns-2, 3)
    errs = (xps[-1] - xps[0]) / xp_num * np.sqrt(
        np.einsum('txi,xi->t', integrand_err**2, np.abs(EE_errs)**2)
    )

    return overlap.real, errs
```

**Expected speedup**: 10–100× for typical `Nt=10000, Ns=100`.

---

## ⚡ Performance: Precompute `E_plus` / `E_cross` outside the `ell` search loop

**File**: [rhs_num.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs_num.py#L86-L87)

The Gaussian kernel depends on `ell`, but the `EE` computation also involves the inner Einsum that projects onto `B_plus/cross` directions. When sweeping `ell_arr`, the retarded-time interpolation grid `t_ret` is **the same for every `ell`** — only the spatial weighting changes.

Extract the retarded waveform evaluation outside the loop:
```python
t_ret = ts[:, None] - xps[None, :] / c_cnst   # compute once
hp_grid = np.interp(t_ret.ravel(), t_data, hplus_dd, ...).reshape(t_ret.shape)
hc_grid = np.interp(t_ret.ravel(), t_data, hcross_dd, ...).reshape(t_ret.shape)

for i, ell in enumerate(tqdm(ell_arr)):
    overlaps[i], errors[i] = compute_num_rhs(..., hp_grid=hp_grid, hc_grid=hc_grid)
```

---

## ⚡ Performance: Use `scipy.interpolate.CubicSpline` instead of `interp1d`

**File**: [rhs/utils.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs/utils.py#L102-L111) and [ode/utils.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/ode/utils.py#L96)

`scipy.interpolate.interp1d` is deprecated and slower than `CubicSpline` or `make_interp_spline`. The `CubicSplineInterp` wrapper class in `rhs/utils.py` already uses `interp1d(kind="cubic")` — switch to `CubicSpline` directly:

```python
from scipy.interpolate import CubicSpline

class CubicSplineInterp:
    def __init__(self, ts, y, left_val=0.0, right_val=0.0):
        self._spline = CubicSpline(ts, y, extrapolate=False)
        self._left   = left_val
        self._right  = right_val

    def __call__(self, tau):
        result = self._spline(tau)
        result = np.where(tau < self._spline.x[0],  self._left,  result)
        result = np.where(tau > self._spline.x[-1], self._right, result)
        return result
```

Also replace all standalone `interp1d(..., kind="linear")` calls with `np.interp` (already used in some places) — it's faster and doesn't generate a deprecation warning.

---

## ⚡ Performance: Avoid recomputing `f_cavity` and `t_match` in the plot loop

**File**: [ode.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/ode.py#L175-L192)

`find_chirp_match_time` is called **once per plot** inside the loop:
```python
for y, label, ylabel, title, filename in plots:
    ...
    t_match, _ = find_chirp_match_time(...)   # called 3 times
```

Move this outside the loop:
```python
f_cavity = omega / (2 * np.pi)
t_match, _ = find_chirp_match_time(ts=ts_ext, f_cavity=f_cavity,
                                    data_dir=args.data_dir, data_file_name=args.data)

for y, label, ylabel, title, filename in plots:
    ...
    if t_match is not None and args.freq_match:
        ax.axvline(...)
```

Similarly in [rhs.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs.py#L96-L111) — `f_cavity` and `t_match` are computed twice (lines 85–86 and 102–103). Factor them out.

---

## ⚡ Performance: `extend_rhs` — use `dt` from the correct end of the array

**File**: [ode/utils.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/ode/utils.py#L110)

```python
dt = abs(time[-2] - time[-1])   # ← last interval only
```

Better to use the median spacing for robustness (unevenly-sampled inputs):
```python
dt = np.median(np.diff(time))
```

---

## 🔧 Code Quality / DRY

### 1. Duplicated plot block for RHS / pre_RHS

Both [rhs.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs.py#L80-L111) and [rhs_num.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs_num.py#L140-L177) repeat identical code to plot RHS and pre_RHS. Extract a shared helper:

```python
def plot_rhs(ts, rhs_arr, label, ylabel, title, save_path,
             t_match=None, freq_match=False):
    fig, ax = new_figure()
    ax.plot(ts * 1e9, rhs_arr, label=label)
    if t_match is not None and freq_match:
        ax.axvline(t_match * 1e9, linestyle="--", linewidth=1.5,
                   color="darkred", label=r"$f_{\rm GW} = f_{\rm cav}$")
    ax.set_xlabel(r"$t\,[\mathrm{ns}]$")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend()
    save_figure(fig, save_path)
```

### 2. `mode_name` construction duplicated across `rhs.py` and `ode.py`

Extract to a shared utility (e.g., in `misc/` or a new `utils.py`):
```python
def build_mode_name(mode_fam, mode_par):
    return mode_fam + mode_par if mode_par is not None else mode_fam
```

### 3. Debug `print("here")` left in production code

**File**: [rhs/utils.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs/utils.py#L117) line 117 — remove it.

### 4. `compute_num_rhs` signature inconsistency

The function takes raw arrays `hplus`, `hcross` as callables in [num_rhs_integration.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs/num_rhs_integration.py#L7), but uses lambdas wrapping `np.interp` in the caller ([rhs_num.py](file:///Users/yborysenkova/Desktop/Projects/Cavity_response_2.0/rhs_num.py#L75-L76)). The vectorisation above removes this abstraction entirely — just pass the raw arrays and `t_data`.

---

## 📋 Priority Summary

| Priority | File | Issue | Effort |
|---|---|---|---|
| 🔴 Bug | `ode/utils.py` | `freqs` used before definition | Trivial |
| 🔴 Bug | `rhs_num.py` | `f_cavity=omega` should be `omega/(2π)` | Trivial |
| 🟠 Perf | `rhs/num_rhs_integration.py` | Vectorise time loop (10–100× speedup) | Medium |
| 🟠 Perf | `rhs_num.py` | Cache retarded waveform outside `ell` loop | Small |
| 🟡 Perf | `ode.py` | Move `find_chirp_match_time` out of plot loop | Trivial |
| 🟡 Perf | `ode/utils.py` & `rhs/utils.py` | Replace `interp1d` with `CubicSpline`/`np.interp` | Small |
| 🟢 Quality | `rhs.py`, `rhs_num.py` | Extract shared `plot_rhs` helper, fix `== True` | Small |
| 🟢 Quality | `rhs/utils.py` | Remove debug `print("here")` | Trivial |
