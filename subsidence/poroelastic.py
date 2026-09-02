"""Poroelastic storage core: seasonal head <-> surface coupling -> elastic
skeletal storage properties.

I/O-free by design (all file access lives in the runner, 08_poroelastic_storage.py)
so the maths stays unit-testable.

Method (Jacob 1940; Galloway & Burbey 2011; Rezaei & Mousavi 2019; Razeghi et al. 2022):
  1. decompose each daily series into a quadratic trend (inelastic compaction +
     long-term head decline) plus an annual harmonic; the trend-removed residual
     carries the seasonal/elastic signal.
  2. confirm poroelastic coupling: trend-removed surface and head should be
     in-phase with high correlation.
  3. invert elastic skeletal storativity  S_ke = dB_elastic / dH  (dimensionless),
     i.e. the ratio of seasonal surface amplitude to seasonal head amplitude.
  4. convert to elastic skeletal specific storage  Ss = S_ke / b  [1/m], where b
     is the thickness of the deforming interval (layer-resolved for MLCW; a bulk
     column thickness for GNSS — caller's responsibility to pass the right b).
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np
import pandas as pd

RHO_W = 1000.0   # kg/m^3, water density
G = 9.80665      # m/s^2


@dataclass
class Decomposition:
    """Result of a quad-trend + single-annual-harmonic least-squares fit."""
    amplitude: float        # seasonal amplitude, same units as the input series
    phase: float            # radians, atan2(B, A) of the (sin, cos) harmonic
    seasonal_r2: float      # variance explained by the harmonic beyond the quad trend
    trend_rate: float       # linear trend coefficient (input units per year)
    n: int                  # number of finite samples used in the fit
    residual: pd.Series     # trend-removed series (seasonal + noise), NaN where input NaN
    t0: pd.Timestamp        # epoch the phase is referenced to (see phase_lag_days)


def _years_from_start(index: pd.DatetimeIndex, t0=None) -> np.ndarray:
    """Elapsed years from `t0` (default: the first sample). Two series whose phases
    will be differenced MUST share the same `t0`, or the phase difference carries a
    spurious w*(t0_a - t0_b) offset."""
    if not isinstance(index, pd.DatetimeIndex):
        raise TypeError("series must have a DatetimeIndex")
    origin = index[0] if t0 is None else pd.Timestamp(t0)
    return (index - origin).total_seconds().to_numpy() / (365.25 * 86400.0)


def common_origin(*series: pd.Series) -> pd.Timestamp:
    """Earliest timestamp across the given series: the shared phase epoch to pass to
    `decompose` when the resulting phases will be compared."""
    return min(pd.Timestamp(s.index[0]) for s in series)


def decompose(series: pd.Series, period_years: float = 1.0, t0=None) -> Decomposition:
    """Joint quad-trend + harmonic fit so the trend cannot leak into the seasonal
    term. Returns amplitude/phase of the harmonic and the trend-removed residual.

    `t0` sets the epoch the phase is referenced to. The amplitude and the residual
    are invariant to it; the phase is not. Pass a shared `t0` (see `common_origin`)
    whenever the phase will be compared against another series."""
    t = _years_from_start(series.index, t0)
    y = series.to_numpy(dtype=float)
    finite = np.isfinite(y)
    if finite.sum() < 5:
        raise ValueError("need >=5 finite samples to decompose")
    w = 2 * np.pi / period_years
    full = np.column_stack([np.ones_like(t), t, t**2, np.sin(w * t), np.cos(w * t)])
    trend = full[:, :3]
    cf, *_ = np.linalg.lstsq(full[finite], y[finite], rcond=None)
    ct, *_ = np.linalg.lstsq(trend[finite], y[finite], rcond=None)
    a_sin, b_cos = cf[3], cf[4]
    rss_full = float(np.sum((y[finite] - full[finite] @ cf) ** 2))
    rss_trend = float(np.sum((y[finite] - trend[finite] @ ct) ** 2))
    seasonal_r2 = 1.0 - rss_full / rss_trend if rss_trend > 0 else np.nan
    resid = pd.Series(y - trend @ ct, index=series.index)  # trend removed, NaN preserved
    resid[~finite] = np.nan
    return Decomposition(
        amplitude=float(np.hypot(a_sin, b_cos)),
        phase=float(np.arctan2(b_cos, a_sin)),
        seasonal_r2=float(seasonal_r2),
        trend_rate=float(cf[1]),
        n=int(finite.sum()),
        residual=resid,
        t0=pd.Timestamp(series.index[0] if t0 is None else t0),
    )


def _harmonic_amp(t: np.ndarray, y: np.ndarray, w: float):
    """OLS quad-trend + annual harmonic on finite rows. Returns (amplitude, X, coef,
    residual) over the finite subset (for residual-bootstrap reuse)."""
    m = np.isfinite(y)
    X = np.column_stack([np.ones_like(t), t, t**2, np.sin(w * t), np.cos(w * t)])[m]
    yf = y[m]
    c, *_ = np.linalg.lstsq(X, yf, rcond=None)
    return float(np.hypot(c[3], c[4])), X, c, yf - X @ c


def bootstrap_ci(disp: pd.Series, head: pd.Series, block_disp: int, block_head: int,
                 period_years: float = 1.0, n_boot: int = 1500, seed: int = 0) -> dict:
    """Moving-block bootstrap of harmonic-fit residuals, preserving residual
    autocorrelation. Returns point estimates and 2.5/97.5 percentile CIs for the
    displacement amplitude, head amplitude, and their ratio S_ke. Deterministic
    given `seed`.

    The block length is per-series and counted in SAMPLES, so it must be set from
    each series' own cadence: the head record is daily even where the paired
    displacement record is monthly, and reusing the displacement's block there
    would understate the head amplitude uncertainty several-fold."""
    w = 2 * np.pi / period_years
    rng = np.random.default_rng(seed)

    def prep(s):
        t = _years_from_start(s.index)
        return _harmonic_amp(t, s.to_numpy(dtype=float), w)

    ad, Xd, cd, rd = prep(disp)
    ah, Xh, ch, rh = prep(head)

    def draw(X, c, r, blk):
        n = len(r); fit = X @ c
        blk = min(blk, n)
        nb = int(np.ceil(n / blk))
        starts = rng.integers(0, n - blk + 1, size=nb)
        br = np.concatenate([r[s:s + blk] for s in starts])[:n]
        cc, *_ = np.linalg.lstsq(X, fit + br, rcond=None)
        return float(np.hypot(cc[3], cc[4]))

    da = np.array([draw(Xd, cd, rd, block_disp) for _ in range(n_boot)])
    ha = np.array([draw(Xh, ch, rh, block_head) for _ in range(n_boot)])
    ratio = da / ha
    pct = lambda a: (float(np.percentile(a, 2.5)), float(np.percentile(a, 97.5)))
    return {"disp_amp": ad, "disp_amp_ci": pct(da),
            "head_amp": ah, "head_amp_ci": pct(ha),
            "s_ke": ad / ah, "s_ke_ci": pct(ratio)}


def couple(resid_a: pd.Series, resid_b: pd.Series, resample: str | None = "D") -> tuple[float, int]:
    """Pearson correlation of two trend-removed series on their common dates.
    Returns (r, n_common). Optionally resample both to a common cadence first."""
    a, b = resid_a, resid_b
    if resample is not None:
        a = a.resample(resample).mean()
        b = b.resample(resample).mean()
    j = pd.concat([a.rename("a"), b.rename("b")], axis=1, sort=True).dropna()
    if len(j) < 3 or j["a"].std() == 0 or j["b"].std() == 0:
        return (np.nan, len(j))
    return (float(np.corrcoef(j["a"], j["b"])[0, 1]), len(j))


def phase_lag_days(disp: Decomposition, head: Decomposition,
                   period_years: float = 1.0) -> float:
    """Days by which the surface peak lags the head peak, wrapped to +/- half a
    period. Positive => surface peaks AFTER head (response lags the forcing);
    negative => surface leads.

    Both decompositions must be referenced to the same epoch. Differencing phases
    fitted from different origins adds a spurious w*(t0_disp - t0_head) offset,
    which for records starting years apart is up to half a period."""
    if disp.t0 != head.t0:
        raise ValueError(
            f"phases reference different epochs (disp t0={disp.t0.date()}, "
            f"head t0={head.t0.date()}); decompose both with a shared t0 "
            f"(see common_origin) before comparing phase"
        )
    period_days = period_years * 365.25
    lag = (head.phase - disp.phase) / (2 * np.pi) * period_days
    half = period_days / 2.0
    return float(((lag + half) % period_days) - half)


def storativity(disp_amplitude_m: float, head_amplitude_m: float) -> float:
    """Elastic skeletal storativity S_ke = dB_elastic / dH (dimensionless).
    Both amplitudes MUST be in metres."""
    if not head_amplitude_m > 0:
        return np.nan
    return float(disp_amplitude_m / head_amplitude_m)


def specific_storage(s_ke: float, thickness_m: float) -> float:
    """Elastic skeletal specific storage Ss = S_ke / b  [1/m].
    b = thickness of the deforming interval (layer thickness for MLCW; bulk
    compacting-column thickness for GNSS)."""
    if not thickness_m > 0:
        return np.nan
    return float(s_ke / thickness_m)


def skeletal_compressibility(ss: float, rho: float = RHO_W, g: float = G) -> float:
    """Skeletal compressibility alpha = Ss / (rho * g)  [1/Pa] from specific storage."""
    return float(ss / (rho * g))


def analyze_station(disp: pd.Series, head: pd.Series, period_years: float = 1.0,
                    resample: str | None = "D") -> dict:
    """Full per-station analysis. `disp` and `head` must both be in metres.
    Returns decomposition summaries, coupling, phase lag, and S_ke."""
    t0 = common_origin(disp, head)
    d = decompose(disp, period_years, t0=t0)
    h = decompose(head, period_years, t0=t0)
    r, n_common = couple(d.residual, h.residual, resample=resample)
    return {
        "disp_seas_amp_m": d.amplitude,
        "disp_seas_r2": d.seasonal_r2,
        "disp_trend_m_yr": d.trend_rate,
        "head_seas_amp_m": h.amplitude,
        "head_seas_r2": h.seasonal_r2,
        "coupling_r": r,
        "coupling_n": n_common,
        "phase_lag_days": phase_lag_days(d, h, period_years),
        "s_ke": storativity(d.amplitude, h.amplitude),
    }


def is_elastic_valid(coupling_r: float, disp_seasonal_r2: float,
                     r_min: float = 0.4, r2_min: float = 0.10) -> bool:
    """QC gate: keep a station's storativity only if the coupling is real, the
    annual harmonic is resolvable, and the response is in-phase (positive r)."""
    if not np.isfinite(coupling_r) or not np.isfinite(disp_seasonal_r2):
        return False
    return (coupling_r > r_min) and (disp_seasonal_r2 > r2_min)
