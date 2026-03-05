import numpy as np
from typing import List, Optional, Tuple


###############################################################################
# Gray-box model simulators (discrete-time Euler, dt = 1)
###############################################################################


def simulate_inland(
	params: Tuple[float, float, float, float, float, float, float],
	t: np.ndarray,
	rainfall: np.ndarray,
	amp: np.ndarray,
	h_up: np.ndarray,
	h0: float,
) -> np.ndarray:
	# Inland ODE (discrete form, dt=1):
	#   h[t+1] = h[t]
	#            + [-a*(h[t] - z) + b*R[t] - c*AMP[t] + k_link*(h_up[t] - h[t])]
	#
	# NOTE:
	# - If using an upstream lag tau, pass h_up already aligned to the
	#   simulation window (e.g., h_up_lagged[t] = h_up_original[t - tau]).
	a, z, b, c, k_link, tau_rain, tau_up = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	max_lag = min(30, n - 1)
	if max_lag < 0:
		return h
	lag_idx = np.arange(max_lag + 1, dtype=float)
	wt_rain = np.exp(-lag_idx / max(tau_rain, 1e-6))
	wt_up = np.exp(-lag_idx / max(tau_up, 1e-6))
	# Pre-compute weighted sums for all time steps at once (vectorised).
	# r_conv[i-1] = sum_{τ=0}^{min(i-1, max_lag)} wt_rain[τ] * rainfall[i-1-τ]
	r_conv    = np.convolve(rainfall, wt_rain, mode='full')
	h_up_conv = np.convolve(h_up,    wt_up,   mode='full')
	for i in range(1, n):
		r_eff    = r_conv[i - 1]
		h_up_eff = h_up_conv[i - 1]
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * r_eff
			- c * amp[i - 1]
			+ k_link * (h_up_eff - h_prev)
		)
	return h


def simulate_coastal(
	params: Tuple[float, float, float, float, float, float, float, float, float, float],
	t: np.ndarray,
	rainfall: np.ndarray,
	amp: np.ndarray,
	amt: np.ndarray,
	h_up: np.ndarray,
	h0: float,
) -> np.ndarray:
	# Coastal ODE (discrete form, dt=1):
	#   h[t+1] = h[t]
	#            + [-a*(h[t] - z) + b*R[t] - c*AMP[t] + k_link*(h_up[t] - h[t])]
	#            - k_sgd*(h[t] - h_sea)
	#            + gamma*AMT[t]
	#
	# NOTE:
	# - If using an upstream lag tau, pass h_up already aligned to the
	#   simulation window.
	a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	max_lag = min(30, n - 1)
	if max_lag < 0:
		return h
	lag_idx = np.arange(max_lag + 1, dtype=float)
	wt_rain = np.exp(-lag_idx / max(tau_rain, 1e-6))
	wt_up = np.exp(-lag_idx / max(tau_up, 1e-6))
	# Pre-compute weighted sums for all time steps at once (vectorised).
	r_conv    = np.convolve(rainfall, wt_rain, mode='full')
	h_up_conv = np.convolve(h_up,    wt_up,   mode='full')
	for i in range(1, n):
		r_eff    = r_conv[i - 1]
		h_up_eff = h_up_conv[i - 1]
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * r_eff
			- c * amp[i - 1]
			+ k_link * (h_up_eff - h_prev)
			- k_sgd * (h_prev - h_sea)
			+ gamma * amt[i - 1]
		)
	return h


def simulate_inland_filtered(
	params: Tuple[float, float, float, float, float, float, float, float],
	t: np.ndarray,
	rainfall: np.ndarray,
	amp: np.ndarray,
	h_up: np.ndarray,
	h0: float,
) -> np.ndarray:
	# Inland ODE with low-pass filtered upstream driver (dt=1):
	#   u[t+1] = (1 - lambda) * u[t] + lambda * h_up[t]
	#   h[t+1] = h[t] + [-a*(h[t] - z) + b*R[t] - c*AMP[t] + k*(u[t] - h[t])]
	#
	# NOTE:
	# - If using upstream lag tau, pass h_up already aligned to the
	#   simulation window.
	a, z, b, c, k_link, lam, tau_rain, tau_up = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	max_lag = min(30, n - 1)
	if max_lag < 0:
		return h
	lag_idx = np.arange(max_lag + 1, dtype=float)
	wt_rain = np.exp(-lag_idx / max(tau_rain, 1e-6))
	wt_up = np.exp(-lag_idx / max(tau_up, 1e-6))
	# Pre-compute weighted sums for all time steps at once (vectorised).
	r_conv    = np.convolve(rainfall, wt_rain, mode='full')
	h_up_conv = np.convolve(h_up,    wt_up,   mode='full')
	u = float(h_up[0]) if len(h_up) > 0 else 0.0
	for i in range(1, n):
		r_eff    = r_conv[i - 1]
		h_up_eff = h_up_conv[i - 1]
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * r_eff
			- c * amp[i - 1]
			+ k_link * (u - h_prev)
		)
		u = (1.0 - lam) * u + lam * h_up_eff
	return h


def simulate_coastal_filtered(
	params: Tuple[float, float, float, float, float, float, float, float, float, float, float],
	t: np.ndarray,
	rainfall: np.ndarray,
	amp: np.ndarray,
	amt: np.ndarray,
	h_up: np.ndarray,
	h0: float,
) -> np.ndarray:
	# Coastal ODE with low-pass filtered upstream driver (dt=1):
	#   u[t+1] = (1 - lambda) * u[t] + lambda * h_up[t]
	#   h[t+1] = h[t] + [-a*(h[t] - z) + b*R[t] - c*AMP[t] + k*(u[t] - h[t])]
	#            - k_sgd*(h[t] - h_sea) + gamma*AMT[t]
	#
	# NOTE:
	# - If using upstream lag tau, pass h_up already aligned to the
	#   simulation window.
	a, z, b, c, k_link, k_sgd, gamma, h_sea, lam, tau_rain, tau_up = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	max_lag = min(30, n - 1)
	if max_lag < 0:
		return h
	lag_idx = np.arange(max_lag + 1, dtype=float)
	wt_rain = np.exp(-lag_idx / max(tau_rain, 1e-6))
	wt_up = np.exp(-lag_idx / max(tau_up, 1e-6))
	# Pre-compute weighted sums for all time steps at once (vectorised).
	r_conv    = np.convolve(rainfall, wt_rain, mode='full')
	h_up_conv = np.convolve(h_up,    wt_up,   mode='full')
	u = float(h_up[0]) if len(h_up) > 0 else 0.0
	for i in range(1, n):
		r_eff    = r_conv[i - 1]
		h_up_eff = h_up_conv[i - 1]
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * r_eff
			- c * amp[i - 1]
			+ k_link * (u - h_prev)
			- k_sgd * (h_prev - h_sea)
			+ gamma * amt[i - 1]
		)
		u = (1.0 - lam) * u + lam * h_up_eff
	return h


###############################################################################
# Define a wrapper that curve_fit will call
###############################################################################
def gw_model_wrapper(
	t: np.ndarray,
	*params: float,
	rainfall: np.ndarray,
	amp: np.ndarray,
	amt: Optional[np.ndarray],
	h_up: np.ndarray,
	h0: float,
	is_coastal: bool,
) -> np.ndarray:
	"""Wrapper for curve_fit.

	Parameters
	----------
	t : np.ndarray
		Time index (0..N-1).
	*params : float
		Model parameters. For inland: (a, z, b, c, k_link, tau_rain, tau_up).
		For coastal: (a, z, b, c, k_link, k_sgd, gamma, h_sea, tau_rain, tau_up).
	rainfall, amp, amt, h_up : np.ndarray
		Input series used by the groundwater model. If using upstream lag tau,
		pass a lagged/trimmed `h_up` aligned to the calibration window.
	h0 : float
		Initial groundwater level.
	is_coastal : bool
		If True, use the coastal model; otherwise inland.
	"""
	if is_coastal:
		return simulate_coastal(
			params=params,  # type: ignore[arg-type]
			t=t,
			rainfall=rainfall,
			amp=amp,
			amt=amt if amt is not None else np.zeros_like(t),
			h_up=h_up,
			h0=h0,
		)
	else:
		return simulate_inland(
			params=params,  # type: ignore[arg-type]
			t=t,
			rainfall=rainfall,
			amp=amp,
			h_up=h_up,
			h0=h0,
		)


def gw_model_wrapper_filtered(
	t: np.ndarray,
	*params: float,
	rainfall: np.ndarray,
	amp: np.ndarray,
	amt: Optional[np.ndarray],
	h_up: np.ndarray,
	h0: float,
	is_coastal: bool,
) -> np.ndarray:
	"""Wrapper for curve_fit for the filtered-upstream model.

	Parameters
	----------
	t : np.ndarray
		Time index (0..N-1).
	*params : float
		Model parameters. For inland: (a, z, b, c, k_link, lambda, tau_rain, tau_up).
		For coastal: (a, z, b, c, k_link, k_sgd, gamma, h_sea, lambda, tau_rain, tau_up).
	rainfall, amp, amt, h_up : np.ndarray
		Input series used by the groundwater model. If using upstream lag tau,
		pass a lagged/trimmed `h_up` aligned to the calibration window.
	h0 : float
		Initial groundwater level.
	is_coastal : bool
		If True, use the coastal model; otherwise inland.
	"""
	if is_coastal:
		return simulate_coastal_filtered(
			params=params,  # type: ignore[arg-type]
			t=t,
			rainfall=rainfall,
			amp=amp,
			amt=amt if amt is not None else np.zeros_like(t),
			h_up=h_up,
			h0=h0,
		)
	else:
		return simulate_inland_filtered(
			params=params,  # type: ignore[arg-type]
			t=t,
			rainfall=rainfall,
			amp=amp,
			h_up=h_up,
			h0=h0,
		)