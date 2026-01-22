import numpy as np
from typing import List, Optional, Tuple


###############################################################################
# Gray-box model simulators (discrete-time Euler, dt = 1)
###############################################################################


def simulate_inland(
	params: Tuple[float, float, float, float, float],
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
	a, z, b, c, k_link = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	for i in range(1, n):
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * rainfall[i - 1]
			- c * amp[i - 1]
			+ k_link * (h_up[i - 1] - h_prev)
		)
	return h


def simulate_coastal(
	params: Tuple[float, float, float, float, float, float, float, float],
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
	a, z, b, c, k_link, k_sgd, gamma, h_sea = params
	n = len(t)
	h = np.zeros(n, dtype=float)
	h[0] = h0
	for i in range(1, n):
		h_prev = h[i - 1]
		h[i] = h_prev + (
			-a * (h_prev - z)
			+ b * rainfall[i - 1]
			- c * amp[i - 1]
			+ k_link * (h_up[i - 1] - h_prev)
			- k_sgd * (h_prev - h_sea)
			+ gamma * amt[i - 1]
		)
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
		Model parameters. For inland: (a, z, b, c, k_link).
		For coastal: (a, z, b, c, k_link, k_sgd, gamma, h_sea).
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