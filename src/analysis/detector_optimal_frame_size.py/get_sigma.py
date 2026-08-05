"""Utilities to read an image `output.npy` and estimate peak values and sigmas.

The file `output.npy` is expected to contain a 2D numpy array with four
Gaussian spots. This module locates the brightest spots, computes their
peak values and estimates the Gaussian sigma using the weighted second
moment of intensity around each peak.

Usage:
	from get_sigma import load_and_compute
	results = load_and_compute('output.npy')
	for r in results:
		print(r)

"""
from __future__ import annotations

from pathlib import Path
from typing import List, Dict, Any

import numpy as np


def _weighted_sigma(window: np.ndarray, cx: int, cy: int) -> float:
	"""Estimate sigma (std dev) of a roughly Gaussian spot in `window`.

	Uses the intensity as weight and computes the root-mean-square radial
	distance from the center: sigma = sqrt(sum(w * r^2) / sum(w)).
	Background median is subtracted before weighting.
	"""
	yy, xx = np.indices(window.shape)
	dx = xx - cx
	dy = yy - cy
	r2 = dx * dx + dy * dy

	bg = np.median(window)
	w = window - bg
	w = np.clip(w, a_min=0.0, a_max=None)
	s = w.sum()
	if s <= 0:
		return 0.0
	mean_r2 = (w * r2).sum() / s
	return float(np.sqrt(mean_r2))


def load_and_compute(np_file: str | Path = "output.npy", n_spots: int = 4,
					 supp_radius: int = 12, window_radius: int = 20) -> List[Dict[str, Any]]:
	"""Load image from `np_file`, detect `n_spots` peaks and return results.

	Returns a list of dicts with keys: `x`, `y`, `peak`, `sigma`.
	- `peak` is the pixel value at the detected maximum
	- `sigma` is the estimated standard deviation in pixels
	"""
	p = Path(np_file)
	if not p.exists():
		raise FileNotFoundError(f"File not found: {p}")

	img = np.load(p)
	if img.ndim > 2:
		# collapse channels if needed
		img = img.mean(axis=2)
	img = img.astype(float)

	# Work on a copy because we will suppress detected peaks
	work = img.copy()
	h, w = work.shape

	results: List[Dict[str, Any]] = []

	for _ in range(n_spots):
		idx = int(np.argmax(work))
		py = idx // w
		px = idx % w
		peak_val = float(work[py, px])

		# define window around the peak for sigma estimation
		x0 = max(0, px - window_radius)
		x1 = min(w, px + window_radius + 1)
		y0 = max(0, py - window_radius)
		y1 = min(h, py + window_radius + 1)

		window = img[y0:y1, x0:x1]
		cx = px - x0
		cy = py - y0
		sigma = _weighted_sigma(window, cx, cy)

		results.append({"x": int(px), "y": int(py), "peak": peak_val, "sigma": sigma})

		# suppress neighborhood to avoid re-detecting the same spot
		sx0 = max(0, px - supp_radius)
		sx1 = min(w, px + supp_radius + 1)
		sy0 = max(0, py - supp_radius)
		sy1 = min(h, py + supp_radius + 1)
		work[sy0:sy1, sx0:sx1] = 0.0

	return results


if __name__ == "__main__":
	default = Path(__file__).parent / "output.npy"
	try:
		res = load_and_compute(default)
		for i, r in enumerate(res, 1):
			print(f"Spot {i}: (x={r['x']}, y={r['y']}), peak={r['peak']:.6g}, sigma={r['sigma']:.4f} px")
	except Exception as e:
		print("Error:", e)

