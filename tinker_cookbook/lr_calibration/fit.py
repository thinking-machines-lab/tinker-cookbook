"""Locate the optimum and the instability edge of a loss-versus-learning-rate curve.

All estimates work in ``x = log(lr)`` and use simple, robust rules:

* **opt_lr** is the vertex of a least-squares quadratic through the points within
  a factor of ``window`` (default 2) of a centre. The centre starts at the
  lowest-loss learning rate and moves to the point nearest the vertex until the
  window is centred on its own vertex, so the result does not depend on which of
  several nearly equal points happens to be lowest. The loss rises much faster
  above the optimum than below it, so a wider window pulls the vertex below the
  measured minimum. The bottom of the curve is flat, so the fit needs closely
  spaced, low-noise points around the centre.
* **edge_lr** is where the measured loss, linearly interpolated in ``x``, first
  rises ``edge_tolerance`` (relative) above the minimum on the high-LR side. Loss
  rises steeply there, so the edge is located precisely even though the bottom of
  the curve is flat.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

EDGE_TOLERANCE = 0.01
"""Relative loss increase above the minimum that defines the instability edge."""


@dataclass(frozen=True)
class CurveFit:
    """Fitted features of one loss-versus-LR curve.

    Attributes:
        opt_lr (float): LR at the vertex of the quadratic fit through the points within
            ``window`` of the centre nearest that vertex.
        edge_lr (float): LR above ``opt_lr`` where the loss first reaches
            ``(1 + edge_tolerance) * min_loss``.
        min_loss (float): Lowest measured loss.
    """

    opt_lr: float
    edge_lr: float
    min_loss: float


class FitError(ValueError):
    """The curve cannot be fit reliably (too few points, minimum at the grid edge, no edge)."""


def _crossing(x0: float, y0: float, x1: float, y1: float, level: float) -> float:
    if y1 == y0:
        return x0
    return x0 + (level - y0) * (x1 - x0) / (y1 - y0)


def _vertex(x: np.ndarray, y: np.ndarray) -> float | None:
    """Vertex of the least-squares quadratic through the points, or None if it has no minimum."""
    if len(x) < 3:
        return None
    a, b, _ = np.polyfit(x, y, 2)
    return float(-b / (2 * a)) if a > 0 else None


def _window_vertex(x: np.ndarray, y: np.ndarray, i: int, half_width: float) -> float:
    """Vertex of the quadratic through the points within ``half_width`` of ``x[i]``, kept
    inside that window."""
    near = np.abs(x - x[i]) <= half_width
    x_opt = _vertex(x[near], y[near])
    if x_opt is None and 0 < i < len(x) - 1:
        # Noise can make the wider fit concave; try the point and its neighbours alone.
        x_opt = _vertex(x[i - 1 : i + 2], y[i - 1 : i + 2])
    if x_opt is None:
        x_opt = float(x[i])
    return float(np.clip(x_opt, x[i] - half_width, x[i] + half_width))


def _centred_fit(x: np.ndarray, y: np.ndarray, half_width: float) -> tuple[float, list[int]]:
    """Fit the window around the lowest loss, move the window to the point nearest its vertex,
    and repeat until the window is centred on its own vertex.

    Returns:
        The optimum and the indices of the centres tried. If the centre alternates between
        points, the optimum is the mean of their vertices.
    """
    i = int(np.argmin(y))
    vertices: dict[int, float] = {}
    while i not in vertices:
        vertices[i] = _window_vertex(x, y, i, half_width)
        i = int(np.argmin(np.abs(x - vertices[i])))
    centres = list(vertices)
    settled = centres[centres.index(i) :]
    return float(np.mean([vertices[c] for c in settled])), centres


def _sorted_logs(lrs: list[float], losses: list[float]) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(lrs)
    return np.log(np.asarray(lrs, dtype=float)[order]), np.asarray(losses, dtype=float)[order]


def _half_width(window: float) -> float:
    return math.log(window) * (1 + 1e-6)


def window_centres(lrs: list[float], losses: list[float], window: float = 2.0) -> list[float]:
    """Learning rates that the fit of ``opt_lr`` centres its window on, in the order tried.

    Args:
        lrs (list[float]): Peak learning rates of the non-diverged runs.
        losses (list[float]): Final held-out loss of each run.
        window (float): Factor around each centre that the fit uses.

    Returns:
        list[float]: The lowest-loss LR first, then each centre the window moved to.
    """
    x, y = _sorted_logs(lrs, losses)
    _, centres = _centred_fit(x, y, _half_width(window))
    return [math.exp(x[i]) for i in centres]


def fit_curve(
    lrs: list[float],
    losses: list[float],
    edge_tolerance: float = EDGE_TOLERANCE,
    window: float = 2.0,
) -> CurveFit:
    """Fit the loss-versus-LR curve of one sweep.

    Args:
        lrs (list[float]): Peak learning rates of the non-diverged runs.
        losses (list[float]): Final held-out loss of each run.
        edge_tolerance (float): Relative loss increase that defines the edge.
        window (float): The optimum is fitted to the points within this factor of the
            window's centre.

    Returns:
        CurveFit: Optimum, edge and lowest loss.

    Raises:
        FitError: If there are fewer than five points, fewer than two points on
            either side of the lowest loss, or no point past the edge.
    """
    if len(lrs) != len(losses):
        raise ValueError("lrs and losses must have the same length")
    if len(lrs) < 5:
        raise FitError(f"need at least 5 learning rates, got {len(lrs)}")
    x, y = _sorted_logs(lrs, losses)
    if not np.all(np.isfinite(y)):
        raise FitError("losses must be finite; drop diverged runs first")
    i = int(np.argmin(y))
    if i < 2 or i > len(x) - 3:
        raise FitError(
            "need two learning rates on each side of the lowest loss; "
            "the minimum is at the edge of the LR grid, so extend the grid"
        )
    y_min = float(y[i])
    edge_level = y_min * (1 + edge_tolerance)
    above = [j for j in range(i + 1, len(x)) if y[j] > edge_level]
    if not above:
        raise FitError("no learning rate is past the edge; extend the grid upward")

    x_opt, _ = _centred_fit(x, y, _half_width(window))

    j = above[0]
    x_edge = _crossing(x[j - 1], y[j - 1], x[j], y[j], edge_level)

    return CurveFit(opt_lr=math.exp(x_opt), edge_lr=math.exp(x_edge), min_loss=y_min)
