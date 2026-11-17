import math

import numpy as np
import pytest

from tinker_cookbook.lr_calibration.fit import FitError, fit_curve, window_centres


def _synthetic(lrs: list[float], opt: float, edge_scale: float) -> list[float]:
    """Gentle under-training side plus steep instability side, minimum near ``opt``."""
    out = []
    for lr in lrs:
        z = math.log(lr / opt)
        out.append(
            0.7 * (1 + 0.004 * math.exp(-0.8 * z) + 0.002 * math.exp(2.5 * (z - edge_scale)))
        )
    return out


GRID = [1e-4 * 2**k for k in range(7)]


def test_fit_recovers_shift():
    """Shifting the curve horizontally by a factor shifts opt_lr and edge_lr by the same factor."""
    a = fit_curve(GRID, _synthetic(GRID, 4e-4, 0.0))
    shifted_grid = [lr * 1.7 for lr in GRID]
    b = fit_curve(shifted_grid, _synthetic(shifted_grid, 4e-4 * 1.7, 0.0))
    assert b.edge_lr / a.edge_lr == pytest.approx(1.7, rel=0.02)
    assert b.opt_lr / a.opt_lr == pytest.approx(1.7, rel=0.05)


def test_edge_above_opt():
    fit = fit_curve(GRID, _synthetic(GRID, 4e-4, 0.0))
    assert fit.opt_lr < fit.edge_lr


def test_cliff_does_not_pull_the_optimum_below_the_minimum():
    """With half-octave points around the minimum, the steep high-LR side is outside the fit."""
    fine = sorted({*GRID, *(4e-4 * 2 ** (k / 2) for k in (-1, 1))})
    losses = _synthetic(fine, 4e-4, 0.0)
    lowest = fine[losses.index(min(losses))]
    fit = fit_curve(fine, losses)
    assert lowest / 2**0.5 < fit.opt_lr < lowest * 2**0.5
    # The five factor-of-2 points around the minimum include the cliff, and their vertex is low.
    wide = fit_curve(GRID, _synthetic(GRID, 4e-4, 0.0), window=4.0)
    assert wide.opt_lr < fit.opt_lr


def test_concave_window_falls_back_to_the_points_next_to_the_minimum():
    lrs = [1e-4 * 2 ** (k / 2) for k in range(9)]
    losses = [0.72, 0.71, 0.7004, 0.7030, 0.7, 0.7030, 0.7004, 0.71, 0.75]
    fit = fit_curve(lrs, losses)
    assert fit.opt_lr == pytest.approx(lrs[4], rel=1e-6)


def test_window_moves_to_the_fitted_optimum():
    """On a flat bottom, the fit does not depend on which of the nearly equal points is lowest."""
    lrs = [1e-4 * 2 ** (k / 2) for k in range(9)]
    flat = [0.52, 0.51, 0.503, 0.5, 0.5, 0.5, 0.504, 0.515, 0.56]
    fits = []
    for lowest in (3, 4, 5):
        losses = list(flat)
        losses[lowest] -= 1e-6
        centres = window_centres(lrs, losses)
        assert centres[0] == pytest.approx(lrs[lowest]) and centres[-1] == pytest.approx(lrs[4])
        fits.append(fit_curve(lrs, losses).opt_lr)
    assert fits[0] == pytest.approx(fits[1], rel=1e-3) and fits[2] == pytest.approx(
        fits[1], rel=1e-3
    )
    assert lrs[3] < fits[1] < lrs[4]


def test_window_between_two_points_averages_their_fits():
    """A curve symmetric about the midpoint of two points makes the window alternate between
    them; the fit takes the mean of the two vertices, which is that midpoint."""
    lrs = [1e-4 * 2 ** (k / 2) for k in range(8)]
    losses = [0.7, 0.52, 0.503, 0.5, 0.5, 0.503, 0.52, 0.7]
    assert window_centres(lrs, losses) == [pytest.approx(lrs[3]), pytest.approx(lrs[4])]
    assert fit_curve(lrs, losses).opt_lr == pytest.approx(math.sqrt(lrs[3] * lrs[4]), rel=1e-6)


def test_rejects_minimum_at_grid_edge():
    losses = [float(v) for v in np.linspace(1.0, 0.9, len(GRID))]
    with pytest.raises(FitError, match="edge of the LR grid"):
        fit_curve(GRID, losses)


def test_rejects_curve_without_edge():
    losses = [0.701, 0.7005, 0.7, 0.7001, 0.7002, 0.7003, 0.7004]
    with pytest.raises(FitError, match="past the edge"):
        fit_curve(GRID, losses)


def test_rejects_too_few_points():
    with pytest.raises(FitError, match="at least 5"):
        fit_curve(GRID[:4], [0.71, 0.70, 0.705, 0.8])
