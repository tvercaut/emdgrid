"""
Python tests for pyemdgrid.emd_l1.

Validates the Python bindings for emd_l1 and cross-checks the result against
POT's ot.emd2_lazy (exact network-simplex solver using cityblock distance) for
3-D grid histograms.
"""

import numpy as np
import pytest
import scipy.special
import ot

import pyemdgrid

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _emd_l1_via_pot(hist1: np.ndarray, hist2: np.ndarray) -> float:
    """
    Exact EMD-L1 reference via POT's lazy solver (ot.emd2_lazy).

    Uses ot.emd2_lazy with the cityblock (L1) metric so that distances are
    computed on-the-fly without materialising the full cost matrix.  This
    uses the classic network-simplex algorithm (no grid-graph exploitation),
    making it the natural apples-to-apples baseline for result correctness
    checks.
    """
    ndim = hist1.ndim
    axes = [np.arange(s) for s in hist1.shape]
    coords = (
        np.array(np.meshgrid(*axes, indexing="ij"))
        .reshape(ndim, -1)
        .T.astype(np.float64)
    )

    a = hist1.ravel().astype(np.float64)
    b = hist2.ravel().astype(np.float64)

    result = ot.emd2_lazy(coords, coords, a, b, metric="cityblock")
    # ot.emd2_lazy returns (cost, log_dict) in newer POT versions
    cost = result[0] if isinstance(result, tuple) else result
    return float(cost)


# ---------------------------------------------------------------------------
# Unit tests for the Python binding
# ---------------------------------------------------------------------------


class TestEmdL1Binding:
    """Basic correctness tests exercised via the Python binding."""

    def test_version_returns_string(self):
        assert isinstance(pyemdgrid.version(), str)
        assert len(pyemdgrid.version()) > 0

    def test_1d_identical_histograms_zero(self):
        h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert pyemdgrid.emd_l1(h, h) == pytest.approx(0.0)

    def test_1d_unit_shift_one_bin(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 1.0, 0.0])
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(1.0)

    def test_1d_unit_shift_two_bins(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(2.0)

    def test_1d_symmetry(self):
        h1 = np.array([0.5, 0.5, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.5, 0.5])
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(pyemdgrid.emd_l1(h2, h1))

    def test_2d_identical_histograms_zero(self):
        h = np.full((3, 3), 1.0 / 9)
        assert pyemdgrid.emd_l1(h, h) == pytest.approx(0.0)

    def test_2d_unit_shift_along_axis(self):
        h1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        h2 = np.array([[0.0, 1.0], [0.0, 0.0]])
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(1.0)

    def test_2d_diagonal_shift_costs_2(self):
        h1 = np.array([[1.0, 0.0], [0.0, 0.0]])
        h2 = np.array([[0.0, 0.0], [0.0, 1.0]])
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(2.0)

    def test_3d_diagonal_shift_costs_3(self):
        h1 = np.zeros((2, 2, 2))
        h2 = np.zeros((2, 2, 2))
        h1[0, 0, 0] = 1.0
        h2[1, 1, 1] = 1.0
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(3.0)

    def test_shape_mismatch_raises(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.emd_l1(h1, h2)

    def test_unsupported_ndim_raises(self):
        h = np.ones((2, 2, 2, 2)) / 16.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.emd_l1(h, h)

    def test_fortran_order_accepted(self):
        """Non-C-contiguous input should be accepted (pybind11 converts it)."""
        h1 = np.asfortranarray(np.array([[0.5, 0.5], [0.0, 0.0]]))
        h2 = np.asfortranarray(np.array([[0.0, 0.0], [0.5, 0.5]]))
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(1.0)

    def test_return_transport_plan_option(self):
        h1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        h2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        cost, plan = pyemdgrid.emd_l1(h1, h2, return_transport_plan=True)
        assert cost == pytest.approx(2.0)
        assert isinstance(plan, pyemdgrid.SparseTransportPlan)
        assert len(plan.source) == len(plan.target) == len(plan.flow)
        assert sum(plan.flow) == pytest.approx(2.0)


# ---------------------------------------------------------------------------
# Cross-validation against POT
# ---------------------------------------------------------------------------


class TestEmdL1VsPot:
    """Compare pyemdgrid.emd_l1 against POT's ot.emd2_lazy for 3-D grids."""

    @pytest.fixture()
    def rng(self):
        return np.random.default_rng(42)

    def _make_histograms(self, rng, shape):
        """Return a pair of normalised positive histograms over *shape*."""
        raw1 = rng.standard_normal(shape)
        raw2 = rng.standard_normal(shape)
        h1 = scipy.special.softmax(raw1.ravel()).reshape(shape)
        h2 = scipy.special.softmax(raw2.ravel()).reshape(shape)
        return h1, h2

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 4),
            (4, 5),
            (2, 3, 4),
            (3, 4, 5),
            (7, 6, 4),
        ],
    )
    def test_agrees_with_pot(self, rng, shape):
        h1, h2 = self._make_histograms(rng, shape)
        cost_lo = pyemdgrid.emd_l1(h1, h2)
        cost_pot = _emd_l1_via_pot(h1, h2)
        assert cost_lo == pytest.approx(cost_pot, rel=1e-5, abs=1e-8), (
            f"shape={shape}: emd_l1={cost_lo:.8f}  pot={cost_pot:.8f}  "
            f"diff={abs(cost_lo - cost_pot):.2e}"
        )

    def test_symmetry_3d(self, rng):
        h1, h2 = self._make_histograms(rng, (4, 3, 5))
        assert pyemdgrid.emd_l1(h1, h2) == pytest.approx(
            pyemdgrid.emd_l1(h2, h1), rel=1e-10
        )

    def test_transport_plan_reconstructs_cost_and_mass(self, rng):
        shape = (3, 4, 2)
        h1, h2 = self._make_histograms(rng, shape)
        cost, plan = pyemdgrid.emd_l1(h1, h2, return_transport_plan=True)

        ndim = len(shape)
        axes = [np.arange(s) for s in shape]
        coords = (
            np.array(np.meshgrid(*axes, indexing="ij"))
            .reshape(ndim, -1)
            .T.astype(np.float64)
        )

        reconstructed_cost = 0.0
        for src, tgt, flow in zip(plan.source, plan.target, plan.flow):
            dist = np.sum(np.abs(coords[src] - coords[tgt]))
            reconstructed_cost += flow * dist

        assert reconstructed_cost == pytest.approx(cost, rel=1e-5, abs=1e-8)
        assert sum(plan.flow) == pytest.approx(1.0, rel=1e-5, abs=1e-8)
