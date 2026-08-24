"""
Python tests for pyemdgrid.emd_l1.

Validates the Python bindings for emd_l1 and cross-checks the result against
POT's ot.emd2_lazy (exact network-simplex solver using cityblock distance) for
3-D grid histograms.
"""

import numpy as np
import pytest
import scipy.spatial
import scipy.special
import ot

import pyemdgrid

# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------


def _emd_l1_via_pot(hist1: np.ndarray, hist2: np.ndarray, return_matrix: bool = False):
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

    cost, log = ot.emd2_lazy(
        coords,
        coords,
        a,
        b,
        metric="cityblock",
        log=True,
        return_matrix=return_matrix,
    )
    if return_matrix:
        return float(cost), log["G"]
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
        assert scipy.sparse.issparse(plan)
        assert plan.shape == (4, 4)
        assert plan.sum() == pytest.approx(2.0)


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

    def test_transport_plan_matches_pot(self, rng):
        """Test transport plan on realistic 3D random histograms against POT."""
        shape = (4, 5, 6)
        h1, h2 = self._make_histograms(rng, shape)
        cost, plan = pyemdgrid.emd_l1(h1, h2, return_transport_plan=True)
        cost_pot, pot_G = _emd_l1_via_pot(h1, h2, return_matrix=True)

        pot_G_dense = pot_G.toarray() if scipy.sparse.issparse(pot_G) else pot_G
        our_G_dense = plan.toarray() if scipy.sparse.issparse(plan) else plan

        # Verify cost equality with EMD-L1 solver and POT
        assert cost == pytest.approx(cost_pot, rel=1e-5, abs=1e-8)

        # Verify plan properties match POT
        assert plan.nnz == pot_G.nnz
        assert plan.shape == pot_G.shape
        assert np.allclose(our_G_dense, pot_G_dense, atol=1e-5, rtol=1e-5)

        # Verify plan satisfies marginal constraints H1 and H2
        assert np.sum(our_G_dense, axis=1) == pytest.approx(
            h1.ravel(), rel=1e-5, abs=1e-8
        )
        assert np.sum(our_G_dense, axis=0) == pytest.approx(
            h2.ravel(), rel=1e-5, abs=1e-8
        )

    def test_transport_plan_exact_match_pot_2d(self):
        """Test exact element-wise transport plan match with POT on 2D grid."""
        h1 = np.array([[0.5, 0.0], [0.0, 0.5]])
        h2 = np.array([[0.0, 0.5], [0.5, 0.0]])
        cost, plan = pyemdgrid.emd_l1(h1, h2, return_transport_plan=True)
        cost_pot, pot_G = _emd_l1_via_pot(h1, h2, return_matrix=True)

        pot_G_dense = pot_G.toarray() if scipy.sparse.issparse(pot_G) else pot_G
        our_G_dense = plan.toarray() if scipy.sparse.issparse(plan) else plan

        assert cost == pytest.approx(cost_pot)
        assert plan.nnz == pot_G.nnz
        assert np.allclose(our_G_dense, pot_G_dense, atol=1e-5, rtol=1e-5)
