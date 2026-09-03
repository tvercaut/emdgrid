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

    def test_max_iter_parameter(self):
        h1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        h2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        cost_0 = pyemdgrid.emd_l1(h1, h2, max_iter=0)
        cost_full = pyemdgrid.emd_l1(h1, h2, max_iter=1000)
        assert cost_full == pytest.approx(2.0)
        assert cost_0 >= cost_full


class TestEmdSqeuclidean1dBinding:
    """Basic correctness tests for emd_sqeuclidean_1d via Python binding."""

    def test_1d_identical_histograms_zero(self):
        h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert pyemdgrid.emd_sqeuclidean_1d(h, h) == pytest.approx(0.0)

    def test_1d_unit_shift_one_bin(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 1.0, 0.0])
        assert pyemdgrid.emd_sqeuclidean_1d(h1, h2) == pytest.approx(1.0)

    def test_1d_unit_shift_two_bins(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        assert pyemdgrid.emd_sqeuclidean_1d(h1, h2) == pytest.approx(4.0)

    def test_1d_symmetry(self):
        h1 = np.array([0.5, 0.5, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.5, 0.5])
        assert pyemdgrid.emd_sqeuclidean_1d(h1, h2) == pytest.approx(
            pyemdgrid.emd_sqeuclidean_1d(h2, h1)
        )

    def test_shape_mismatch_raises(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.emd_sqeuclidean_1d(h1, h2)

    def test_unsupported_ndim_raises(self):
        h = np.ones((2, 2)) / 4.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.emd_sqeuclidean_1d(h, h)

    def test_return_transport_plan_option(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        cost, plan = pyemdgrid.emd_sqeuclidean_1d(h1, h2, return_transport_plan=True)
        assert cost == pytest.approx(4.0)
        assert scipy.sparse.issparse(plan)
        assert plan.shape == (3, 3)
        assert plan.sum() == pytest.approx(1.0)


class TestKnotheRosenblattBinding:
    """Basic correctness tests for pyemdgrid.knothe_rosenblatt."""

    def test_1d_identical_histograms_zero(self):
        h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert pyemdgrid.knothe_rosenblatt(h, h, metric="l1") == pytest.approx(0.0)
        assert pyemdgrid.knothe_rosenblatt(
            h, h, metric=pyemdgrid.GroundMetric.SqEuclidean
        ) == pytest.approx(0.0)

    def test_1d_matches_exact_1d(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 1.0])
        cost_l1 = pyemdgrid.knothe_rosenblatt(h1, h2, metric="l1")
        cost_sq = pyemdgrid.knothe_rosenblatt(h1, h2, metric="sqeuclidean")

        assert cost_l1 == pytest.approx(pyemdgrid.emd_l1(h1, h2))
        assert cost_sq == pytest.approx(pyemdgrid.emd_sqeuclidean_1d(h1, h2))

    def test_2d_metric_string_and_enum_choices(self):
        h1 = np.zeros((3, 3))
        h2 = np.zeros((3, 3))
        h1[0, 0] = 1.0
        h2[2, 2] = 1.0

        c_l1_str = pyemdgrid.knothe_rosenblatt(h1, h2, metric="l1")
        c_l1_enum = pyemdgrid.knothe_rosenblatt(
            h1, h2, metric=pyemdgrid.GroundMetric.L1
        )
        c_sq_str1 = pyemdgrid.knothe_rosenblatt(h1, h2, metric="sqeuclidean")
        c_sq_str2 = pyemdgrid.knothe_rosenblatt(h1, h2, metric="squared_euclidean")
        c_sq_enum = pyemdgrid.knothe_rosenblatt(
            h1, h2, metric=pyemdgrid.GroundMetric.SqEuclidean
        )

        assert c_l1_str == pytest.approx(4.0)
        assert c_l1_enum == pytest.approx(4.0)
        assert c_sq_str1 == pytest.approx(8.0)
        assert c_sq_str2 == pytest.approx(8.0)
        assert c_sq_enum == pytest.approx(8.0)

    def test_2d_dimension_order_permutation(self):
        h1 = np.array([[0.5, 0.0], [0.0, 0.5]])
        h2 = np.array([[0.0, 0.5], [0.0, 0.5]])

        cost_01 = pyemdgrid.knothe_rosenblatt(h1, h2, dimension_order=[0, 1])
        cost_10 = pyemdgrid.knothe_rosenblatt(h1, h2, dimension_order=[1, 0])

        assert isinstance(cost_01, float)
        assert isinstance(cost_10, float)

    def test_3d_diagonal_shift(self):
        h1 = np.zeros((2, 2, 2))
        h2 = np.zeros((2, 2, 2))
        h1[0, 0, 0] = 1.0
        h2[1, 1, 1] = 1.0

        assert pyemdgrid.knothe_rosenblatt(h1, h2, metric="l1") == pytest.approx(3.0)
        assert pyemdgrid.knothe_rosenblatt(
            h1, h2, metric="sqeuclidean"
        ) == pytest.approx(3.0)

    def test_4d_histogram_support(self):
        h1 = np.zeros((2, 2, 2, 2))
        h2 = np.zeros((2, 2, 2, 2))
        h1[0, 0, 0, 0] = 1.0
        h2[1, 1, 1, 1] = 1.0

        assert pyemdgrid.knothe_rosenblatt(h1, h2, metric="l1") == pytest.approx(4.0)

    def test_return_transport_plan_and_marginal_constraints(self):
        h1 = np.array([[0.6, 0.4], [0.0, 0.0]])
        h2 = np.array([[0.0, 0.0], [0.5, 0.5]])

        cost, plan = pyemdgrid.knothe_rosenblatt(
            h1, h2, metric="l1", return_transport_plan=True
        )

        assert scipy.sparse.issparse(plan)
        assert plan.shape == (4, 4)
        assert plan.sum() == pytest.approx(1.0)

        plan_dense = plan.toarray()
        assert np.sum(plan_dense, axis=1) == pytest.approx(h1.ravel())
        assert np.sum(plan_dense, axis=0) == pytest.approx(h2.ravel())

    def test_invalid_dimension_order_raises(self):
        h1 = np.ones((2, 2)) / 4.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.knothe_rosenblatt(h1, h1, dimension_order=[0])
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.knothe_rosenblatt(h1, h1, dimension_order=[1, 1])

    def test_invalid_metric_raises(self):
        h1 = np.ones((2, 2)) / 4.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.knothe_rosenblatt(h1, h1, metric="invalid_metric")

    def test_shape_mismatch_raises(self):
        h1 = np.ones((2, 2)) / 4.0
        h2 = np.ones((3, 3)) / 9.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.knothe_rosenblatt(h1, h2)


class TestDpartionBinding:
    """Basic correctness tests for pyemdgrid.dpartion."""

    def test_1d_identical_histograms_zero(self):
        h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert pyemdgrid.dpartion(h, h, metric="l1") == pytest.approx(0.0)
        assert pyemdgrid.dpartion(
            h, h, metric=pyemdgrid.GroundMetric.SqEuclidean
        ) == pytest.approx(0.0)

    def test_2d_shift_and_metrics(self):
        h1 = np.zeros((3, 3))
        h2 = np.zeros((3, 3))
        h1[0, 0] = 1.0
        h2[2, 2] = 1.0

        c_l1 = pyemdgrid.dpartion(h1, h2, metric="l1")
        c_sq = pyemdgrid.dpartion(h1, h2, metric="sqeuclidean")

        assert c_l1 == pytest.approx(4.0)
        assert c_sq == pytest.approx(8.0)

    def test_algorithms_network_simplex_and_cost_scaling(self):
        h1 = np.array([[0.5, 0.5], [0.0, 0.0]])
        h2 = np.array([[0.0, 0.0], [0.5, 0.5]])

        c_ns = pyemdgrid.dpartion(h1, h2, algorithm="network_simplex")
        c_cs = pyemdgrid.dpartion(h1, h2, algorithm="cost_scaling")
        c_enum = pyemdgrid.dpartion(
            h1, h2, algorithm=pyemdgrid.McfLemonAlgorithm.CostScaling
        )

        assert c_ns == pytest.approx(1.0)
        assert c_cs == pytest.approx(1.0)
        assert c_enum == pytest.approx(1.0)

    def test_3d_diagonal_shift(self):
        h1 = np.zeros((2, 2, 2))
        h2 = np.zeros((2, 2, 2))
        h1[0, 0, 0] = 1.0
        h2[1, 1, 1] = 1.0

        assert pyemdgrid.dpartion(h1, h2, metric="l1") == pytest.approx(3.0)
        assert pyemdgrid.dpartion(h1, h2, metric="sqeuclidean") == pytest.approx(3.0)

    def test_return_transport_plan_option(self):
        h1 = np.array([[0.5, 0.0], [0.0, 0.5]])
        h2 = np.array([[0.0, 0.5], [0.5, 0.0]])
        cost, plan = pyemdgrid.dpartion(
            h1, h2, metric="sqeuclidean", return_transport_plan=True
        )
        assert cost == pytest.approx(1.0)
        assert scipy.sparse.issparse(plan)
        assert plan.shape == (4, 4)
        assert plan.sum() == pytest.approx(1.0)

    def test_shape_mismatch_raises(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.dpartion(h1, h2)


class TestGreedyEmdL1ApproxBinding:
    """Basic correctness tests exercised via the Python binding for greedy_emd_l1_approx."""

    def test_1d_identical_histograms_zero(self):
        h = np.array([0.1, 0.2, 0.4, 0.2, 0.1])
        assert pyemdgrid.greedy_emd_l1_approx(h, h) == pytest.approx(0.0)

    def test_2d_identical_histograms_zero(self):
        h = np.full((3, 3), 1.0 / 9)
        assert pyemdgrid.greedy_emd_l1_approx(h, h) == pytest.approx(0.0)

    def test_2d_upper_bound(self):
        h1 = np.array([[0.5, 0.5], [0.0, 0.0]])
        h2 = np.array([[0.0, 0.0], [0.5, 0.5]])
        approx_cost = pyemdgrid.greedy_emd_l1_approx(h1, h2)
        exact_cost = pyemdgrid.emd_l1(h1, h2)
        assert exact_cost == pytest.approx(1.0)
        assert approx_cost >= exact_cost

    def test_return_transport_plan_option(self):
        h1 = np.array([[1.0, 0.0], [0.0, 1.0]])
        h2 = np.array([[0.0, 1.0], [1.0, 0.0]])
        cost, plan = pyemdgrid.greedy_emd_l1_approx(h1, h2, return_transport_plan=True)
        assert cost == pytest.approx(2.0)
        assert scipy.sparse.issparse(plan)
        assert plan.shape == (4, 4)
        assert plan.sum() == pytest.approx(2.0)

    def test_shape_mismatch_raises(self):
        h1 = np.array([1.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 1.0])
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.greedy_emd_l1_approx(h1, h2)

    def test_unsupported_ndim_raises(self):
        h = np.ones((2, 2, 2, 2)) / 16.0
        with pytest.raises((ValueError, RuntimeError)):
            pyemdgrid.greedy_emd_l1_approx(h, h)


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

    def test_transport_plan_matches_pot(self, rng, request):
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
        assert plan.shape == pot_G.shape

        if request.config.getoption("--strict-pot"):
            assert plan.nnz == pot_G.nnz
            assert np.allclose(our_G_dense, pot_G_dense, atol=1e-5, rtol=1e-5)
        else:
            max_nnz_diff = 0.01 * h1.size * h2.size
            assert abs(plan.nnz - pot_G.nnz) <= max_nnz_diff

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


class TestDpartionVsPot:
    """Compare pyemdgrid.dpartion against POT's ot.emd2 for 2D/3D grids."""

    @pytest.fixture()
    def rng(self):
        return np.random.default_rng(42)

    def _make_histograms(self, rng, shape):
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
            (3, 4, 3),
        ],
    )
    def test_l1_agrees_with_pot(self, rng, shape):
        h1, h2 = self._make_histograms(rng, shape)
        cost_dpart = pyemdgrid.dpartion(h1, h2, metric="l1")
        cost_pot = _emd_l1_via_pot(h1, h2)
        assert cost_dpart == pytest.approx(cost_pot, rel=1e-5, abs=1e-8)

    @pytest.mark.parametrize(
        "shape",
        [
            (3, 4),
            (4, 5),
            (2, 3, 4),
        ],
    )
    def test_sqeuclidean_agrees_with_pot(self, rng, shape):
        h1, h2 = self._make_histograms(rng, shape)
        cost_dpart = pyemdgrid.dpartion(h1, h2, metric="sqeuclidean")

        # Build explicit squared Euclidean cost matrix M
        ndim = len(shape)
        axes = [np.arange(s) for s in shape]
        coords = np.array(np.meshgrid(*axes, indexing="ij")).reshape(ndim, -1).T
        diffs = coords[:, None, :] - coords[None, :, :]
        M = np.sum(diffs**2, axis=-1).astype(np.float64)

        cost_pot = ot.emd2(h1.ravel(), h2.ravel(), M)
        assert cost_dpart == pytest.approx(cost_pot, rel=1e-5, abs=1e-8)


class TestEmdSqeuclidean1dVsPot:
    """Compare pyemdgrid.emd_sqeuclidean_1d against POT's ot.emd2 for 1-D grids."""

    @pytest.fixture()
    def rng(self):
        return np.random.default_rng(42)

    @pytest.mark.parametrize("n_bins", [5, 10, 25, 100])
    def test_agrees_with_pot(self, rng, n_bins):
        raw1 = rng.standard_normal(n_bins)
        raw2 = rng.standard_normal(n_bins)
        h1 = scipy.special.softmax(raw1)
        h2 = scipy.special.softmax(raw2)

        cost_our = pyemdgrid.emd_sqeuclidean_1d(h1, h2)

        M = (np.arange(n_bins)[:, None] - np.arange(n_bins)[None, :]) ** 2
        cost_pot = ot.emd2(h1, h2, M.astype(np.float64))

        assert cost_our == pytest.approx(cost_pot, rel=1e-5, abs=1e-8)
