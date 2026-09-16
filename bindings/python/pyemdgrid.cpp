#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <span>
#include <stdexcept>
#include <type_traits>
#include <string>
#include <vector>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"
#include "emdgrid/mcf_lemon_l1.hpp"
#include "emdgrid/opencv_emd.hpp"

namespace py = pybind11;

namespace {

using DoubleArray = py::array_t<double, py::array::c_style>;

/// Builds the grid layout and both views for `Dim`, then runs `solve`.
///
/// GridDataView only borrows its layout, so the layout has to outlive the
/// views; keeping the whole lifetime inside one call makes that automatic.
/// `solve` is handed the dimensionality as a compile-time tag followed by the
/// two views.
template <std::size_t Dim, class Solve>
double solve_with_views(const DoubleArray& h1, const DoubleArray& h2,
                        Solve& solve) {
  typename emdgrid::GridLayout<Dim>::Shape shape{};
  for (std::size_t a = 0; a < Dim; ++a) {
    shape[a] = static_cast<std::size_t>(h1.shape(static_cast<py::ssize_t>(a)));
  }
  const emdgrid::GridLayout<Dim> layout(shape);
  const emdgrid::GridDataView<Dim, double> v1(
      layout, std::span<const double>(h1.data(), h1.size()));
  const emdgrid::GridDataView<Dim, double> v2(
      layout, std::span<const double>(h2.data(), h2.size()));
  return solve(std::integral_constant<std::size_t, Dim>{}, v1, v2);
}

/// Walks Dim upwards until it matches the arrays' runtime ndim.
template <std::size_t Dim, std::size_t MaxDim, class Solve>
double dispatch_from(const DoubleArray& h1, const DoubleArray& h2,
                     const char* solver, Solve& solve) {
  if (h1.ndim() == static_cast<py::ssize_t>(Dim)) {
    return solve_with_views<Dim>(h1, h2, solve);
  }
  if constexpr (Dim < MaxDim) {
    return dispatch_from<Dim + 1, MaxDim>(h1, h2, solver, solve);
  } else {
    throw std::invalid_argument(std::string(solver) + " only supports 1- to " +
                                std::to_string(MaxDim) +
                                "-dimensional histograms");
  }
}

/// Runs `solve` at the compile-time dimensionality the arrays call for.
///
/// The templated solvers need Dim at compile time, while numpy only reports it
/// at run time, so every entry point has to turn one into the other.
template <std::size_t MaxDim, class Solve>
double dispatch_by_ndim(const DoubleArray& h1, const DoubleArray& h2,
                        const char* solver, Solve solve) {
  if (h1.ndim() != h2.ndim()) {
    throw std::invalid_argument(
        "h1 and h2 must have the same number of dimensions");
  }
  return dispatch_from<1, MaxDim>(h1, h2, solver, solve);
}

/// Packages a solver result for Python.
///
/// Without a plan the result is the bare cost. With one it is (cost, plan),
/// where the plan is a scipy COO matrix when scipy is importable and the raw
/// SparseTransportPlan otherwise.
py::object make_result(double cost,
                       const emdgrid::SparseTransportPlan<>& plan,
                       std::size_t n_nodes, bool return_transport_plan) {
  if (!return_transport_plan) {
    return py::cast(cost);
  }
  try {
    py::module_ scipy_sparse = py::module_::import("scipy.sparse");
    const py::object coo_matrix = scipy_sparse.attr("coo_matrix")(
        py::make_tuple(plan.flow, py::make_tuple(plan.source, plan.target)),
        py::make_tuple(n_nodes, n_nodes));
    return py::make_tuple(cost, coo_matrix);
  } catch (const py::error_already_set&) {
    // Fallback if scipy is not installed
    return py::make_tuple(cost, plan);
  }
}

/// Python-level emd_l1: accepts any numpy array with ndim in {1,2,3}.
py::object emd_l1_py(const DoubleArray& h1, const DoubleArray& h2,
                     bool return_transport_plan = false,
                     int max_iter = 500000) {
  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<3>(
      h1, h2, "emd_l1", [&](auto dim, const auto& v1, const auto& v2) {
        if constexpr (decltype(dim)::value == 1) {
          static_cast<void>(max_iter);
          return emdgrid::emd_l1(v1, v2, plan_ptr);
        } else {
          return emdgrid::emd_l1(v1, v2, plan_ptr, max_iter);
        }
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

/// Python-level emd_sqeuclidean_1d: accepts 1-dimensional numpy arrays.
py::object emd_sqeuclidean_1d_py(const DoubleArray& h1, const DoubleArray& h2,
                                 bool return_transport_plan = false) {
  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<1>(
      h1, h2, "emd_sqeuclidean_1d",
      [&](auto, const auto& v1, const auto& v2) {
        return emdgrid::emd_sqeuclidean_1d(v1, v2, plan_ptr);
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

/// Python-level greedy_emd_l1_approx: accepts numpy array with ndim in {1,2,3}.
py::object greedy_emd_l1_approx_py(const DoubleArray& h1,
                                   const DoubleArray& h2,
                                   bool return_transport_plan = false) {
  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<3>(
      h1, h2, "greedy_emd_l1_approx",
      [&](auto, const auto& v1, const auto& v2) {
        return emdgrid::greedy_emd_l1_approx(v1, v2, plan_ptr);
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

emdgrid::McfLemonAlgorithm parse_mcf_lemon_algorithm(const py::object& obj) {
  if (py::isinstance<emdgrid::McfLemonAlgorithm>(obj)) {
    return obj.cast<emdgrid::McfLemonAlgorithm>();
  }
  if (py::isinstance<py::str>(obj)) {
    std::string s = obj.cast<std::string>();
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "network_simplex" || s == "networksimplex" || s == "ns") {
      return emdgrid::McfLemonAlgorithm::NetworkSimplex;
    }
    if (s == "cost_scaling" || s == "costscaling" || s == "cs") {
      return emdgrid::McfLemonAlgorithm::CostScaling;
    }
    throw std::invalid_argument(
        "Invalid algorithm string. Expected 'network_simplex' or "
        "'cost_scaling'.");
  }
  throw std::invalid_argument(
      "algorithm must be a McfLemonAlgorithm enum or string");
}

emdgrid::GroundMetric parse_ground_metric(const py::object& obj) {
  if (py::isinstance<emdgrid::GroundMetric>(obj)) {
    return obj.cast<emdgrid::GroundMetric>();
  }
  if (py::isinstance<py::str>(obj)) {
    std::string s = obj.cast<std::string>();
    std::transform(s.begin(), s.end(), s.begin(),
                   [](unsigned char c) { return std::tolower(c); });
    if (s == "l1") {
      return emdgrid::GroundMetric::L1;
    }
    if (s == "sqeuclidean" || s == "squared_euclidean" || s == "sq_euclidean") {
      return emdgrid::GroundMetric::SqEuclidean;
    }
    throw std::invalid_argument(
        "Invalid metric string. Expected 'l1' or 'sqeuclidean'.");
  }
  throw std::invalid_argument("metric must be a GroundMetric enum or string");
}

py::object knothe_rosenblatt_py(const DoubleArray& h1, const DoubleArray& h2,
                                const py::object& metric_obj = py::cast("l1"),
                                const py::object& dimension_order_obj =
                                    py::none(),
                                bool return_transport_plan = false) {
  const emdgrid::GroundMetric metric = parse_ground_metric(metric_obj);

  std::vector<std::size_t> dimension_order;
  if (!dimension_order_obj.is_none()) {
    dimension_order = dimension_order_obj.cast<std::vector<std::size_t>>();
  }

  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<4>(
      h1, h2, "knothe_rosenblatt",
      [&](auto, const auto& v1, const auto& v2) {
        return emdgrid::knothe_rosenblatt(
            v1, v2, metric, std::span<const std::size_t>(dimension_order),
            plan_ptr);
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

py::object dpartion_py(const DoubleArray& h1, const DoubleArray& h2,
                       const py::object& metric_obj = py::cast("l1"),
                       const py::object& algo_obj =
                           py::cast("network_simplex"),
                       bool return_transport_plan = false) {
  const emdgrid::GroundMetric metric = parse_ground_metric(metric_obj);
  const emdgrid::McfLemonAlgorithm algo = parse_mcf_lemon_algorithm(algo_obj);

  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<4>(
      h1, h2, "dpartion", [&](auto, const auto& v1, const auto& v2) {
        return emdgrid::mcf_dpartion(v1, v2, metric, algo, plan_ptr);
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

py::object opencv_emd_py(const DoubleArray& h1, const DoubleArray& h2,
                         const py::object& metric_obj = py::cast("l1"),
                         bool return_transport_plan = false) {
  const emdgrid::GroundMetric metric = parse_ground_metric(metric_obj);

  emdgrid::SparseTransportPlan<> plan;
  emdgrid::SparseTransportPlan<>* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  const double cost = dispatch_by_ndim<3>(
      h1, h2, "opencv_emd", [&](auto, const auto& v1, const auto& v2) {
        return emdgrid::opencv_emd(v1, v2, metric, plan_ptr);
      });

  return make_result(cost, plan, static_cast<std::size_t>(h1.size()),
                     return_transport_plan);
}

}  // namespace

PYBIND11_MODULE(pyemdgrid, module) {
  module.doc() = "Python bindings for emdgrid";
  module.def("version", []() { return std::string(emdgrid::version()); });

  py::class_<emdgrid::SparseTransportPlan<>>(module, "SparseTransportPlan")
      .def(py::init<>())
      .def_readwrite("source", &emdgrid::SparseTransportPlan<>::source)
      .def_readwrite("target", &emdgrid::SparseTransportPlan<>::target)
      .def_readwrite("flow", &emdgrid::SparseTransportPlan<>::flow);

  module.def(
      "emd_l1", &emd_l1_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("return_transport_plan") = false,
      py::arg("max_iter") = 500000,
      R"doc(
Compute the Earth Mover's Distance under the L1 (Manhattan) ground metric
for discrete histograms on a regular integer grid.

Both arrays must have the same shape and the same total mass (sum of
elements). The supported dimensionalities are 1, 2, and 3.

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First histogram (C-contiguous).
h2 : numpy.ndarray, dtype=float64
    Second histogram (C-contiguous, same shape as *h1*).
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan) where plan is a SparseTransportPlan.
max_iter : int, optional
    Maximum network-simplex pivot iterations (default: 500000).

Returns
-------
float or tuple(float, SparseTransportPlan)
    The EMD-L1 cost, or (cost, plan) if return_transport_plan is True.
)doc");

  module.def(
      "greedy_emd_l1_approx", &greedy_emd_l1_approx_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("return_transport_plan") = false,
      R"doc(
Compute an approximate Earth Mover's Distance under the L1 metric
using a greedy basic feasible solution initialization.

Both arrays must have the same shape and the same total mass.
The supported dimensionalities are 1, 2, and 3.

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First histogram (C-contiguous).
h2 : numpy.ndarray, dtype=float64
    Second histogram (C-contiguous, same shape as *h1*).
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan).

Returns
-------
float or tuple(float, SparseTransportPlan)
    The approximate EMD-L1 cost, or (cost, plan) if return_transport_plan is True.
)doc");

  module.def(
      "emd_sqeuclidean_1d", &emd_sqeuclidean_1d_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("return_transport_plan") = false,
      R"doc(
Compute the 1-D Optimal Transport cost under the squared Euclidean ground metric.

Both arrays must be 1-dimensional with the same shape and equal total mass.

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First 1-D histogram (C-contiguous).
h2 : numpy.ndarray, dtype=float64
    Second 1-D histogram (C-contiguous, same shape as *h1*).
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan).

Returns
-------
float or tuple(float, SparseTransportPlan)
    The 1-D squared Euclidean OT cost, or (cost, plan) if return_transport_plan is True.
)doc");

  py::enum_<emdgrid::McfLemonAlgorithm>(module, "McfLemonAlgorithm")
      .value("NetworkSimplex", emdgrid::McfLemonAlgorithm::NetworkSimplex)
      .value("CostScaling", emdgrid::McfLemonAlgorithm::CostScaling)
      .export_values();

  py::enum_<emdgrid::GroundMetric>(module, "GroundMetric")
      .value("L1", emdgrid::GroundMetric::L1)
      .value("SqEuclidean", emdgrid::GroundMetric::SqEuclidean)
      .value("SquaredEuclidean", emdgrid::GroundMetric::SqEuclidean)
      .export_values();

  module.def(
      "dpartion", &dpartion_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("metric") = "l1",
      py::arg("algorithm") = "network_simplex",
      py::arg("return_transport_plan") = false,
      R"doc(
Compute Optimal Transport on N-D grid histograms using (d+1)-partite graph MCF (Auricchio et al. 2018).

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First histogram (C-contiguous).
h2 : numpy.ndarray, dtype=float64
    Second histogram (C-contiguous, same shape as *h1*).
metric : GroundMetric or str, optional
    Ground metric choice: GroundMetric.L1 or "l1" (default) versus
    GroundMetric.SqEuclidean or "sqeuclidean".
algorithm : McfLemonAlgorithm or str, optional
    Solver algorithm choice: "network_simplex" (default) or "cost_scaling".
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan).

Returns
-------
float or tuple(float, SparseTransportPlan)
    The optimal transport cost, or (cost, plan) if return_transport_plan is True.
)doc");

  module.def(
      "knothe_rosenblatt", &knothe_rosenblatt_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("metric") = "l1",
      py::arg("dimension_order") = py::none(),
      py::arg("return_transport_plan") = false,
      R"doc(
Compute an N-D Knothe-Rosenblatt transport heuristic plan and ground cost.

Sequentially solves 1-D Optimal Transport problems along specified grid dimensions.

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First histogram (C-contiguous).
h2 : numpy.ndarray, dtype=float64
    Second histogram (C-contiguous, same shape as *h1*).
metric : GroundMetric or str, optional
    Ground metric choice: GroundMetric.L1 or "l1" (default) versus
    GroundMetric.SqEuclidean or "sqeuclidean".
dimension_order : sequence of int, optional
    Traversal order of dimensions (permutation of [0, ..., ndim-1]).
    Defaults to increasing order [0, 1, ..., ndim-1].
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan).

Returns
-------
float or tuple(float, SparseTransportPlan)
    The Knothe-Rosenblatt cost, or (cost, plan) if return_transport_plan is True.
)doc");

  module.def(
      "opencv_emd", &opencv_emd_py, py::arg("h1"), py::arg("h2"),
      py::arg("metric") = "l1", py::arg("return_transport_plan") = false,
      R"doc(
Compute Earth Mover's Distance using Rubner's transportation simplex.

This is a port of the OpenCV EMD implementation (opencv/modules/imgproc/src/emd_new.cpp,
branch 5.x), which itself is based on Yossi Rubner's 1998 code. The algorithm uses
Russell's initialization and the primal transportation simplex method.

Parameters
----------
h1 : numpy.ndarray, dtype=float64
    First histogram (C-contiguous, 1-, 2-, or 3-dimensional).
h2 : numpy.ndarray, dtype=float64
    Second histogram (C-contiguous, same shape as *h1*).
metric : GroundMetric or str, optional
    Ground metric choice: GroundMetric.L1 or "l1" (default) versus
    GroundMetric.SqEuclidean or "sqeuclidean".
return_transport_plan : bool, optional
    If True, return a tuple (cost, plan) where plan is a sparse COO matrix
    (scipy.sparse.coo_matrix if scipy is available, otherwise SparseTransportPlan).

Returns
-------
float or tuple(float, sparse matrix)
    The Earth Mover's Distance, or (cost, plan) if return_transport_plan is True.
)doc");
}
