#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <span>
#include <string>
#include <vector>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"
#include "emdgrid/knothe_rosenblatt.hpp"

namespace py = pybind11;

namespace {

/// Dispatch emd_l1 for a numpy array of the given dimensionality.
template <std::size_t Dim>
double emd_l1_impl(const py::array_t<double, py::array::c_style>& h1,
                   const py::array_t<double, py::array::c_style>& h2,
                   emdgrid::SparseTransportPlan* plan = nullptr) {
  if (h1.ndim() != static_cast<py::ssize_t>(Dim) ||
      h2.ndim() != static_cast<py::ssize_t>(Dim)) {
    throw std::invalid_argument("array dimensionality does not match Dim");
  }
  typename emdgrid::GridLayout<Dim>::Shape shape{};
  for (std::size_t a = 0; a < Dim; ++a) {
    shape[a] = static_cast<std::size_t>(h1.shape(static_cast<py::ssize_t>(a)));
  }
  const emdgrid::GridLayout<Dim> layout(shape);
  const emdgrid::GridDataView<Dim, double> v1(
      layout, std::span<const double>(h1.data(), h1.size()));
  const emdgrid::GridDataView<Dim, double> v2(
      layout, std::span<const double>(h2.data(), h2.size()));
  return emdgrid::emd_l1(v1, v2, plan);
}

/// Python-level emd_l1: accepts any numpy array with ndim in {1,2,3}.
py::object emd_l1_py(const py::array_t<double, py::array::c_style>& h1,
                     const py::array_t<double, py::array::c_style>& h2,
                     bool return_transport_plan = false) {
  if (h1.ndim() != h2.ndim()) {
    throw std::invalid_argument(
        "h1 and h2 must have the same number of dimensions");
  }

  emdgrid::SparseTransportPlan plan;
  emdgrid::SparseTransportPlan* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  double cost = 0.0;
  switch (h1.ndim()) {
    case 1:
      cost = emd_l1_impl<1>(h1, h2, plan_ptr);
      break;
    case 2:
      cost = emd_l1_impl<2>(h1, h2, plan_ptr);
      break;
    case 3:
      cost = emd_l1_impl<3>(h1, h2, plan_ptr);
      break;
    default:
      throw std::invalid_argument(
          "emd_l1 only supports 1-, 2-, or 3-dimensional histograms");
  }

  if (return_transport_plan) {
    py::object coo_matrix;
    try {
      py::module_ scipy_sparse = py::module_::import("scipy.sparse");
      std::size_t n_nodes = static_cast<std::size_t>(h1.size());
      coo_matrix = scipy_sparse.attr("coo_matrix")(
          py::make_tuple(plan.flow,
                         py::make_tuple(plan.source, plan.target)),
          py::make_tuple(n_nodes, n_nodes));
    } catch (const py::error_already_set&) {
      // Fallback if scipy is not installed
      return py::make_tuple(cost, plan);
    }
    return py::make_tuple(cost, coo_matrix);
  }
  return py::cast(cost);
}

/// Python-level emd_sqeuclidean_1d: accepts 1-dimensional numpy arrays.
py::object emd_sqeuclidean_1d_py(
    const py::array_t<double, py::array::c_style>& h1,
    const py::array_t<double, py::array::c_style>& h2,
    bool return_transport_plan = false) {
  if (h1.ndim() != 1 || h2.ndim() != 1) {
    throw std::invalid_argument(
        "emd_sqeuclidean_1d only supports 1-dimensional histograms");
  }

  emdgrid::SparseTransportPlan plan;
  emdgrid::SparseTransportPlan* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  typename emdgrid::GridLayout<1>::Shape shape{
      static_cast<std::size_t>(h1.shape(0))};
  const emdgrid::GridLayout<1> layout(shape);
  const emdgrid::GridDataView<1, double> v1(
      layout, std::span<const double>(h1.data(), h1.size()));
  const emdgrid::GridDataView<1, double> v2(
      layout, std::span<const double>(h2.data(), h2.size()));

  double cost = emdgrid::emd_sqeuclidean_1d(v1, v2, plan_ptr);

  if (return_transport_plan) {
    py::object coo_matrix;
    try {
      py::module_ scipy_sparse = py::module_::import("scipy.sparse");
      std::size_t n_nodes = static_cast<std::size_t>(h1.size());
      coo_matrix = scipy_sparse.attr("coo_matrix")(
          py::make_tuple(plan.flow,
                         py::make_tuple(plan.source, plan.target)),
          py::make_tuple(n_nodes, n_nodes));
    } catch (const py::error_already_set&) {
      // Fallback if scipy is not installed
      return py::make_tuple(cost, plan);
    }
    return py::make_tuple(cost, coo_matrix);
  }
  return py::cast(cost);
}

/// Dispatch greedy_emd_l1_approx for a numpy array of given dimensionality.
template <std::size_t Dim>
double greedy_emd_l1_approx_impl(
    const py::array_t<double, py::array::c_style>& h1,
    const py::array_t<double, py::array::c_style>& h2,
    emdgrid::SparseTransportPlan* plan = nullptr) {
  if (h1.ndim() != static_cast<py::ssize_t>(Dim) ||
      h2.ndim() != static_cast<py::ssize_t>(Dim)) {
    throw std::invalid_argument("array dimensionality does not match Dim");
  }
  typename emdgrid::GridLayout<Dim>::Shape shape{};
  for (std::size_t a = 0; a < Dim; ++a) {
    shape[a] = static_cast<std::size_t>(h1.shape(static_cast<py::ssize_t>(a)));
  }
  const emdgrid::GridLayout<Dim> layout(shape);
  const emdgrid::GridDataView<Dim, double> v1(
      layout, std::span<const double>(h1.data(), h1.size()));
  const emdgrid::GridDataView<Dim, double> v2(
      layout, std::span<const double>(h2.data(), h2.size()));
  return emdgrid::greedy_emd_l1_approx(v1, v2, plan);
}

/// Python-level greedy_emd_l1_approx: accepts numpy array with ndim in {1,2,3}.
py::object greedy_emd_l1_approx_py(
    const py::array_t<double, py::array::c_style>& h1,
    const py::array_t<double, py::array::c_style>& h2,
    bool return_transport_plan = false) {
  if (h1.ndim() != h2.ndim()) {
    throw std::invalid_argument(
        "h1 and h2 must have the same number of dimensions");
  }

  emdgrid::SparseTransportPlan plan;
  emdgrid::SparseTransportPlan* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  double cost = 0.0;
  switch (h1.ndim()) {
    case 1:
      cost = greedy_emd_l1_approx_impl<1>(h1, h2, plan_ptr);
      break;
    case 2:
      cost = greedy_emd_l1_approx_impl<2>(h1, h2, plan_ptr);
      break;
    case 3:
      cost = greedy_emd_l1_approx_impl<3>(h1, h2, plan_ptr);
      break;
    default:
      throw std::invalid_argument(
          "greedy_emd_l1_approx only supports 1-, 2-, or 3-dimensional"
          " histograms");
  }

  if (return_transport_plan) {
    py::object coo_matrix;
    try {
      py::module_ scipy_sparse = py::module_::import("scipy.sparse");
      std::size_t n_nodes = static_cast<std::size_t>(h1.size());
      coo_matrix = scipy_sparse.attr("coo_matrix")(
          py::make_tuple(plan.flow,
                         py::make_tuple(plan.source, plan.target)),
          py::make_tuple(n_nodes, n_nodes));
    } catch (const py::error_already_set&) {
      // Fallback if scipy is not installed
      return py::make_tuple(cost, plan);
    }
    return py::make_tuple(cost, coo_matrix);
  }
  return py::cast(cost);
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

template <std::size_t Dim>
double knothe_rosenblatt_impl(
    const py::array_t<double, py::array::c_style>& h1,
    const py::array_t<double, py::array::c_style>& h2,
    emdgrid::GroundMetric metric,
    std::span<const std::size_t> dimension_order,
    emdgrid::SparseTransportPlan* plan = nullptr) {
  if (h1.ndim() != static_cast<py::ssize_t>(Dim) ||
      h2.ndim() != static_cast<py::ssize_t>(Dim)) {
    throw std::invalid_argument("array dimensionality does not match Dim");
  }
  typename emdgrid::GridLayout<Dim>::Shape shape{};
  for (std::size_t a = 0; a < Dim; ++a) {
    shape[a] = static_cast<std::size_t>(h1.shape(static_cast<py::ssize_t>(a)));
  }
  const emdgrid::GridLayout<Dim> layout(shape);
  const emdgrid::GridDataView<Dim, double> v1(
      layout, std::span<const double>(h1.data(), h1.size()));
  const emdgrid::GridDataView<Dim, double> v2(
      layout, std::span<const double>(h2.data(), h2.size()));
  return emdgrid::knothe_rosenblatt(v1, v2, metric, dimension_order, plan);
}

py::object knothe_rosenblatt_py(
    const py::array_t<double, py::array::c_style>& h1,
    const py::array_t<double, py::array::c_style>& h2,
    const py::object& metric_obj = py::cast("l1"),
    const py::object& dimension_order_obj = py::none(),
    bool return_transport_plan = false) {
  if (h1.ndim() != h2.ndim()) {
    throw std::invalid_argument(
        "h1 and h2 must have the same number of dimensions");
  }

  const emdgrid::GroundMetric metric = parse_ground_metric(metric_obj);

  std::vector<std::size_t> dimension_order;
  if (!dimension_order_obj.is_none()) {
    dimension_order = dimension_order_obj.cast<std::vector<std::size_t>>();
  }

  emdgrid::SparseTransportPlan plan;
  emdgrid::SparseTransportPlan* plan_ptr =
      return_transport_plan ? &plan : nullptr;

  double cost = 0.0;
  switch (h1.ndim()) {
    case 1:
      cost = knothe_rosenblatt_impl<1>(h1, h2, metric, dimension_order,
                                       plan_ptr);
      break;
    case 2:
      cost = knothe_rosenblatt_impl<2>(h1, h2, metric, dimension_order,
                                       plan_ptr);
      break;
    case 3:
      cost = knothe_rosenblatt_impl<3>(h1, h2, metric, dimension_order,
                                       plan_ptr);
      break;
    case 4:
      cost = knothe_rosenblatt_impl<4>(h1, h2, metric, dimension_order,
                                       plan_ptr);
      break;
    default:
      throw std::invalid_argument(
          "knothe_rosenblatt only supports 1-, 2-, 3-, or 4-dimensional"
          " histograms");
  }

  if (return_transport_plan) {
    py::object coo_matrix;
    try {
      py::module_ scipy_sparse = py::module_::import("scipy.sparse");
      std::size_t n_nodes = static_cast<std::size_t>(h1.size());
      coo_matrix = scipy_sparse.attr("coo_matrix")(
          py::make_tuple(plan.flow,
                         py::make_tuple(plan.source, plan.target)),
          py::make_tuple(n_nodes, n_nodes));
    } catch (const py::error_already_set&) {
      return py::make_tuple(cost, plan);
    }
    return py::make_tuple(cost, coo_matrix);
  }
  return py::cast(cost);
}

}  // namespace

PYBIND11_MODULE(pyemdgrid, module) {
  module.doc() = "Python bindings for emdgrid";
  module.def("version", []() { return std::string(emdgrid::version()); });

  py::class_<emdgrid::SparseTransportPlan>(module, "SparseTransportPlan")
      .def(py::init<>())
      .def_readwrite("source", &emdgrid::SparseTransportPlan::source)
      .def_readwrite("target", &emdgrid::SparseTransportPlan::target)
      .def_readwrite("flow", &emdgrid::SparseTransportPlan::flow);

  module.def(
      "emd_l1", &emd_l1_py,
      py::arg("h1"), py::arg("h2"),
      py::arg("return_transport_plan") = false,
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

  py::enum_<emdgrid::GroundMetric>(module, "GroundMetric")
      .value("L1", emdgrid::GroundMetric::L1)
      .value("SqEuclidean", emdgrid::GroundMetric::SqEuclidean)
      .value("SquaredEuclidean", emdgrid::GroundMetric::SqEuclidean)
      .export_values();

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
}
