#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <span>
#include <string>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"
#include "emdgrid/greedy_emd_l1.hpp"

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
}
