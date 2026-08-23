#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <span>
#include <string>

#include "emdgrid/emd_l1.hpp"
#include "emdgrid/emdgrid.hpp"

namespace py = pybind11;

namespace {

/// Dispatch emd_l1 for a numpy array of the given dimensionality.
template <std::size_t Dim>
double emd_l1_impl(py::array_t<double, py::array::c_style> h1,
                   py::array_t<double, py::array::c_style> h2) {
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
  return emdgrid::emd_l1(v1, v2);
}

/// Python-level emd_l1: accepts any numpy array with ndim in {1,2,3}.
double emd_l1_py(py::array_t<double, py::array::c_style> h1,
                 py::array_t<double, py::array::c_style> h2) {
  if (h1.ndim() != h2.ndim()) {
    throw std::invalid_argument(
        "h1 and h2 must have the same number of dimensions");
  }
  switch (h1.ndim()) {
    case 1:
      return emd_l1_impl<1>(h1, h2);
    case 2:
      return emd_l1_impl<2>(h1, h2);
    case 3:
      return emd_l1_impl<3>(h1, h2);
    default:
      throw std::invalid_argument(
          "emd_l1 only supports 1-, 2-, or 3-dimensional histograms");
  }
}

}  // namespace

PYBIND11_MODULE(pyemdgrid, module) {
  module.doc() = "Python bindings for emdgrid";
  module.def("version", []() { return std::string(emdgrid::version()); });
  module.def(
      "emd_l1", &emd_l1_py,
      py::arg("h1"), py::arg("h2"),
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

Returns
-------
float
    The EMD-L1 cost.
)doc");
}
