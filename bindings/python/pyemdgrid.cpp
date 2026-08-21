#include <pybind11/pybind11.h>

#include <string>

#include "emdgrid/emdgrid.hpp"

namespace py = pybind11;

PYBIND11_MODULE(pyemdgrid, module) {
  module.doc() = "Python bindings for emdgrid";
  module.def("version", []() { return std::string(emdgrid::version()); });
}
