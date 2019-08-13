

#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"

namespace py = pybind11;

namespace statick{

PYBIND11_MODULE(statick_prox, m) {
  m.attr("__name__") = "statick.prox";
  py::class_<statick::ProxZero<double>>(m, "zero_d").def(py::init<>());
  py::class_<statick::ProxZero<float >>(m, "zero_s").def(py::init<>());
  py::class_<statick::ProxL2Sq<double>>(m, "l2sq_d").def(py::init<double &>());
  py::class_<statick::ProxL2Sq<float >>(m, "l2sq_s").def(py::init<float  &>());
}
}
