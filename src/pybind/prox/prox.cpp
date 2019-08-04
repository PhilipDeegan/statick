

#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"

namespace py = pybind11;

namespace statick{

PYBIND11_MODULE(statick_prox, m) {
  m.attr("__name__") = "statick.prox";
  py::class_<statick::ProxZero<double>>(m, "PROX_ZERO_d").def(py::init<>());
  py::class_<statick::ProxZero<float >>(m, "PROX_ZERO_s").def(py::init<>());
  py::class_<statick::ProxL2Sq<double>>(m, "PROX_L2SQ_d").def(py::init<double &>());
  py::class_<statick::ProxL2Sq<float >>(m, "PROX_L2SQ_s").def(py::init<float  &>());
}
}
