#include <random>
#include "kul/log.hpp"
#include <Python.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "statick/pybind/numpy.hpp"

namespace py = pybind11;

/*
Function mapping convention:
 log_reg_fit_sd = s(sparse), d(double)
 log_reg_fit_dd = d(dense) , d(double)
*/

template <typename T>
using py_array_t = py::array_t<T, py::array::c_style | py::array::forcecast>;
using py_array_d = py_array_t<double>;
using py_array_s = py_array_t<float>;

template <typename T>
using py_csr_t = py::csr_t<T, py::array::c_style | py::array::forcecast>;
using py_csr_d = py_csr_t<double>;
using py_csr_s = py_csr_t<float>;

namespace statick{

template <typename T>
void log_reg_fit_d(py_array_t<T> & a, py_array_t<T> & b){}
template void log_reg_fit_d<double>(py_array_d & a, py_array_d & b);

template <typename T>
void log_reg_fit_s(py_csr_t<T> & a, py_array_t<T> & b){}
template void log_reg_fit_s<double>(py_csr_d & a, py_array_d & b);

PYBIND11_MODULE(statick_linear_model, m) {
  m.attr("__name__") = "statick.linear_model";
  m.def("log_reg_fit_dd", &log_reg_fit_d<double>, "fit dense double");
  m.def("log_reg_fit_sd", &log_reg_fit_s<double>, "fit sparse double");
}
}
