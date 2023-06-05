#ifndef STATICK_PYBIND_DEF_HPP_
#define STATICK_PYBIND_DEF_HPP_

#include "Python.h"

#include "pybind11/stl.h"
#include "pybind11/numpy.h"
#include "pybind11/chrono.h"
#include "pybind11/complex.h"
#include "pybind11/functional.h"

template <typename T>
using py_array_t = pybind11::array_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
using py_array_d = py_array_t<double>;
using py_array_s = py_array_t<float>;

#include "statick/pybind/csr.hpp"
#include "statick/pybind/numpy.hpp"

template <typename T>
using py_csr_t = pybind11::csr_t<T, pybind11::array::c_style | pybind11::array::forcecast>;
using py_csr_d = py_csr_t<double>;
using py_csr_s = py_csr_t<float>;

#endif  // STATICK_PYBIND_DEF_HPP_
