#include <random>
#include "kul/log.hpp"
#include <Python.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/solver/saga.hpp"
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


namespace statick {

// using FEATURES = statick::Sparse2D<T>;
// using LABELS   = statick::Array<T>;
// using PROX     = statick::ProxL2Sq<T>;

using sparse2d_d     = statick::Sparse2D<double>;
using sparse2d_d_ptr = std::shared_ptr<sparse2d_d>;

using array2d_d     = statick::Array2D<double>;
using array2d_d_ptr = std::shared_ptr<array2d_d>;

using LOGREG_DAO_sd = statick::logreg::DAO<sparse2d_d, array2d_d>;
using SAGA_DAO_sd = statick::saga::dense::DAO<LOGREG_DAO_sd, false>;

using LOGREG_DAO_sd_ptr = statick::logreg::DAO<sparse2d_d_ptr, array2d_d_ptr>;
using SAGA_LOG_REG_DAO_sd_ptr = statick::saga::dense::DAO<LOGREG_DAO_sd_ptr, false>;

template <typename DAO>
void saga_solve_log_reg_s(DAO &dao){}
template void saga_solve_log_reg_s<SAGA_LOG_REG_DAO_sd_ptr>(SAGA_LOG_REG_DAO_sd_ptr &);

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";
  m.def("saga_solve_log_reg_sd_ptr", &saga_solve_log_reg_s<SAGA_LOG_REG_DAO_sd_ptr>, "solve sparse double");
  py::class_<SAGA_LOG_REG_DAO_sd_ptr>(m, "SAGA_LOG_REG_DAO_sd_ptr")
      .def(py::init<>())
      .def_readwrite("step", &SAGA_LOG_REG_DAO_sd_ptr::step);
}

}  //  namespace statick
