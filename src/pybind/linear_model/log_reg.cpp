#include <random>
#include <Python.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include <pybind11/stl.h>
#include <pybind11/complex.h>
#include <pybind11/functional.h>
#include <pybind11/chrono.h>

#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
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

using array_d     = statick::ArrayView<double>;
using array_d_ptr = std::shared_ptr<array_d>;

using array2d_d     = statick::Array2DView<double>;
using array2d_d_ptr = std::shared_ptr<array2d_d>;

using sparse2d_d     = statick::Sparse2DView<double>;
using sparse2d_d_ptr = std::shared_ptr<sparse2d_d>;

using LOGREG_DAO_dd = statick::logreg::DAO<array2d_d, array_d>;
using LOGREG_DAO_dd_ptr = statick::logreg::DAO<array2d_d_ptr, array_d_ptr>;
template <typename T, typename DAO>
DAO log_reg_fit_d(py_array_t<T> & a, py_array_t<T> & b){
  auto make = [](){
    if constexpr(is_shared_ptr<DAO>::value) return std::make_shared<typename DAO::element_type>();
    else return DAO();
  };
  auto t_dao = make();
  DAO * dao = nullptr;
  if constexpr(is_shared_ptr<DAO>::value) dao = t_dao.get(); else dao = &t_dao;
  py::buffer_info a_info = a.request(), b_info = b.request();

  std::vector<size_t> ainfo(3);
  ainfo[0] = a_info.shape[1];
  ainfo[1] = a_info.shape[0];
  ainfo[2] = a_info.shape[0]*a_info.shape[1];

  if constexpr(is_shared_ptr<typename DAO::FEATURES>::value) {
    dao->m_features = std::make_shared<statick::Array2DView<T>>((T*)a_info.ptr, ainfo.data());
    dao->m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
  } else {
    dao->m_features = statick::Array2DView<T>((T*)a_info.ptr, ainfo.data());
    dao->m_labels = statick::ArrayView<T>((T*)b_info.ptr, b_info.shape[0]);
  }
  dao->init();
  return t_dao;
}
template LOGREG_DAO_dd_ptr log_reg_fit_d<double, LOGREG_DAO_dd_ptr>(py_array_d & a, py_array_d & b);

using LOGREG_DAO_sd = statick::logreg::DAO<sparse2d_d, array_d>;
using LOGREG_DAO_sd_ptr = statick::logreg::DAO<sparse2d_d_ptr, array_d_ptr>;

template <typename T, typename DAO>
DAO log_reg_fit_s(py_csr_t<T> & a, py_array_t<T> & b){
  auto make = [](){
    if constexpr(is_shared_ptr<DAO>::value) return std::make_shared<typename DAO::element_type>();
    else return DAO();
  };
  auto t_dao = make();
  DAO * dao = nullptr;
  if constexpr(is_shared_ptr<DAO>::value) dao = t_dao.get(); else dao = &t_dao;
  py::buffer_info b_info = b.request();

  if constexpr(is_shared_ptr<typename DAO::FEATURES>::value) {
    dao->m_features = a.m_data_ptr;
    dao->m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
  } else {
    dao->m_features = *a.m_data_ptr;
    dao->m_labels = statick::ArrayView<T>((T*)b_info.ptr, b_info.shape[0]);
  }
  dao->init();
  return t_dao;
}
template LOGREG_DAO_sd_ptr log_reg_fit_s<double, LOGREG_DAO_sd_ptr>(py_csr_d & a, py_array_d & b);

PYBIND11_MODULE(statick_linear_model, m) {
  m.attr("__name__") = "statick.linear_model";
  py::class_<LOGREG_DAO_dd_ptr>(m, "LOGREG_DAO_dd_ptr").def("print", &LOGREG_DAO_dd_ptr::print);
  m.def("log_reg_fit_dd", &log_reg_fit_d<double, LOGREG_DAO_dd_ptr>, "fit dense double");
  py::class_<LOGREG_DAO_sd_ptr>(m, "LOGREG_DAO_sd_ptr").def("print", &LOGREG_DAO_sd_ptr::print);
  m.def("log_reg_fit_sd", &log_reg_fit_s<double, LOGREG_DAO_sd_ptr>, "fit sparse double");
}
}
