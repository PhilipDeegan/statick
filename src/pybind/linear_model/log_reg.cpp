#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
#include "statick/linear_model/model_logreg.hpp"
namespace py = pybind11;
/*
Function mapping convention:
 log_reg_fit_sd = s(sparse), d(double)
 log_reg_fit_dd = d(dense) , d(double)
*/

namespace statick{

using LOGREG_DAO_dd = statick::logreg::DAO<array2dv_d, arrayv_d>;
using LOGREG_DAO_dd_ptr = statick::logreg::DAO<array2dv_d_ptr, arrayv_d_ptr>;
template <typename T, typename DAO>
DAO log_reg_fit_d(py_array_t<T> & a, py_array_t<T> & b){
  py::buffer_info a_info = a.request(), b_info = b.request();
  std::vector<size_t> ainfo(3);
  ainfo[0] = a_info.shape[1];
  ainfo[1] = a_info.shape[0];
  ainfo[2] = a_info.shape[0]*a_info.shape[1];
  DAO dao;
  dao.m_features = std::make_shared<statick::Array2DView<T>>((T*)a_info.ptr, ainfo.data());
  dao.m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
  dao.init();
  return dao;
}
template LOGREG_DAO_dd_ptr log_reg_fit_d<double, LOGREG_DAO_dd_ptr>(py_arrayv_d & a, py_arrayv_d & b);

using LOGREG_DAO_sd = statick::logreg::DAO<sparse2dv_d, arrayv_d>;
using LOGREG_DAO_sd_ptr = statick::logreg::DAO<sparse2dv_d_ptr, arrayv_d_ptr>;
template <typename T, typename DAO>
DAO log_reg_fit_s(py_csr_t<T> & a, py_array_t<T> & b){
  py::buffer_info b_info = b.request();
  DAO dao;
  dao.m_features = a.m_data_ptr;
  dao.m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
  dao.init();
  return dao;
}
template LOGREG_DAO_sd_ptr log_reg_fit_s<double, LOGREG_DAO_sd_ptr>(py_csr_d & a, py_arrayv_d & b);

PYBIND11_MODULE(statick_linear_model, m) {
  m.attr("__name__") = "statick.linear_model";
  py::class_<LOGREG_DAO_dd_ptr>(m, "LOGREG_DAO_dd_ptr").def("print", &LOGREG_DAO_dd_ptr::print);
  m.def("log_reg_fit_dd", &log_reg_fit_d<double, LOGREG_DAO_dd_ptr>, "fit dense double");
  py::class_<LOGREG_DAO_sd_ptr>(m, "LOGREG_DAO_sd_ptr").def("print", &LOGREG_DAO_sd_ptr::print);
  m.def("log_reg_fit_sd", &log_reg_fit_s<double, LOGREG_DAO_sd_ptr>, "fit sparse double");
}
}
