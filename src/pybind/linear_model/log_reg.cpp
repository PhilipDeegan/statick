
#include "statick/pybind/linear_model.hpp"

namespace py = pybind11;
/*
Function mapping convention:
 log_reg_fit_sd = s(sparse), d(double)
 log_reg_fit_dd = d(dense) , d(double)
*/

namespace statick {

using LOGREG_DAO_ds = statick_py::logreg::DAO<array2dv_s_ptr, arrayv_s_ptr>;
using LOGREG_DAO_ss = statick_py::logreg::DAO<sparse2dv_s_ptr, arrayv_s_ptr>;

using LOGREG_DAO_dd = statick_py::logreg::DAO<array2dv_d_ptr, arrayv_d_ptr>;
using LOGREG_DAO_sd = statick_py::logreg::DAO<sparse2dv_d_ptr, arrayv_d_ptr>;

PYBIND11_MODULE(statick_linear_model, m) {
  m.attr("__name__") = "statick.linear_model";
  py::class_<LOGREG_DAO_dd>(m, "log_reg_dao_dd")
      .def(py::init<py_array_t<double> &, py_array_t<double> &>());
  py::class_<LOGREG_DAO_sd>(m, "log_reg_dao_sd")
      .def(py::init<py_csr_t<double> &, py_array_t<double> &>());
  py::class_<LOGREG_DAO_ds>(m, "log_reg_dao_ds")
      .def(py::init<py_array_t<float> &, py_array_t<float> &>());
  py::class_<LOGREG_DAO_ss>(m, "log_reg_dao_ss")
      .def(py::init<py_csr_t<float> &, py_array_t<float> &>());
}
}  // namespace statick
