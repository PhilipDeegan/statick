#include "statick/pybind/linear_model.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"
#include "statick/solver/saga.hpp"
#include "statick/random.hpp"
namespace py = pybind11;

/*
Function mapping convention:
 log_reg_fit_sd = s(sparse), d(double)
 log_reg_fit_dd = d(dense) , d(double)
*/

namespace statick {

using PROX_ZERO_d = statick::ProxZero<double>;
using PROX_L2SQ_d = statick::ProxL2Sq<double>;

using LOGREG_DAO_sd = statick_py::logreg::DAO<sparse2dv_d_ptr, arrayv_d_ptr>;
using SAGA_LOG_REG_DAO_sd = statick::saga::sparse::DAO<LOGREG_DAO_sd, false>;
using saga_solve_log_reg_sd = statick::solver::SAGA<statick_py::ModelLogReg<sparse2dv_d_ptr, arrayv_d_ptr>>;

using LOGREG_DAO_dd = statick_py::logreg::DAO<array2dv_d_ptr , arrayv_d_ptr>;
using SAGA_LOG_REG_DAO_dd = statick::saga::dense::DAO<LOGREG_DAO_dd, false>;
using saga_solve_log_reg_dd = statick::solver::SAGA<statick_py::ModelLogReg<array2dv_d_ptr , arrayv_d_ptr>>;

using PROX_ZERO_s = statick::ProxZero<float>;
using PROX_L2SQ_s = statick::ProxL2Sq<float>;

using LOGREG_DAO_ss = statick_py::logreg::DAO<sparse2dv_s_ptr, arrayv_s_ptr>;
using SAGA_LOG_REG_DAO_ss = statick::saga::sparse::DAO<LOGREG_DAO_ss, false>;
using saga_solve_log_reg_ss = statick::solver::SAGA<statick_py::ModelLogReg<sparse2dv_s_ptr, arrayv_s_ptr>>;

using LOGREG_DAO_ds = statick_py::logreg::DAO<array2dv_s_ptr , arrayv_s_ptr>;
using SAGA_LOG_REG_DAO_ds = statick::saga::dense::DAO<LOGREG_DAO_ds, false>;
using saga_solve_log_reg_ds = statick::solver::SAGA<statick_py::ModelLogReg<array2dv_s_ptr , arrayv_s_ptr>>;

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";
  py::class_<SAGA_LOG_REG_DAO_sd>(m, "SAGA_LOG_REG_DAO_sd").def(py::init<LOGREG_DAO_sd &>())
                                                           .def_readwrite("step", &SAGA_LOG_REG_DAO_sd::step);
  m.def("saga_solve_log_reg_zero_sd", &saga_solve_log_reg_sd::SOLVE<PROX_ZERO_d>, "solve sparse double prox zero");
  m.def("saga_solve_log_reg_l2sq_sd", &saga_solve_log_reg_sd::SOLVE<PROX_L2SQ_d>, "solve sparse double prox l2sq");

  py::class_<SAGA_LOG_REG_DAO_dd>(m, "SAGA_LOG_REG_DAO_dd").def(py::init<LOGREG_DAO_dd &>())
                                                           .def_readwrite("step", &SAGA_LOG_REG_DAO_dd::step);
  m.def("saga_solve_log_reg_zero_dd", &saga_solve_log_reg_dd::SOLVE<PROX_ZERO_d>, "solve dense double prox zero");
  m.def("saga_solve_log_reg_l2sq_dd", &saga_solve_log_reg_dd::SOLVE<PROX_L2SQ_d>, "solve dense double prox l2sq");

  py::class_<SAGA_LOG_REG_DAO_ss>(m, "SAGA_LOG_REG_DAO_ss").def(py::init<LOGREG_DAO_ss &>())
                                                           .def_readwrite("step", &SAGA_LOG_REG_DAO_ss::step);
  m.def("saga_solve_log_reg_zero_ss", &saga_solve_log_reg_ss::SOLVE<PROX_ZERO_s>, "solve sparse single prox zero");
  m.def("saga_solve_log_reg_l2sq_ss", &saga_solve_log_reg_ss::SOLVE<PROX_L2SQ_s>, "solve sparse single prox l2sq");

  py::class_<SAGA_LOG_REG_DAO_ds>(m, "SAGA_LOG_REG_DAO_ds").def(py::init<LOGREG_DAO_ds &>())
                                                           .def_readwrite("step", &SAGA_LOG_REG_DAO_ds::step);
  m.def("saga_solve_log_reg_zero_ds", &saga_solve_log_reg_ds::SOLVE<PROX_ZERO_s>, "solve dense single prox zero");
  m.def("saga_solve_log_reg_l2sq_ds", &saga_solve_log_reg_ds::SOLVE<PROX_L2SQ_s>, "solve dense single prox l2sq");
}

}  //  namespace statick
