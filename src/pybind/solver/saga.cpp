#include "statick/pybind/linear_model.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"
#include "statick/solver/asaga.hpp"
#include "statick/solver/saga.hpp"
#include "statick/random.hpp"
namespace py = pybind11;

namespace statick {
template <typename M, typename D>
class PySolvers {
 public:
  static pybind11::class_<D> & INIT(pybind11::class_<D> &&m){
    return m.def(py::init<typename M::DAO &>());
  }
};

template <typename S, typename D, typename M, typename P>
auto def_solve(pybind11::module &m, std::stringstream &st){
  std::stringstream ss;
  ss << "solve_" << S::NAME << "_" << M::NAME << "_" << P::NAME << "_" << st.str();
  m.def(ss.str().c_str(), &S::template SOLVE<D, P>);
}

template <typename S, typename M, typename D = typename S::template DAO<M>>
auto def(pybind11::module &m, pybind11::class_<D>&(*init)(pybind11::class_<D> &&) = &PySolvers<M, D>::INIT ){
  using T = typename M::DAO::FEATURE::value_type;
  std::stringstream ss, st;
  if constexpr (M::DAO::FEATURE::is_sparse)     st << "s";
  else                                          st << "d";
  if constexpr (std::is_same<double, T>::value) st << "d";
  else                                          st << "s";
  def_solve<S, D, M, statick::ProxZero<T>>(m, st);
  def_solve<S, D, M, statick::ProxL2Sq<T>>(m, st);
  ss << S::NAME << "_" << M::NAME << "_dao_" << st.str();
  return init(py::class_<D>(m, ss.str().c_str())).def_readwrite("step", &D::step);
}

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";

  using log_reg_sd = statick_py::ModelLogReg<sparse2dv_d_ptr, arrayv_d_ptr>;
  using log_reg_ss = statick_py::ModelLogReg<sparse2dv_s_ptr, arrayv_s_ptr>;
  using log_reg_dd = statick_py::ModelLogReg<array2dv_d_ptr , arrayv_d_ptr>;
  using log_reg_ds = statick_py::ModelLogReg<array2dv_s_ptr , arrayv_s_ptr>;

  using _saga = statick::solver::SAGA;
  statick::def<_saga, log_reg_sd>(m);
  statick::def<_saga, log_reg_dd>(m);
  statick::def<_saga, log_reg_ss>(m);
  statick::def<_saga, log_reg_ds>(m);

  using tolerance_d = statick::solver::Tolerance<double>;
  py::class_<tolerance_d>(m, "tolerance_d").def_readwrite("val", &tolerance_d::val);
  using tolerance_s = statick::solver::Tolerance<float >;
  py::class_<tolerance_s>(m, "tolerance_s").def_readwrite("val", &tolerance_s::val);

  using history_tol_d = statick::solver::History<double, tolerance_d>;
  py::class_<history_tol_d>(m, "history_tol_d").def_readwrite("tol", &history_tol_d::tol);
  using history_tol_s = statick::solver::History<float , tolerance_s>;
  py::class_<history_tol_s>(m, "history_tol_s").def_readwrite("tol", &history_tol_s::tol);

  using _asaga = statick::solver::ASAGA;
  using _asaga_dao_d = typename _asaga::template DAO<log_reg_sd, history_tol_d>;
  using _asaga_dao_s = typename _asaga::template DAO<log_reg_ss, history_tol_s>;

  statick::def<statick::solver::ASAGA, log_reg_sd, _asaga_dao_d>(m, [](pybind11::class_<_asaga_dao_d>&& m) -> pybind11::class_<_asaga_dao_d>& {
    return m.def(py::init<typename log_reg_sd::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_asaga_dao_d::history);
  statick::def<statick::solver::ASAGA, log_reg_ss, _asaga_dao_s>(m, [](pybind11::class_<_asaga_dao_s>&& m) -> pybind11::class_<_asaga_dao_s>& {
    return m.def(py::init<typename log_reg_ss::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_asaga_dao_s::history);
  // statick::def<statick::solver::ASAGA, log_reg_ss, typename _asaga::template DAO<log_reg_ss, history_tol_s>>(m);

}

}  //  namespace statick
