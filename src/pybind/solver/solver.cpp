#include "statick/pybind/linear_model.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"
#include "statick/solver/asaga.hpp"
#include "statick/solver/saga.hpp"
#include "statick/solver/svrg.hpp"
#include "statick/solver/sgd.hpp"
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
auto def_solver(pybind11::module &m, pybind11::class_<D>&(*init)(pybind11::class_<D> &&) = &PySolvers<M, D>::INIT ){
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

template <typename T, typename TS, typename HS>
void def_history(pybind11::module &m, std::string &&ts, std::string &&hs){
  py::class_<TS>(m, ts.c_str()).def_readwrite("val", &TS::val);
  py::class_<HS>(m, hs.c_str()).def_readonly("tol", &HS::tol)
      .def_readonly("objectives", &HS::objectives)
      .def_readonly("time_history", &HS::time_history)
      .def_readwrite("log_every_n_epochs", &HS::log_every_n_epochs);
}

#define DEF_TO_XSTRING(s) DEF_TO_STRING(s)
#define DEF_TO_STRING(s) #s
#define DEF_HISTORY(T, ts, hs)                \
  using ts = statick::solver::Tolerance<T>;   \
  using hs = statick::solver::History<T, ts>; \
  statick::def_history<T, ts, hs>(m, DEF_TO_STRING(ts), DEF_TO_STRING(hs));

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";

  using log_reg_sd = statick_py::ModelLogReg<sparse2dv_d_ptr, arrayv_d_ptr>;
  using log_reg_ss = statick_py::ModelLogReg<sparse2dv_s_ptr, arrayv_s_ptr>;
  using log_reg_dd = statick_py::ModelLogReg<array2dv_d_ptr , arrayv_d_ptr>;
  using log_reg_ds = statick_py::ModelLogReg<array2dv_s_ptr , arrayv_s_ptr>;

  using _saga = statick::SAGA;
  statick::def_solver<_saga, log_reg_sd>(m); statick::def_solver<_saga, log_reg_dd>(m);
  statick::def_solver<_saga, log_reg_ss>(m); statick::def_solver<_saga, log_reg_ds>(m);

  using _sgd = statick::SGD;
  statick::def_solver<_sgd, log_reg_sd>(m); statick::def_solver<_sgd, log_reg_dd>(m);
  statick::def_solver<_sgd, log_reg_ss>(m); statick::def_solver<_sgd, log_reg_ds>(m);

  DEF_HISTORY(double, tolerance_d, history_tol_d);
  DEF_HISTORY(float , tolerance_s, history_tol_s);

  using _asaga = statick::ASAGA;
  using _asaga_dao_d = typename _asaga::template DAO<log_reg_sd, history_tol_d>;
  using _asaga_dao_s = typename _asaga::template DAO<log_reg_ss, history_tol_s>;
  statick::def_solver<_asaga, log_reg_sd, _asaga_dao_d>(
    m, [](pybind11::class_<_asaga_dao_d>&& m) -> pybind11::class_<_asaga_dao_d>& {
    return m.def(py::init<typename log_reg_sd::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_asaga_dao_d::history);
  statick::def_solver<_asaga, log_reg_ss, _asaga_dao_s>(
    m, [](pybind11::class_<_asaga_dao_s>&& m) -> pybind11::class_<_asaga_dao_s>& {
    return m.def(py::init<typename log_reg_ss::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_asaga_dao_s::history);

  using _svrg = statick::SVRG;
  using _svrg_dao_sd = typename _svrg::template DAO<log_reg_sd, history_tol_d>;
  using _svrg_dao_ss = typename _svrg::template DAO<log_reg_ss, history_tol_s>;
  statick::def_solver<_svrg, log_reg_sd, _svrg_dao_sd>(
    m, [](pybind11::class_<_svrg_dao_sd>&& m) -> pybind11::class_<_svrg_dao_sd>& {
    return m.def(py::init<typename log_reg_sd::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_svrg_dao_sd::history);
  statick::def_solver<_svrg, log_reg_ss, _svrg_dao_ss>(
    m, [](pybind11::class_<_svrg_dao_ss>&& m) -> pybind11::class_<_svrg_dao_ss>& {
    return m.def(py::init<typename log_reg_ss::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_svrg_dao_ss::history);

  using _svrg_dao_dd = typename _svrg::template DAO<log_reg_dd, history_tol_d>;
  using _svrg_dao_ds = typename _svrg::template DAO<log_reg_ds, history_tol_s>;
  statick::def_solver<_svrg, log_reg_dd, _svrg_dao_dd>(
    m, [](pybind11::class_<_svrg_dao_dd>&& m) -> pybind11::class_<_svrg_dao_dd>& {
    return m.def(py::init<typename log_reg_dd::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_svrg_dao_dd::history);
  statick::def_solver<_svrg, log_reg_ds, _svrg_dao_ds>(
    m, [](pybind11::class_<_svrg_dao_ds>&& m) -> pybind11::class_<_svrg_dao_ds>& {
    return m.def(py::init<typename log_reg_ds::DAO &, size_t, size_t, size_t>());
  }).def_readonly("history", &_svrg_dao_ds::history);
}
#undef DEF_TO_XSTRING
#undef DEF_TO_STRING
#undef DEF_HISTORY

}  //  namespace statick
