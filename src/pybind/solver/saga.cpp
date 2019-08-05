#include "statick/pybind/linear_model.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/prox/prox_zero.hpp"
#include "statick/solver/saga.hpp"
#include "statick/random.hpp"
namespace py = pybind11;

namespace statick {

template <typename S, typename M, typename P>
auto def(pybind11::module &m, std::stringstream &st){
  std::stringstream ss;
  ss << "solve_" << S::NAME << "_" << M::NAME << "_" << P::NAME << "_" << st.str();
  m.def(ss.str().c_str(), &S::template SOLVE<M, P>, "solve sparse double prox l2sq");
}

template <typename S, typename M>
auto def(pybind11::module &m){
  using T = typename M::DAO::FEATURE::value_type;
  std::stringstream ss, st;
  if constexpr (M::DAO::FEATURE::is_sparse)     st << "s";
  else                                          st << "d";
  if constexpr (std::is_same<double, T>::value) st << "d";
  else                                          st << "s";
  def<S, M, statick::ProxZero<T>>(m, st);
  def<S, M, statick::ProxL2Sq<T>>(m, st);
  using D = typename S::template DAO<M, false>;
  ss << S::NAME << "_" << M::NAME << "_dao_" << st.str();
  return py::class_<D>(m, ss.str().c_str()).def(py::init<typename M::DAO &>())
                                           .def_readwrite("step", &D::step);
}

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";
  statick::def<statick::solver::SAGA, statick_py::ModelLogReg<sparse2dv_d_ptr, arrayv_d_ptr>>(m);
  statick::def<statick::solver::SAGA, statick_py::ModelLogReg<array2dv_d_ptr , arrayv_d_ptr>>(m);
  statick::def<statick::solver::SAGA, statick_py::ModelLogReg<sparse2dv_s_ptr, arrayv_s_ptr>>(m);
  statick::def<statick::solver::SAGA, statick_py::ModelLogReg<array2dv_s_ptr , arrayv_s_ptr>>(m);
}

}  //  namespace statick
