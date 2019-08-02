#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
#include "statick/solver/saga.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/linear_model/model_logreg.hpp"
namespace py = pybind11;

/*
Function mapping convention:
 log_reg_fit_sd = s(sparse), d(double)
 log_reg_fit_dd = d(dense) , d(double)
*/
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

namespace statick {

using LOGREG_DAO_sd_ptr = statick::logreg::DAO<sparse2dv_d_ptr, arrayv_d_ptr>;
using SAGA_LOG_REG_DAO_sd_ptr = statick::saga::sparse::DAO<LOGREG_DAO_sd_ptr, false>;

template <typename DAO>
void saga_solve_log_reg_s(DAO &dao, typename DAO::MODAO &modao){
  constexpr bool INTERCEPT = false;
  constexpr size_t N_ITER = 111;

  using T        = typename DAO::value_type;
  using FEATURES = statick::Sparse2DView<T>;
  using LABELS   = statick::ArrayView<T>;
  using PROX     = statick::ProxL2Sq<T>;
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using SOLVER   = statick::solver::SAGA<MODEL>;

  const size_t n_samples = modao.n_samples();
  KLOG(INF) << n_samples;

  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, n_samples - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };

  const T STRENGTH = (1. / n_samples) + 1e-10;
  PROX prox(STRENGTH); std::vector<T> objs; auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    SOLVER::SOLVE(dao, modao, prox, next_i);
    if (j % 10 == 0) objs.emplace_back(MODEL::LOSS(modao, dao.iterate.data()));
  }
}
template void saga_solve_log_reg_s<SAGA_LOG_REG_DAO_sd_ptr>(SAGA_LOG_REG_DAO_sd_ptr &, typename SAGA_LOG_REG_DAO_sd_ptr::MODAO &);

PYBIND11_MODULE(statick_solver, m) {
  m.attr("__name__") = "statick.solver";
  m.def("saga_solve_log_reg_sd_ptr", &saga_solve_log_reg_s<SAGA_LOG_REG_DAO_sd_ptr>, "solve sparse double");
  py::class_<SAGA_LOG_REG_DAO_sd_ptr>(m, "SAGA_LOG_REG_DAO_sd_ptr")
      .def(py::init<LOGREG_DAO_sd_ptr &>())
      .def_readwrite("step", &SAGA_LOG_REG_DAO_sd_ptr::step);
}

}  //  namespace statick
