#include "ipp.ipp"
#include "statick/thread/pool.hpp"
#include "statick/solver/svrg.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 200;
size_t THREADS = kul::cpu::threads() < 12 ? kul::cpu::threads() : 12;
int main() {
  using T        = double;
  using HISTORY  = statick::solver::History<T, statick::solver::Tolerance<T>>;
  using FEATURES = statick::Sparse2D<T>;
  using PROX     = statick::ProxL2Sq<T>;
  using LABELS   = statick::Array<T>;
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using SOLVER   = statick::solver::SVRG<MODEL, HISTORY>;
  std::string labels_s("adult.labels.cereal"), features_s("adult.features.cereal");
  MODEL::DAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const auto STRENGTH = (1. / modao.n_samples()) + 1e-10;
  SOLVER::DAO dao(modao, N_ITER, modao.n_samples(), THREADS); PROX prox(STRENGTH); auto start = NOW;
  dao.history.tol.val = 1e-5;
  SOLVER::SOLVE(dao, modao, prox);
  std::cout << (NOW - start) / 1e3 << std::endl;
  return 0;
}
