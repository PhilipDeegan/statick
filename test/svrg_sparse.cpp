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
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::svrg::sparse::DAO<MODAO, HISTORY>;
  std::string labels_s("adult.labels.cereal"), features_s("adult.features.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples(); // is used in "random_seq.ipp"
#include "random_seq.ipp"
  const auto STRENGTH = (1. / n_samples) + 1e-10;
  DAO dao(modao, N_ITER, n_samples, THREADS); PROX prox(STRENGTH); auto start = NOW;
  dao.history.tol.val = 1e-5;
  statick::svrg::sparse::solve<MODEL>(dao, modao, prox, next_i);
  std::cout << (NOW - start) / 1e3 << std::endl;
  return 0;
}
