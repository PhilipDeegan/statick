#include "ipp.ipp"
#include "statick/solver/svrg.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 200;
size_t THREADS = mkn::kul::cpu::threads() < 12 ? mkn::kul::cpu::threads() : 12;
int main() {
  using T = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS = statick::Array<T>;
  using PROX = statick::ProxL2Sq<T>;
  using MODEL = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using SOLVER = statick::SVRG;
  using SODAO = SOLVER::DAO<MODEL, statick::solver::History<T, statick::solver::Tolerance<T>>>;
  auto SOLVE = &SOLVER::template SOLVE<SODAO, PROX>;
  MODEL::DAO modao(FEATURES::FROM_CEREAL("adult.features.cereal"),
                   LABELS::FROM_CEREAL("adult.labels.cereal"));
  const T STRENGTH = (1. / modao.n_samples()) + 1e-10;
  SODAO dao(modao, N_ITER, modao.n_samples(), THREADS);
  PROX prox(STRENGTH);
  auto start = NOW;
  dao.history.tol = 1e-5;
  SOLVE(dao, modao, prox);
  std::cout << (NOW - start) / 1e3 << std::endl;
  return 0;
}
