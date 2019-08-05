#include "ipp.ipp"
#include "statick/solver/saga.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_FEATURES = 1000, N_SAMPLES = 1000, N_ITER = 100, SEED = 1933;
int main() {
  using T        = double;
  using FEATURES = statick::Array2D<T>;
  using LABELS   = statick::Array<T>;
  using SOLVER   = statick::solver::SAGA;
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using PROX     = statick::ProxL2Sq<T>;
  auto  SOLVE    = &SOLVER::template SOLVE<MODEL, PROX>;
  MODEL::DAO modao(FEATURES::RANDOM(N_SAMPLES, N_FEATURES, SEED),
                   LABELS::RANDOM(N_FEATURES, SEED));
  SOLVER::DAO<MODEL> dao(modao);
  PROX prox(/*strength=*/(1. / modao.n_samples()) + 1e-10);
  std::vector<T> objs; auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    SOLVE(dao, modao, prox);
    if (j % 10 == 0) objs.emplace_back(MODEL::LOSS(modao, dao.iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
