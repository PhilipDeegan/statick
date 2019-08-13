#include "ipp.ipp"
#include "statick/solver/sgd.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 111;
int main() {
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using PROX     = statick::ProxL2Sq<T>;
  using SOLVER   = statick::SGD;
  using SODAO    = SOLVER::DAO<MODEL>;
  auto  SOLVE    = &SOLVER::template SOLVE<SODAO, PROX>;
  MODEL::DAO modao(FEATURES::FROM_CEREAL("adult.features.cereal"),
                   LABELS::FROM_CEREAL("adult.labels.cereal"));
  SODAO dao(modao);
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
