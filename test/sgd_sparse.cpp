#include "ipp.ipp"
#include "statick/solver/sgd.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 200;
int main() {
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using PROX     = statick::ProxL2Sq<T>;
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::sgd::DAO<MODAO>;
  const std::string features_s("adult.features.cereal"), labels_s("adult.labels.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples();
#include "random_seq.ipp"
  const T STRENGTH = (1. / n_samples) + 1e-10;
  DAO dao(modao); PROX prox(STRENGTH); std::vector<T> objs; auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    statick::sgd::sparse::solve<MODEL>(dao, modao, prox, next_i);
    if (j % 10 == 0)
      objs.emplace_back(
          statick::logreg::loss(modao.features(), modao.labels().data(), dao.iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
