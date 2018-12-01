#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/solver/saga.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 111;
int main() {
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::saga::sparse::DAO<MODAO>;
  const std::string features_s("url.features.cereal"), labels_s("url.labels.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples();
#include "random_seq.ipp"
  const T STRENGTH = (1. / n_samples) + 1e-10;
  auto call_single = [&](size_t i, const T *coeffs, T step, T *out) {
    statick::prox_l2sq::call_single(i, coeffs, step, out, STRENGTH);
  };
  std::vector<T> objs;
  DAO dao(modao);
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    statick::saga::sparse::solve<MODEL>(dao, modao, call_single, next_i);
    if (j % 10 == 0)
      objs.emplace_back(
          statick::logreg::loss(modao.features(), modao.labels().data(), dao.iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
