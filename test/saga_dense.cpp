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
constexpr size_t N_FEATURES = 1000, N_SAMPLES = 1000, N_ITER = 100, SEED = 1933;
int main() {
  using T        = double;
  using FEATURES = statick::Array2D<T>;
  using LABELS   = statick::Array<T>;
  using MODAO    = statick::logreg::dense::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::saga::dense::DAO<MODAO>;
  MODAO modao(
      FEATURES::RANDOM(N_SAMPLES, N_FEATURES, SEED),
      LABELS::RANDOM(N_FEATURES, SEED));
  const size_t n_samples = modao.n_samples();
#include "random_seq.ipp"
  const T STRENGTH = (1. / n_samples) + 1e-10;
  auto call = [&](const T *coeffs, T step, T *out, size_t size) {
    statick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };
  DAO dao(modao);
  std::vector<T> objs;
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    statick::saga::dense::solve<MODEL>(dao, modao, call, next_i);
    if (j % 10 == 0)
      objs.emplace_back(
          statick::logreg::loss(modao.features(), modao.labels().data(), dao.iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
