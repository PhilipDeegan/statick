#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "kul/log.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/thread/pool.hpp"
#include "statick/solver/svrg.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 200;
size_t THREADS = kul::cpu::threads() < 12 ? kul::cpu::threads() : 12;
int main() {
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::svrg::sparse::DAO<MODAO>;
  std::string labels_s("adult.labels.cereal"), features_s("adult.features.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples(); // is used in "random_seq.ipp"
#include "random_seq.ipp"
  const auto STRENGTH = (1. / n_samples) + 1e-10;
  auto call = [&](size_t i, T x, T step, T *out) {
    out[i] = statick::prox_l2sq::call_single(x, step, STRENGTH);
  };
  DAO dao(modao, N_ITER, n_samples, THREADS);
  auto start = NOW;
  statick::svrg::sparse::solve<MODEL>(dao, modao, call, next_i);
  auto finish = NOW;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
