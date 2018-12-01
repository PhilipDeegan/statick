#include <atomic>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include "cereal/archives/portable_binary.hpp"
#include "kul/cpu.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/solver/asaga.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr bool INTERCEPT = false;
constexpr size_t N_ITER = 200;
int main() {
  size_t THREADS = kul::cpu::threads() < 12 ? kul::cpu::threads() : 12;
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::asaga::DAO<MODAO>;
  const std::string features_s("url.features.cereal"), labels_s("url.labels.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples();
#include "random_seq.ipp"
  const double STRENGTH = (1. / modao.n_samples()) + 1e-10;
  auto call_single = [&](const T x, T step) {
    return statick::prox_l2sq::call_single(x, step, STRENGTH);
  };
  DAO dao(modao, N_ITER, n_samples, THREADS);
  auto start = NOW;
  statick::asaga::sparse::solve<MODEL>(dao, modao, call_single, next_i);
  auto finish = NOW;
  std::cout << (finish - start) / 1e3 << std::endl;
  statick::sparse_2d::save(modao.features(), features_s);
  return 0;
}
