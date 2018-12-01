#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/vector.hpp"
#include "kul/os.hpp"
#include "kul/log.hpp"
#include "kul/signal.hpp"
#include "statick/array.hpp"
#include "statick/base_model/dao/model_lipschitz.hpp"
#include "statick/survival/model_sccs.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/solver/svrg.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr bool INTERCEPT = false;
constexpr size_t N_FEATURES = 10000, N_SAMPLES = 100, N_PERIODS = 100, N_ITER = 100, SEED = 1933;
constexpr size_t N_THREADS = 1;
int main() {
  using T        = double;
  using FEATURES = statick::Array2D<T>;
  using LABELS   = statick::Array<int32_t>;
  using MODAO    = statick::sccs::DAO<typename FEATURES::value_type>;
  using MODEL    = statick::TModelSCCS<MODAO>;
  using DAO      = statick::svrg::dense::DAO<MODAO>;
  std::vector<std::shared_ptr<statick::Array2D<T>>> features(N_SAMPLES);
  std::vector<std::shared_ptr<statick::Array<int32_t>>>  labels(N_SAMPLES);
  for (size_t i = 0; i < N_SAMPLES; ++i)
    features[i] = FEATURES::RANDOM(N_PERIODS, N_FEATURES, SEED);
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    labels[i] = LABELS::RANDOM(N_PERIODS, SEED);
    for (size_t j = 0; j < N_PERIODS; ++j) (*labels[i])[j] = (*labels[i])[j] % 2 == 0 ? 1 : 0;
  }
  MODAO madao(features, labels);
  const size_t n_samples = madao.n_samples();
#include "random_seq.ipp"
  const double STRENGTH = (1. / N_SAMPLES) + 1e-10;
  auto call = [&](const T *coeffs, T step, T *out, size_t size) {
    statick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };
  auto start = NOW;
  DAO dao(madao, N_ITER, N_SAMPLES, N_THREADS);
  statick::svrg::dense::solve<MODEL>(dao, madao, call, next_i);
  auto finish = NOW;
  std::cout << (finish - start) / 1e3 << std::endl;
  kul::File tick_interop("sccs.cereal");
  if(tick_interop) statick::sccs::load_from<T>(tick_interop.real());
  return 0;
}
