#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "tick/array/array.hpp"
#include "tick/linear_model/model_logreg.hpp"
#include "tick/prox/prox_l2.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/prox/prox_tv.hpp"
#include "tick/solver/sgd.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr size_t N_ITER = 11;
constexpr bool INTERCEPT = false;
int main() {
  std::string labels_s("labels.cereal"), features_s("features.cereal");
  auto features = tick::Array2D<double>::FROM_CEREAL(features_s);
  auto labels = tick::Array<double>::FROM_CEREAL(labels_s);
  const size_t N_FEATURES = features->cols(), N_SAMPLES = features->rows();
  std::vector<double> gradients_average(N_FEATURES), gradients_memory(N_SAMPLES),
      iterate(N_FEATURES + static_cast<uint>(INTERCEPT)), objs;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937_64 generator(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const auto STRENGTH = (1. / N_SAMPLES) + 1e-10;
  auto call = [&](const double *coeffs, double step, double *out, size_t size) {
    tick::prox_tv::call(coeffs, step, out, size, STRENGTH);
  };
  size_t t = 0;
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    tick::sgd::dense::solve<INTERCEPT>(*features, labels->data(), iterate.data(), call, next_i, t);
    if (j % 10 == 0)
      objs.emplace_back(tick::logreg::loss(*features, labels->data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
