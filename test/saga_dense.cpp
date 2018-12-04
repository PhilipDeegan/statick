
#include <cmath>
#include <math.h>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include <malloc.h>

#include "cereal/archives/portable_binary.hpp"
#include "tick/array/array.hpp"
#include "tick/linear_model/model_logreg.hpp"
#include "tick/prox/prox_l2.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/solver/saga.hpp"

#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

constexpr size_t N_ITER = 100;

int main() {
  std::string labels_s("labels.cereal"), features_s("features.cereal");
  auto features = tick::Array2D<double>::FROM_CEREAL(features_s);
  auto labels = tick::Array<double>::FROM_CEREAL(labels_s);
  const size_t N_FEATURES = features->cols(), N_SAMPLES = features->rows();

  std::vector<double> gradients_average(N_FEATURES), gradients_memory(N_SAMPLES),
      iterate(N_FEATURES);

  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };

  const auto BETA = 1e-10;
  const auto STRENGTH = (1. / N_SAMPLES) + BETA;

  auto call = [&](const double *coeffs, double step, double *out, size_t size) {
    tick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };

  std::vector<double> objs;
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    tick::saga::dense::solve(*features, labels->data(), gradients_average.data(),
                             gradients_memory.data(), iterate.data(), call, next_i);

    if (j % 10 == 0)
      objs.emplace_back(tick::logreg::loss(*features, labels->data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
