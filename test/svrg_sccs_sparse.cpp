#include "kul/log.hpp"

#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "tick/array.hpp"
#include "tick/survival/model_sccs.hpp"
#include "tick/prox/prox_l2.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/prox/prox_tv.hpp"
#include "tick/solver/svrg.hpp"

#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()

constexpr bool INTERCEPT = false;
constexpr size_t N_SAMPLES = 50, N_FEATURES = 20, N_PERIODS = 10, N_ITER = 10, SEED = 1933;

int main() {
  using namespace tick::logreg::sparse;
  using namespace tick::svrg::sparse;
  tick::TModelSCCS<double>::DAO dao(N_SAMPLES, N_FEATURES);

  for (size_t i = 0; i < N_SAMPLES; ++i)
    dao.features[i] = tick::Array2D<double>::RANDOM(N_PERIODS, N_FEATURES, SEED);
  KLOG(INF);
  for (size_t i = 0; i < N_SAMPLES; ++i)
    dao.labels[i] = tick::Array<size_t>::RANDOM(N_PERIODS, SEED);
  KLOG(INF);

  std::vector<double> iterate(N_FEATURES + static_cast<uint>(INTERCEPT)), objs;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  std::mt19937_64 generator(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  KLOG(INF);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const auto STRENGTH = (1. / N_SAMPLES) + 1e-10;
  auto call = [&](const double *coeffs, double step, double *out, size_t size) {
    tick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };
  KLOG(INF);
  size_t t = 0;
  auto start = NOW;
  KLOG(INF);
  for (size_t j = 0; j < N_ITER; ++j) {
    KLOG(INF);
    solve<tick::TModelSCCS<double>>(dao, call, iterate.data(), next_i, t, 4, N_SAMPLES);
    KLOG(INF);
    if (j % 10 == 0) objs.emplace_back(tick::TModelSCCS<double>::loss(dao, iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
