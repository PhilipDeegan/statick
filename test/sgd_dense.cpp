#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "tick/array.hpp"
#include "tick/linear_model/model_logreg.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/solver/sgd.hpp"
#define NOW \
  std::chrono::duration_cast<std::chrono::milliseconds>(std::chrono::system_clock::now().time_since_epoch()).count()
constexpr size_t N_ITER = 11;
constexpr bool INTERCEPT = false;
int main() {
  using namespace tick::logreg::dense;
  using namespace tick::sgd::dense;
  std::string labels_s("labels.cereal"), features_s("features.cereal");
  tick::TModelLogReg<double, DAO<double>>::DAO modao(tick::Array2D<double>::FROM_CEREAL(features_s),
                                                     tick::Array<double>::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  std::vector<double> iterate(n_features), objs;
  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, n_samples - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const auto STRENGTH = (1. / n_samples) + 1e-10;
  auto call = [&](const double *coeffs, double step, double *out, size_t size) {
    tick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };
  size_t t = 0;
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    solve<tick::TModelLogReg<double, DAO<double>>>(modao, iterate.data(), call, next_i, t);
    if (j % 10 == 0) objs.emplace_back(tick::logreg::loss(modao.features(), modao.labels().data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
