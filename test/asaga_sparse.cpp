#include "kul/log.hpp"
#include <atomic>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include "cereal/archives/portable_binary.hpp"
#include "tick/array.hpp"
#include "tick/linear_model/model_logreg.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/solver/saga.hpp"
#include "tick/solver/asaga.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr size_t N_ITER = 1, THREADS = 12;
constexpr bool INTERCEPT = false;
int main() {

  using namespace tick::logreg::sparse;
  using namespace tick::asaga::sparse;
  std::string labels_s("url.labels.cereal"), features_s("url.features.cereal");
  tick::TModelLogReg<double, DAO<double>>::DAO modao(
      tick::Sparse2D<double>::FROM_CEREAL(features_s), tick::Array<double>::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  std::vector<std::atomic<double>> gradients_average(n_features), gradients_memory(n_samples);
  std::vector<double> iterate(n_features + static_cast<uint>(INTERCEPT)), objs,
      steps_corrections(tick::saga::sparse::compute_step_corrections(modao.features()));
  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, n_samples - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const double STRENGTH = (1. / n_samples) + 1e-10;
  auto call_single = [&](size_t i, const double *coeffs, double step, double *out) {
    tick::prox_l2sq::call_single(i, coeffs, step, out, STRENGTH);
  };
  KLOG(INF) << n_samples;
  KLOG(INF) << n_features;
  auto start = NOW;
  tick::solver::NoHistory<double> history;
  for (size_t j = 0; j < N_ITER; ++j) {
    solve<tick::TModelLogReg<double, DAO<double>>>(modao, iterate.data(), steps_corrections.data(),
                                                   call_single, next_i, THREADS, history);
    // if (j % 10 == 0)
    //   objs.emplace_back(
    //       tick::logreg::loss(modao.features(), modao.labels().data(), iterate.data()));
  }

  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}
