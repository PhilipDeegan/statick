#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include "cereal/archives/portable_binary.hpp"
#include "tick/array/array.hpp"
#include "tick/linear_model/model_logreg.hpp"
#include "tick/prox/prox_l2sq.hpp"
#include "tick/solver/saga.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
constexpr size_t N_ITER = 111;
int main() {
  std::string labels_s("url.labels.cereal"), features_s("url.features.cereal");
  auto features = tick::Sparse2D<double>::FROM_CEREAL(features_s);
  auto labels = tick::Array<double>::FROM_CEREAL(labels_s);
  const size_t N_FEATURES = features->cols(), N_SAMPLES = features->rows();
  std::vector<double> gradients_average(N_FEATURES), gradients_memory(N_SAMPLES),
      iterate(N_FEATURES),
      steps_corrections(tick::saga::sparse::compute_step_corrections(*features));
  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
  const double STRENGTH = (1. / N_SAMPLES) + 1e-10;
  auto call_single = [&](size_t i, const double *coeffs, double step, double *out) {
    tick::prox_l2sq::call_single(i, coeffs, step, out, STRENGTH);
  };
  std::vector<double> objs;
  auto start = NOW;
  for (size_t j = 0; j < N_ITER; ++j) {
    tick::saga::sparse::solve<false>(*features.get(), labels->data(), gradients_average.data(),
                              gradients_memory.data(), iterate.data(), steps_corrections.data(),
                              call_single, next_i);
    if (j % 10 == 0)
      objs.emplace_back(tick::logreg::loss(*features.get(), labels->data(), iterate.data()));
  }
  auto finish = NOW;
  for (auto &o : objs) std::cout << __LINE__ << " " << o << std::endl;
  std::cout << (finish - start) / 1e3 << std::endl;
  return 0;
}