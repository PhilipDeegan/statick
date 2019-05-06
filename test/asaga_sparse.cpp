#include "ipp.ipp"
#include "statick/solver/asaga.hpp"
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
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
#include "random_seq.ipp"
  const double STRENGTH = (1. / modao.n_samples()) + 1e-10;
  auto call_single = [&](const T x, T step) {
    return statick::prox_l2sq::call_single(x, step, STRENGTH);
  };
  DAO dao(modao, N_ITER, n_samples, THREADS);
  statick::asaga::sparse::solve<MODEL>(dao, modao, call_single, next_i);
  std::vector<double> objs(dao.history.iterate_history.size());
  for (int i = 0; i < objs.size(); ++i) {
    objs[i] = statick::logreg::loss(modao.features(), modao.labels().data(), dao.history.iterate_history[i].data())
              + statick::prox_l2sq::value(
                  dao.history.iterate_history[i].data(),
                  dao.history.iterate_history[i].size(), STRENGTH);;
  }
  auto min_objective = *std::min_element(std::begin(objs), std::end(objs));
  auto history = dao.history.time_history;
  auto record_every = dao.history.record_every;
  for (size_t i = 1; i < objs.size(); i++) {
    auto log_dist = objs[i] == min_objective ? 0 : log10(objs[i] - min_objective);
    std::cout << THREADS << " " << i * record_every << " " << history[i] << " "
              << "1e" << log_dist << std::endl;
  }
  return 0;
}
