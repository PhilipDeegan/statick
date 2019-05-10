#include "ipp.ipp"
#include "statick/solver/asaga.hpp"
constexpr size_t N_ITER = 200;
int main() {
  size_t THREADS = kul::cpu::threads() < 12 ? kul::cpu::threads() : 12;
  using T        = double;
  using HISTORY  = statick::solver::History<T, statick::solver::Tolerance<T>>;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using PROX     = statick::ProxL2Sq<T>;
  using MODAO    = statick::logreg::sparse::DAO<FEATURES, LABELS>;
  using MODEL    = statick::TModelLogReg<MODAO>;
  using DAO      = statick::asaga::DAO<MODAO, HISTORY>;
  const std::string features_s("url.features.cereal"), labels_s("url.labels.cereal");
  MODAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
#include "random_seq.ipp"
  const double STRENGTH = (1. / modao.n_samples()) + 1e-10;
  DAO dao(modao, N_ITER, n_samples, THREADS); PROX prox(STRENGTH);
  std::function<T(T*, size_t)> objectife = [&](T* iterate, size_t size){
    return statick::logreg::loss(modao.features(), modao.labels().data(), iterate)
              + statick::prox_l2sq::value(iterate, size, STRENGTH);;
  };
  dao.history.set_f_objective(objectife);
  statick::asaga::sparse::solve<MODEL>(dao, modao, prox, next_i);
  std::vector<double> &objs(dao.history.objectives);
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
