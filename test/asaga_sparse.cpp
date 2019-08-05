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
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using SOLVER   = statick::solver::ASAGA<MODEL, HISTORY>;
  const std::string features_s("url.features.cereal"), labels_s("url.labels.cereal");
  MODEL::DAO modao(FEATURES::FROM_CEREAL(features_s), LABELS::FROM_CEREAL(labels_s));
  const T STRENGTH = (1. / modao.n_samples()) + 1e-10;
  SOLVER::DAO dao(modao, N_ITER, modao.n_samples(), THREADS); PROX prox(STRENGTH);
  auto objectife = [&](T* iterate, size_t size){
    return MODEL::LOSS(modao, iterate) + statick::prox_l2sq::value(iterate, size, STRENGTH);
  };
  dao.history.tol.val = 1e-5;
  dao.history.f_objective = objectife;
  SOLVER::SOLVE(dao, modao, prox);
  std::vector<T> &objs(dao.history.objectives);
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
