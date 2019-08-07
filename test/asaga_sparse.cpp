#include "ipp.ipp"
#include "statick/solver/asaga.hpp"
constexpr size_t N_ITER = 200;
size_t THREADS = kul::cpu::threads() < 12 ? kul::cpu::threads() : 12;
int main() {
  using T        = double;
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<T>;
  using MODEL    = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using PROX     = statick::ProxL2Sq<T>;
  using SOLVER   = statick::solver::ASAGA;
  using SODAO    = SOLVER::DAO<MODEL, statick::solver::History<T, statick::solver::Tolerance<T>>>;
  auto  SOLVE    = &SOLVER::template SOLVE<SODAO, PROX>;
  MODEL::DAO modao(FEATURES::FROM_CEREAL("url.features.cereal"),
                   LABELS::FROM_CEREAL("url.labels.cereal"));
  const T STRENGTH = (1. / modao.n_samples()) + 1e-10;
  SODAO dao(modao, N_ITER, modao.n_samples(), THREADS); PROX prox(STRENGTH);
  auto objectife = [&](T* iterate, size_t size){
    return MODEL::LOSS(modao, iterate) + PROX::value(prox, iterate, size);
  };
  dao.history.tol.val = 1e-5;
  dao.history.f_objective = objectife;
  SOLVE(dao, modao, prox);
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
