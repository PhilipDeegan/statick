#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include "mkn/kul/cpu.hpp"
#include "statick/array.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/solver/svrg.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_FEATURES = 1000, N_SAMPLES = 1000, N_ITER = 100, SEED = 1933;
size_t THREADS = mkn::kul::cpu::threads() < 12 ? mkn::kul::cpu::threads() : 12;
int main() {
  using T = double;
  using FEATURES = statick::Array2D<T>;
  using LABELS = statick::Array<T>;
  using PROX = statick::ProxL2Sq<T>;
  using MODEL = statick::ModelLogReg<std::shared_ptr<FEATURES>, std::shared_ptr<LABELS>>;
  using SOLVER = statick::SVRG;
  using SODAO = SOLVER::DAO<MODEL, statick::solver::History<T, statick::solver::Tolerance<T>>>;
  auto SOLVE = &SOLVER::template SOLVE<SODAO, PROX>;
  MODEL::DAO modao(FEATURES::RANDOM(N_SAMPLES, N_FEATURES, SEED), LABELS::RANDOM(N_FEATURES, SEED));
  const T STRENGTH = (1. / modao.n_samples()) + 1e-10;
  SODAO dao(modao, N_ITER, modao.n_samples(), THREADS);
  PROX prox(STRENGTH);
  auto start = mkn::kul::Now::MILLIS();
  dao.history.tol.val = 1e-5;
  SOLVE(dao, modao, prox);
  std::cout << (mkn::kul::Now::MILLIS() - start) / 1e3 << std::endl;
  return 0;
}
