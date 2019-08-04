#include "ipp.ipp"
#include "statick/base_model/dao/model_lipschitz.hpp"
#include "statick/survival/model_sccs.hpp"
#include "statick/thread/pool.hpp"
#include "statick/solver/svrg.hpp"
constexpr bool INTERCEPT = false;
constexpr size_t N_FEATURES = 10000, N_SAMPLES = 100, N_PERIODS = 100, N_ITER = 100, SEED = 1933;
constexpr size_t N_THREADS = 1;
int main() {
  using T        = double;
  using HISTORY  = statick::solver::History<T, statick::solver::Tolerance<T>>;
  using FEATURES = statick::Array2D<T>;
  using LABELS   = statick::Array<int32_t>;
  using PROX     = statick::ProxL2Sq<T>;
  using MODAO    = statick::sccs::DAO<typename FEATURES::value_type>;
  using MODEL    = statick::TModelSCCS<MODAO>;
  using DAO      = statick::svrg::dense::DAO<MODAO, HISTORY>;
  std::vector<std::shared_ptr<statick::Array2D<T>>> features(N_SAMPLES);
  std::vector<std::shared_ptr<statick::Array<int32_t>>>  labels(N_SAMPLES);
  for (size_t i = 0; i < N_SAMPLES; ++i)
    features[i] = FEATURES::RANDOM(N_PERIODS, N_FEATURES, SEED);
  for (size_t i = 0; i < N_SAMPLES; ++i) {
    labels[i] = LABELS::RANDOM(N_PERIODS, SEED);
    for (size_t j = 0; j < N_PERIODS; ++j) (*labels[i])[j] = (*labels[i])[j] % 2 == 0 ? 1 : 0;
  }
  MODAO madao(features, labels);
  const double STRENGTH = (1. / N_SAMPLES) + 1e-10;
  DAO dao(madao, N_ITER, N_SAMPLES, N_THREADS); PROX prox(STRENGTH); auto start = NOW;
  dao.step = MODEL::lip_max(madao);
  statick::svrg::dense::solve<MODEL>(dao, madao, prox);
  std::cout << (NOW - start) / 1e3 << std::endl;
  kul::File tick_interop("sccs.cereal");
  if(tick_interop) statick::sccs::load_from<T>(tick_interop.real());
  return 0;
}
