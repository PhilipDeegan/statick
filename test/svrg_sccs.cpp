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
  using FEATURES = statick::Sparse2D<T>;
  using LABELS   = statick::Array<int32_t>;
  using PROX     = statick::ProxL2Sq<T>;
  using MODAO    = statick::sccs::DAO<typename FEATURES::value_type, FEATURES>;
  using MODEL    = statick::TModelSCCS<MODAO>;
  using DAO      = statick::svrg::dense::DAO<MODAO, HISTORY>;
  kul::File tick_interop("sccs.cereal");
  std::shared_ptr<MODAO> pmadao = statick::sccs::load_from<MODAO>(tick_interop.real());
  auto &madao = *pmadao.get();
  const size_t n_samples = madao.n_samples();
#include "random_seq.ipp"
  const double STRENGTH = (1. / N_SAMPLES) + 1e-10;
  DAO dao(madao, N_ITER, N_SAMPLES, N_THREADS); PROX prox(STRENGTH); auto start = NOW;
  MODEL::compute_lip_consts(madao, 8);
  dao.step = MODEL::lip_max(madao);
  KLOG(INF) << dao.step;
  return 0;
}
