#ifndef STATICK_SOLVER_SVRG_HPP_
#define STATICK_SOLVER_SVRG_HPP_

#include <thread>
#include "statick/random.hpp"
#include "statick/solver/history.hpp"

namespace statick {
namespace svrg {
namespace VarianceReductionMethod {
constexpr uint16_t Last = 1, Average = 2, Random = 3;
}
namespace StepType {
constexpr uint16_t Fixed = 1, BarzilaiBorwein = 2;
}
template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false,
          typename T = typename MODEL::value_type, typename DAO>
void prepare_solve(DAO &dao, typename MODEL::DAO &modao, size_t &t) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  std::vector<T> previous_iterate, previous_full_gradient;
  size_t iterate_size = n_features + static_cast<size_t>(INTERCEPT);
  if constexpr (ST == StepType::BarzilaiBorwein) {
    if (t > 1) {
      previous_iterate = dao.fixed_w;
      previous_full_gradient = dao.full_gradient;
    }
  }
  dao.next_iterate = std::vector<T>(iterate_size, 0);
  copy(dao.iterate.data(), dao.next_iterate.data(), iterate_size);
  dao.fixed_w = dao.next_iterate;
  dao.full_gradient = std::vector<T>(iterate_size, 0);
  MODEL::template grad<INTERCEPT>(modao, dao.fixed_w.data(), dao.full_gradient.data(),
                                  iterate_size);
  if constexpr (ST == StepType::BarzilaiBorwein) {
    if (t > 1) {
      std::vector<T> iterate_diff = dao.next_iterate;
      mult_incr(iterate_diff.data(), previous_iterate.data(), -1, iterate_size);
      std::vector<T> full_gradient_diff = dao.full_gradient;
      mult_incr(full_gradient_diff.data(), previous_full_gradient.data(), -1, iterate_size);
      dao.step = 1. / n_samples * norm_sq(iterate_diff.data(), iterate_size) /
                 dot(iterate_diff.data(), full_gradient_diff.data(), iterate_size);
    }
  }
  dao.grad_i = std::vector<T>(iterate_size);
  dao.grad_i_fixed_w = std::vector<T>(iterate_size);
  if constexpr (RM == VarianceReductionMethod::Random || RM == VarianceReductionMethod::Average)
    std::fill(dao.next_iterate.begin(), dao.next_iterate.end(), 0);
  dao.rand_index = 0;
  if constexpr (RM == VarianceReductionMethod::Random) dao.rand_index = dao.rand.next();
}
}  // namespace svrg
}  // namespace statick

#include "statick/solver/svrg/dense.hpp"
#include "statick/solver/svrg/sparse.hpp"

namespace statick {
class SVRG {
 public:
  static constexpr std::string_view NAME = "svrg";
  template <typename M, typename H = statick::solver::NoHistory, bool I = false>
  using DAO =
      typename std::conditional<M::DAO::FEATURE::is_sparse, statick::svrg::sparse::DAO<M, H, I>,
                                statick::svrg::dense::DAO<M, H, I>>::type;

  template <typename _DAO, typename PROX>
  static inline void SOLVE(_DAO &dao, typename _DAO::MODAO &modao, PROX &prox) {
    using M = typename _DAO::MODEL;
    if constexpr (M::DAO::FEATURE::is_sparse)
      statick::svrg::sparse::solve<M>(dao, modao, prox);
    else
      statick::svrg::dense::solve<M>(dao, modao, prox);
  }
};
}  // namespace statick

#endif  // STATICK_SOLVER_SVRG_HPP_
