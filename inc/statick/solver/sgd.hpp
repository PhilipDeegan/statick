#ifndef STATICK_SOLVER_SGD_HPP_
#define STATICK_SOLVER_SGD_HPP_
#include "statick/random.hpp"
namespace statick {
namespace sgd {
template <typename MODAO, bool INTERCEPT = false>
class DAO {
 public:
  using T = typename MODAO::value_type;
  DAO(MODAO &modao) :
    rand(0, modao.n_samples() - 1),
    iterate(modao.n_features() + static_cast<size_t>(INTERCEPT)) {}

  T step = 1e-5;
  size_t t = 0;
  std::vector<T> iterate;
  RandomMinMax<INDICE_TYPE> rand;
};

namespace dense {
template <typename MODEL, bool INTERCEPT = false, typename PROX,
          typename T = typename MODEL::value_type,
          typename DAO = sgd::DAO<T>>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox) {
  auto &t = dao.t;
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features(), start_t = t;
  auto &features = modao.features();
  auto *labels = modao.labels().data();
  auto *iterate = dao.iterate.data();
  const T step = dao.step;
  std::vector<T> v_grad(n_features, 0);
  T *grad = v_grad.data();
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDICE_TYPE i = dao.rand.next();
    T step_t = step / (dao.t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    for (size_t j = 0; j < n_features; ++j) grad[j] = x_i[j] * grad_i_factor;
    for (size_t j = 0; j < n_features; j++) iterate[j] += grad[j] * -step_t;
    PROX::call(prox, iterate, step_t, iterate, n_features);
  }
}
}  // namespace dense
namespace sparse {
template <typename MODEL, bool INTERCEPT = false, typename PROX,
          typename T = typename MODEL::value_type,
          typename DAO = sgd::DAO<T>>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox) {
  auto &t = dao.t;
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features(), start_t = t;
  const T step = dao.step;
  auto &features = modao.features();
  auto *labels = modao.labels().data();
  auto *iterate = dao.iterate.data();
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDICE_TYPE i = dao.rand.next();
    T step_t = step / (t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T delta = -step_t * grad_i_factor;
    const INDICE_TYPE *x_indices = features.row_indices(i);
    const size_t row_size = features.row_size(i);
    for (size_t j = 0; j < row_size; j++) iterate[x_indices[j]] += x_i[j] * delta;
    if constexpr(INTERCEPT) { iterate[n_features] += delta; }
    PROX::call(prox, iterate, step_t, iterate, n_features);
  }
}
}  // namespace sparse
}  // namespace sgd
namespace solver {
template <typename MODEL, bool INTERCEPT = false>
class SGD {
 public:
  using DAO = typename statick::sgd::DAO<typename MODEL::DAO, INTERCEPT>;

  template <typename PROX>
  static inline void SOLVE(DAO &dao, typename MODEL::DAO &modao, PROX &prox) {
    if constexpr (MODEL::DAO::FEATURE::is_sparse)
      statick::sgd::sparse::solve<MODEL>(dao, modao, prox);
    else
      statick::sgd::dense::solve<MODEL>(dao, modao, prox);
  }
};
}
}  // namespace statick
#endif  // STATICK_SOLVER_SGD_HPP_
