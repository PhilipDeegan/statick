#ifndef TICK_SOLVER_SGD_HPP_
#define TICK_SOLVER_SGD_HPP_
namespace tick {
namespace sgd {
template <typename T = double>
class DAO {
 public:
  DAO() {}
  T step{0};
};

namespace dense {
template <typename MODEL, bool INTERCEPT = false, typename T, typename PROX, typename NEXT_I,
          typename SGD_DAO = sgd::DAO<T>>
auto solve(typename MODEL::DAO &modao, T *iterate, PROX call, NEXT_I _next_i, size_t &t,
           std::shared_ptr<SGD_DAO> p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features(), start_t = t;
  if (p_dao == nullptr) p_dao = std::make_shared<SGD_DAO>();
  auto &dao = *p_dao.get();
  auto &features = modao.features();
  auto *labels = modao.labels().data();
  constexpr double step = 1e-5;
  std::vector<T> v_grad(n_features, 0);
  T *grad = v_grad.data();
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDICE_TYPE i = _next_i();
    T step_t = step / (t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    for (size_t j = 0; j < n_features; ++j) grad[j] = x_i[j] * grad_i_factor;
    for (size_t j = 0; j < n_features; j++) iterate[j] += grad[j] * -step_t;
    call(iterate, step_t, iterate, n_features);
  }
  return p_dao;
}
}  // namespace dense
namespace sparse {
template <typename MODEL, bool INTERCEPT = false, typename T, typename PROX, typename NEXT_I,
          typename SGD_DAO = sgd::DAO<T>>
auto solve(typename MODEL::DAO &modao, T *iterate, PROX call, NEXT_I _next_i, size_t &t,
           std::shared_ptr<SGD_DAO> p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features(), start_t = t;
  if (p_dao == nullptr) p_dao = std::make_shared<SGD_DAO>();
  auto &dao = *p_dao.get();
  double step = 1e-5;
  auto &features = modao.features();
  auto *labels = modao.labels().data();
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDICE_TYPE i = _next_i();
    T step_t = step / (t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T delta = -step_t * grad_i_factor;
    const INDICE_TYPE *x_indices = features.row_indices(i);
    const size_t row_size = features.row_size(i);
    for (size_t j = 0; j < row_size; j++) iterate[x_indices[j]] += x_i[j] * delta;
    if
      constexpr(INTERCEPT) { iterate[n_features] += delta; }
    call(iterate, step_t, iterate, n_features);
  }
  return p_dao;
}
}  // namespace sparse
}  // namespace sgd
}  // namespace tick
#endif  // TICK_SOLVER_SGD_HPP_
