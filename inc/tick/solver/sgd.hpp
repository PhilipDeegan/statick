

#ifndef TICK_SOLVER_SGD_HPP_
#define TICK_SOLVER_SGD_HPP_

namespace tick {

namespace sgd {
using INDEX_TYPE = INDICE_TYPE;

namespace dense {
template <bool INTERCEPT = false, typename T, typename FEATURES, typename PROX, typename NEXT_I>
void solve(const FEATURES &features, const T *labels, T *iterate, PROX call, NEXT_I _next_i,
           size_t &t) {
  constexpr double step = 1e-5;
  size_t n_samples = features.rows(), n_features = features.cols(), start_t = t;
  std::vector<T> v_grad(n_features, 0);
  T *grad = v_grad.data();
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDEX_TYPE i = _next_i();
    T step_t = step / (t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    for (size_t j = 0; j < n_features; ++j) grad[j] = x_i[j] * grad_i_factor;
    for (size_t j = 0; j < n_features; j++) iterate[j] += grad[j] * -step_t;
    call(iterate, step_t, iterate, n_features);
  }
}
}  // namespace dense

namespace sparse {
template <bool INTERCEPT = false, typename T, typename Sparse2D, typename PROX, typename NEXT_I>
void solve(const Sparse2D &features, const T *labels, T *iterate, PROX call, NEXT_I _next_i, size_t &t) {
  size_t n_samples = features.rows(), n_features = features.cols(), start_t = t;
  double step = 1e-5;
  for (t = start_t; t < start_t + n_samples; ++t) {
    INDEX_TYPE i = _next_i();
    T step_t = step / (t + 1);
    const T *x_i = features.row_raw(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T delta = -step_t * grad_i_factor;
    const INDEX_TYPE *x_indices = features.row_indices(i);
    const size_t row_size = features.row_size(i);
    if constexpr (INTERCEPT) {
      // Get the features vector, which is sparse here
      for (size_t j = 0; j < row_size; j++) iterate[x_indices[j]] += x_i[j] * delta;
      iterate[n_features] += delta;
    } else {
      for (size_t j = 0; j < row_size; j++) iterate[x_indices[j]] += x_i[j] * delta;
    }
    call(iterate, step_t, iterate, n_features);
  }
}
}  // namespace sparse
}  // namespace sgd
}  // namespace tick

#endif  // TICK_SOLVER_SGD_HPP_