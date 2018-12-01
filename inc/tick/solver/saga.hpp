


#ifndef TICK_SOLVER_SAGA_HPP_
#define TICK_SOLVER_SAGA_HPP_


#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE ulong
#else
#define INDICE_TYPE std::uint32_t
#endif

namespace tick {


namespace saga {
using INDEX_TYPE = INDICE_TYPE;
namespace dense {
template <typename T, typename FEATURES, typename PROX, typename NEXT_I>
void solve(const FEATURES &features, const T * const labels, T *gradients_average, T *gradients_memory,
           T *iterate, PROX call, NEXT_I _next_i) {
  size_t N_SAMPLES = features.rows(), N_FEATURES = features.cols();
  T N_SAMPLES_inverse = ((double)1 / (double)N_SAMPLES);
  double step = 0.00257480411965l;
  ulong n_features = N_FEATURES;
  for (ulong t = 0; t < N_SAMPLES; ++t) {
    INDEX_TYPE i = _next_i();
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    const T * const x_i = &features[N_FEATURES * i];
    for (ulong j = 0; j < n_features; ++j) {
      T grad_avg_j = gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_i[j] + grad_avg_j);
      gradients_average[j] += grad_factor_diff * x_i[j] * N_SAMPLES_inverse;
    }
    call(iterate, step, iterate, n_features);
  }
}
}  // namespace dense

namespace sparse {
template <typename T, typename Sparse2D, typename PROX, typename NEXT_I>
void solve(const Sparse2D &features, const T * labels, T *gradients_average, T *gradients_memory,
           T *iterate, T *steps_correction, PROX call_single, NEXT_I _next_i) {
  size_t n_samples = features.rows();
  T n_samples_inverse = ((double)1 / (double)n_samples);
  double step = 0.00257480411965l;
  for (ulong t = 0; t < n_samples; ++t) {
    INDEX_TYPE i = _next_i();
    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDEX_TYPE *x_i_indices = features.row_indices(i);
    T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      const INDEX_TYPE &j = x_i_indices[idx_nnz];
      iterate[j] -=
          step * (grad_factor_diff * x_i[idx_nnz] + steps_correction[j] * gradients_average[j]);
      gradients_average[j] += grad_factor_diff * x_i[idx_nnz] * n_samples_inverse;
      call_single(j, iterate, step * steps_correction[j], iterate);
    }
  }
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_columns_sparsity(const Sparse2D &features) {
  std::vector<T> column_sparsity(features.cols());
  std::fill(column_sparsity.begin(), column_sparsity.end(), 0);
  double samples_inverse = 1. / features.rows();
  for (ulong i = 0; i < features.rows(); ++i)
    for (ulong j = 0; j < features.row_size(i); ++j)
      column_sparsity[features.row_indices(i)[j]] += 1;
  for (uint64_t i = 0; i < features.cols(); ++i) column_sparsity[i] *= samples_inverse;
  return column_sparsity;
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_step_corrections(const Sparse2D &features) {
  std::vector<T> steps_correction(features.cols()),
      columns_sparsity(compute_columns_sparsity(features));
  for (ulong j = 0; j < features.cols(); ++j) steps_correction[j] = 1. / columns_sparsity[j];
  return steps_correction;
}

}  // namespace sparse
}  // namespace saga

}  // namespace tick

#endif  // TICK_SOLVER_SAGA_HPP_