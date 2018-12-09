#ifndef TICK_SOLVER_SAGA_HPP_
#define TICK_SOLVER_SAGA_HPP_
namespace tick {
namespace saga {
template <typename T = double>
class DAO {
 public:
  DAO(size_t n_samples, size_t n_features) : gradients_average(n_features), gradients_memory(n_samples) {}
  DAO() {}
  T step{0};
  std::vector<T> gradients_average, gradients_memory;
};

namespace dense {
template <typename MODEL, bool INTERCEPT = false, typename T, typename PROX, typename NEXT_I,
          typename SAGA_DAO = saga::DAO<T>>
auto solve(typename MODEL::DAO &modao, T *iterate, PROX call, NEXT_I _next_i,
           std::shared_ptr<SAGA_DAO> p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  if (p_dao == nullptr) p_dao = std::make_shared<SAGA_DAO>(n_samples, n_features);
  auto &dao = *p_dao.get();
  auto *features = modao.features().data();
  T n_samples_inverse = ((double)1 / (double)n_samples);
  double step = 0.00257480411965l;
  for (size_t t = 0; t < n_samples; ++t) {
    INDICE_TYPE i = _next_i();
    T grad_i_factor = MODEL::grad_i_factor(modao, iterate, i);
    T grad_i_factor_old = dao.gradients_memory[i];
    dao.gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    const T *const x_i = &features[n_features * i];
    for (size_t j = 0; j < n_features; ++j) {
      T grad_avg_j = dao.gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_i[j] + grad_avg_j);
      dao.gradients_average[j] += grad_factor_diff * x_i[j] * n_samples_inverse;
    }
    if constexpr (INTERCEPT) {
      iterate[n_features] -= step * (grad_factor_diff + dao.gradients_average[n_features]);
      dao.gradients_average[n_features] += grad_factor_diff * n_samples_inverse;
    }
    call(iterate, step, iterate, n_features);
  }
  return dao;
}
}  // namespace dense
namespace sparse {
template <typename MODEL, bool INTERCEPT = false, typename T, typename PROX, typename NEXT_I,
          typename SAGA_DAO = saga::DAO<T>>
void solve(typename MODEL::DAO &modao, T *iterate, T *steps_correction, PROX call_single, NEXT_I _next_i,
           std::shared_ptr<SAGA_DAO> p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  if (p_dao == nullptr) p_dao = std::make_shared<SAGA_DAO>(n_samples, n_features);
  auto &dao = *p_dao.get();
  T n_samples_inverse = ((double)1 / (double)n_samples);
  double step = 0.00257480411965l;
  auto &features = modao.features();
  for (size_t t = 0; t < n_samples; ++t) {
    INDICE_TYPE i = _next_i();
    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDICE_TYPE *x_i_indices = features.row_indices(i);
    T grad_i_factor = MODEL::grad_i_factor(modao, iterate, i);
    T grad_i_factor_old = dao.gradients_memory[i];
    dao.gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (size_t idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      const INDICE_TYPE &j = x_i_indices[idx_nnz];
      iterate[j] -= step * (grad_factor_diff * x_i[idx_nnz] + steps_correction[j] * dao.gradients_average[j]);
      dao.gradients_average[j] += grad_factor_diff * x_i[idx_nnz] * n_samples_inverse;
      call_single(j, iterate, step * steps_correction[j], iterate);
    }
    if constexpr (INTERCEPT) {
      iterate[n_features] -= step * (grad_factor_diff + dao.gradients_average[n_features]);
      dao.gradients_average[n_features] += grad_factor_diff * n_samples_inverse;
      call_single(n_features, iterate, step, iterate);
    }
  }
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_columns_sparsity(const Sparse2D &features) {
  std::vector<T> column_sparsity(features.cols());
  std::fill(column_sparsity.begin(), column_sparsity.end(), 0);
  double samples_inverse = 1. / features.rows();
  for (size_t i = 0; i < features.rows(); ++i)
    for (size_t j = 0; j < features.row_size(i); ++j) column_sparsity[features.row_indices(i)[j]] += 1;
  for (size_t i = 0; i < features.cols(); ++i) column_sparsity[i] *= samples_inverse;
  return column_sparsity;
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_step_corrections(const Sparse2D &features) {
  std::vector<T> steps_correction(features.cols()), columns_sparsity(compute_columns_sparsity(features));
  for (size_t j = 0; j < features.cols(); ++j) steps_correction[j] = 1. / columns_sparsity[j];
  return steps_correction;
}
}  // namespace sparse
}  // namespace saga
}  // namespace tick
#endif  // TICK_SOLVER_SAGA_HPP_
