#ifndef STATICK_SOLVER_SAGA_HPP_
#define STATICK_SOLVER_SAGA_HPP_

#include "statick/random.hpp"

namespace statick {
namespace saga {

template <typename _MODEL, bool INTERCEPT>
class DAO {
 public:
  using MODEL = _MODEL;
  using MODAO = typename _MODEL::DAO;
  using T = typename MODAO::value_type;
  using value_type = T;

  DAO(MODAO &modao)
    : iterate(modao.n_features() + static_cast<size_t>(INTERCEPT)),
      gradients_average(modao.n_features()), gradients_memory(modao.n_samples()),
      rand(0, modao.n_samples() - 1) { }

  std::vector<T> iterate, gradients_average, gradients_memory;
  RandomMinMax<INDICE_TYPE> rand;
};

namespace dense {
template <typename _MODEL, bool _INTERCEPT = false>
class DAO : public statick::saga::DAO<_MODEL, _INTERCEPT> {
 public:
  using SUPER = statick::saga::DAO<_MODEL, _INTERCEPT>;
  using T = typename SUPER::value_type;
  using MODEL = _MODEL;
  using MODAO = typename MODEL::DAO;
  using value_type = T;
  static constexpr bool INTERCEPT = _INTERCEPT;
  using SUPER::iterate;
  using SUPER::gradients_average;
  using SUPER::gradients_memory;

  DAO(MODAO &modao) : statick::saga::DAO<_MODEL, INTERCEPT>(modao) {}

  T step = 0.00257480411965l;
};

template <typename MODEL, bool INTERCEPT = false, typename PROX,
          typename T = typename MODEL::value_type,
          typename DAO = statick::saga::dense::DAO<T>>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  auto *features = modao.features().data();
  T n_samples_inverse = ((double)1 / (double)n_samples), step = dao.step;
  auto *iterate = dao.iterate.data();
  for (size_t t = 0; t < n_samples; ++t) {
    INDICE_TYPE i = dao.rand.next();
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
    if constexpr(INTERCEPT) {
      iterate[n_features] -= step * (grad_factor_diff + dao.gradients_average[n_features]);
      dao.gradients_average[n_features] += grad_factor_diff * n_samples_inverse;
    }
    PROX::call(prox, iterate, step, iterate, n_features);
  }
}
}  // namespace dense

namespace sparse {
template <typename T, typename Sparse2D>
std::vector<T> compute_columns_sparsity(const Sparse2D &features) {
  std::vector<T> column_sparsity(features.cols());
  std::fill(column_sparsity.begin(), column_sparsity.end(), 0);
  double samples_inverse = 1. / features.rows();
  for (size_t i = 0; i < features.rows(); ++i)
    for (size_t j = 0; j < features.row_size(i); ++j)
      column_sparsity[features.row_indices(i)[j]] += 1;
  for (size_t i = 0; i < features.cols(); ++i) column_sparsity[i] *= samples_inverse;
  return column_sparsity;
}
template <typename Sparse2D, typename T = typename Sparse2D::value_type>
std::vector<T> compute_step_corrections(const Sparse2D &features) {
  std::vector<T> steps_corrections(features.cols()),
      columns_sparsity(compute_columns_sparsity<T>(features));
  for (size_t j = 0; j < features.cols(); ++j) steps_corrections[j] = 1. / columns_sparsity[j];
  return steps_corrections;
}

template <typename _MODEL, bool _INTERCEPT = false>
class DAO : public statick::saga::DAO<_MODEL, _INTERCEPT> {
 public:
  using SUPER = statick::saga::DAO<_MODEL, _INTERCEPT>;
  using T = typename SUPER::value_type;
  using MODEL = _MODEL;
  using MODAO = typename MODEL::DAO;
  using value_type = T;
  static constexpr bool INTERCEPT = _INTERCEPT;
  using SUPER::iterate;
  using SUPER::gradients_average;
  using SUPER::gradients_memory;

  DAO(MODAO &modao)
     : statick::saga::DAO<_MODEL, INTERCEPT>(modao),
       steps_corrections(statick::saga::sparse::compute_step_corrections(modao.features())) {}

  T step = 0.00257480411965l;
  std::vector<T> steps_corrections;
};

template <typename MODEL, bool INTERCEPT = false, typename PROX,
          typename T = typename MODEL::value_type,
          typename DAO = statick::saga::sparse::DAO<T>>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  (void) n_features;  // possibly unused if constexpr
  T n_samples_inverse = ((double)1 / (double)n_samples), step = dao.step;
  auto * iterate = dao.iterate.data();
  auto * steps_corrections = dao.steps_corrections.data();
  auto &features = modao.features();
  for (size_t t = 0; t < n_samples; ++t) {
    INDICE_TYPE i = dao.rand.next();
    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDICE_TYPE *x_i_indices = features.row_indices(i);
    T grad_i_factor = MODEL::grad_i_factor(modao, iterate, i);
    T grad_i_factor_old = dao.gradients_memory[i];
    dao.gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (size_t idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      const INDICE_TYPE &j = x_i_indices[idx_nnz];
      iterate[j] -=
          step * (grad_factor_diff * x_i[idx_nnz] + steps_corrections[j] * dao.gradients_average[j]);
      dao.gradients_average[j] += grad_factor_diff * x_i[idx_nnz] * n_samples_inverse;
      iterate[j] = PROX::call_single(prox, iterate[j], step * steps_corrections[j]);
    }
    if constexpr(INTERCEPT) {
      iterate[n_features] -= step * (grad_factor_diff + dao.gradients_average[n_features]);
      dao.gradients_average[n_features] += grad_factor_diff * n_samples_inverse;
      iterate[n_features] = PROX::call_single(prox, iterate[n_features], step);
    }
  }
}

}  // namespace sparse
}  // namespace saga
namespace solver {

class SAGA {
 public:
  static constexpr std::string_view NAME = "saga";
  template <typename M, bool I>
  using sparse_dao = statick::saga::sparse::DAO<M, I>;

  template <typename M, bool I>
  using  dense_dao = statick::saga:: dense::DAO<M, I>;

  template <typename M, bool I = false>
  using DAO = typename std::conditional<M::DAO::FEATURE::is_sparse, sparse_dao<M, I>, dense_dao<M, I>>::type;

  template <typename _DAO, typename PROX>
  static inline void SOLVE(_DAO &dao, typename _DAO::MODAO &modao, PROX &prox) {
    using M = typename _DAO::MODEL;
    if constexpr (std::is_same<_DAO, sparse_dao<M, _DAO::INTERCEPT>>::value)
      statick::saga::sparse::solve<M>(dao, modao, prox);
    else
      statick::saga:: dense::solve<M>(dao, modao, prox);
  }
};
}  // namespace solver
}  // namespace statick
#endif  // STATICK_SOLVER_SAGA_HPP_
