#ifndef TICK_SOLVER_SVRG_HPP_
#define TICK_SOLVER_SVRG_HPP_

#include <thread>

namespace tick {
namespace svrg {
template <typename T = double>
class DAO {
 public:
  size_t rand_index = 0;
  T step = 0.00257480411965;
  std::vector<T> full_gradient, fixed_w, grad_i, grad_i_fixed_w, next_iterate;
};
namespace VarianceReductionMethod {
constexpr uint16_t Last = 1, Average = 2, Random = 3;
}
namespace StepType {
constexpr uint16_t Fixed = 1, BarzilaiBorwein = 2;
}
template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename T, typename NEXT_I,
          typename SVRG_DAO = svrg::DAO<T>>
void prepare_solve(SVRG_DAO &dao, typename MODEL::DAO &modao, T *iterate, size_t &t,
                   NEXT_I fn_next_i) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  std::vector<T> previous_iterate, previous_full_gradient;
  size_t iterate_size = n_features + static_cast<uint>(INTERCEPT);
  if
    constexpr(ST == StepType::BarzilaiBorwein) {
      if (t > 1) {
        previous_iterate = dao.fixed_w;
        previous_full_gradient = dao.full_gradient;
      }
    }
  dao.next_iterate = std::vector<T>(iterate_size, 0);
  copy(iterate, dao.next_iterate.data(), iterate_size);
  dao.fixed_w = dao.next_iterate;
  dao.full_gradient = std::vector<T>(iterate_size, 0);
  MODEL::grad(modao, dao.fixed_w.data(), dao.full_gradient.data(), dao.full_gradient.size());
  if
    constexpr(ST == StepType::BarzilaiBorwein) {
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
  if
    constexpr(RM == VarianceReductionMethod::Random || RM == VarianceReductionMethod::Average) {
      std::fill(dao.next_iterate.begin(), dao.next_iterate.end(), 0);
    }
  dao.rand_index = 0;
  if
    constexpr(RM == VarianceReductionMethod::Random) dao.rand_index = fn_next_i();
}

namespace dense {
template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename T, typename PROX,
          typename NEXT_I, typename SVRG_DAO = svrg::DAO<T>>
void solve_thread(SVRG_DAO &dao, typename MODEL::DAO &modao, PROX call, T *iterate,
                  NEXT_I fn_next_i, size_t &t, size_t n_threads, size_t epoch_size) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  size_t iterate_size = n_features + static_cast<uint>(INTERCEPT);
  T epoch_size_inverse = 1.0 / epoch_size;
  for (size_t k = 0; k < (epoch_size / n_threads); k++) {
    const size_t i = fn_next_i();
    MODEL::grad_i(modao, iterate, dao.grad_i.data(), iterate_size, i);
    MODEL::grad_i(modao, dao.fixed_w.data(), dao.grad_i_fixed_w.data(), iterate_size, i);
    for (size_t j = 0; j < iterate_size; ++j)
      iterate[j] =
          iterate[j] - dao.step * (dao.grad_i[j] - dao.grad_i_fixed_w[j] + dao.full_gradient[j]);
    call(iterate, dao.step, iterate, iterate_size);
    if
      constexpr(RM == VarianceReductionMethod::Random) {
        copy(iterate, dao.next_iterate.data(), dao.next_iterate.size());
      }
    if
      constexpr(RM == VarianceReductionMethod::Average) {
        mult_incr(dao.next_iterate.data(), iterate, n_features, epoch_size_inverse);
      }
  }
}

template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename PROX, typename T,
          typename NEXT_I, typename SVRG_DAO = svrg::DAO<T>>
auto solve(typename MODEL::DAO &modao, PROX call, T *iterate, NEXT_I fn_next_i, size_t &t,
           size_t n_threads, size_t epoch_size, std::shared_ptr<SVRG_DAO> p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  double step = 0.00257480411965l;
  const size_t iterate_size = n_features + static_cast<uint>(INTERCEPT);
  if (p_dao == nullptr) p_dao = std::make_shared<SVRG_DAO>();
  auto &dao = *p_dao.get();
  tick::svrg::prepare_solve<MODEL>(dao, modao, iterate, t, fn_next_i);

  if (n_threads > 1) {
    std::vector<std::thread> threadsV;
    for (size_t i = 0; i < n_threads; i++)
      threadsV.emplace_back([&]() {
        solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, iterate, fn_next_i, t, n_threads,
                                               epoch_size);
      });
    for (size_t i = 0; i < n_threads; i++) threadsV[i].join();
  } else {
    solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, iterate, fn_next_i, t, n_threads,
                                           epoch_size);
  }
  if
    constexpr(RM == VarianceReductionMethod::Last) {
      for (size_t i = 0; i < iterate_size; i++) dao.next_iterate[i] = iterate[i];
    }
  t += epoch_size;
  return p_dao;
}
}  // namespace dense

namespace sparse {

template <uint16_t RM = VarianceReductionMethod::Last, uint16_t ST = StepType::Fixed,
          bool INTERCEPT = false, typename MODEL, typename PROX, typename T, typename Sparse2D,
          typename DAO, typename NEXT_I>
void solve_thread(const Sparse2D &features, const T *labels, T *iterate, T *steps_correction,
                  DAO &&dao, NEXT_I fn_next_i, size_t &t, size_t n_threads, size_t epoch_size) {
  for (size_t k = 0; k < (epoch_size / n_threads); k++) {
    const size_t i = fn_next_i();

    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDICE_TYPE *x_i_indices = features.row_indices(i);

    // // Gradients factors (model is a GLM)
    // // TODO: a grad_i_factor(i, array1, array2) to loop once on the features
    T grad_i_diff = MODEL::grad_i_factor(i, iterate) - MODEL::grad_i_factor(i, dao.fixed_w.data());

    // T grad_i_diff =
    //     model->grad_i_factor(i, iterate) - model->grad_i_factor(i, fixed_w);
    // // We update the iterate within the support of the features vector, with the
    // // probabilistic correction
    // for (size_t idx_nnz = 0; idx_nnz < x_i.size_sparse(); ++idx_nnz) {
    //   // Get the index of the idx-th sparse feature of x_i
    //   size_t j = x_i.indices()[idx_nnz];
    //   T full_gradient_j = full_gradient[j];
    //   // Step-size correction for coordinate j
    //   T step_correction = steps_correction[j];
    //   // Gradient descent with probabilistic step-size correction
    //   T descent_direction = step * (x_i.data()[idx_nnz] * grad_i_diff +
    //       step_correction * full_gradient_j);
    //   if (casted_prox->is_in_range(j))
    //     iterate[j] = casted_prox->call_single_with_index(
    //         iterate[j] - descent_direction, step * step_correction, j);
    //   else
    //     iterate[j] -= descent_direction;
    // }
    // // And let's not forget to update the intercept as well. It's updated at each
    // // step, so no step-correction. Note that we call the prox, in order to be
    // // consistent with the dense case (in the case where the user has the weird
    // // desire to to regularize the intercept)
    // if (use_intercept) {
    //   T descent_direction = step * (grad_i_diff + full_gradient[n_features]);

    //   if (casted_prox->is_in_range(n_features))
    //     iterate[n_features] = casted_prox->call_single_with_index(
    //         iterate[n_features] - descent_direction, step, n_features);
    //   else
    //     iterate[n_features] -= descent_direction;
    // }
    // // Note that the average option for variance reduction with sparse data is a
    // // very bad idea, but this is caught in the python class

    // if constexpr (RM == VarianceReductionMethod::Random) if(t == dao.rand_index) // next_iterate
    // = iterate;
    // if constexpr (RM == VarianceReductionMethod::Average) // dao.next_iterate.mult_incr(iterate,
    // 1.0 / epoch_size);
  }
}

template <uint16_t RM = VarianceReductionMethod::Last, uint16_t ST = StepType::Fixed,
          bool INTERCEPT = false, typename MODEL, typename PROX, typename T, typename Sparse2D,
          typename DAO, typename NEXT_I>
void solve(const Sparse2D &features, const T *labels, T *iterate, T *steps_correction, DAO &dao,
           NEXT_I fn_next_i, size_t &t, size_t n_threads, size_t epoch_size) {
  double step = 0.00257480411965l;
  const size_t n_samples = features.rows(), n_features = features.cols();
  const size_t iterate_size = n_features + static_cast<uint>(INTERCEPT);
  if (n_threads > 1) {
    std::vector<std::thread> threadsV;
    for (size_t i = 0; i < n_threads; i++)
      threadsV.emplace_back([&]() {
        for (size_t t = 0; t < (epoch_size / n_threads); ++t)
          solve_thread<RM, ST, INTERCEPT, MODEL, PROX>(features, labels, iterate, steps_correction,
                                                       dao, fn_next_i, t, n_threads, epoch_size);
      });
    for (size_t i = 0; i < n_threads; i++) threadsV[i].join();
  } else {
    for (size_t i = 0; i < epoch_size; ++i)
      solve_thread<RM, ST, INTERCEPT, MODEL, PROX>(features, labels, iterate, steps_correction, dao,
                                                   fn_next_i, t, n_threads, epoch_size);
  }
  if
    constexpr(RM == VarianceReductionMethod::Last) {
      for (size_t i = 0; i < iterate_size; i++) dao.next_iterate[i] = iterate[i];
    }
  t += epoch_size;
}

template <typename Sparse2D, class T = double>
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
template <typename Sparse2D, class T = double>
std::vector<T> compute_step_corrections(const Sparse2D &features) {
  std::vector<T> steps_correction(features.cols()),
      columns_sparsity(compute_columns_sparsity(features));
  for (size_t j = 0; j < features.cols(); ++j) steps_correction[j] = 1. / columns_sparsity[j];
  return steps_correction;
}
}  // namespace sparse
}  // namespace svrg
}  // namespace tick
#endif  // TICK_SOLVER_SVRG_HPP_
