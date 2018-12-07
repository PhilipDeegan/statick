

#ifndef TICK_SOLVER_ASAGA_HPP_
#define TICK_SOLVER_ASAGA_HPP_

#include "history.hpp"

#include <thread>

namespace tick {
namespace asaga {
namespace sparse {
template <bool INTERCEPT, typename HISTORY, typename T, typename Sparse2D, typename PROX, typename NEXT_I>
void threaded_solve(const Sparse2D &features, const T *labels, std::atomic<T> *gradients_average,
                    std::atomic<T> *gradients_memory, T *iterate, T *steps_correction, PROX call_single, NEXT_I _next_i,
                    size_t n_threads, size_t n_thread, HISTORY &history) {
  double step = 0.00257480411965l;
  const size_t n_samples = features.rows(), n_features = features.cols();
  size_t n_epochs = 1, epoch_size = n_samples;
  T n_samples_inverse = ((double)1 / (double)n_samples);
  T x_ij = 0, step_correction = 0;
  T grad_factor_diff = 0, grad_avg_j = 0, grad_i_factor = 0, grad_i_factor_old = 0;
  ulong idx_nnz = 0, thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);
  auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    for (ulong t = 0; t < thread_epoch_size; ++t) {
      INDICE_TYPE i = _next_i();
      size_t x_i_size = features.row_size(i);
      const T *x_i = features.row_raw(i);
      const INDICE_TYPE *x_i_indices = features.row_indices(i);
      T grad_i_factor = labels[i] * (sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
      T grad_i_factor_old = gradients_memory[i];
      while (!gradients_memory[i].compare_exchange_weak(grad_i_factor_old, grad_i_factor)) {
      }
      grad_factor_diff = grad_i_factor - grad_i_factor_old;
      for (idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
        const INDICE_TYPE &j = x_i_indices[idx_nnz];
        x_ij = x_i[idx_nnz];
        grad_avg_j = gradients_average[j];
        step_correction = steps_correction[j];
        while (!gradients_average[j].compare_exchange_weak(
            grad_avg_j, grad_avg_j + (grad_factor_diff * x_ij * n_samples_inverse))) {
        }
        iterate[j] -= step * (grad_factor_diff * x_ij + steps_correction[j] * grad_avg_j);
        gradients_average[j] = gradients_average[j] + (grad_factor_diff * x_ij * n_samples_inverse);
        call_single(j, iterate, step * steps_correction[j], iterate);
      }
      if constexpr (INTERCEPT) {
        iterate[n_features] -= step * (grad_factor_diff + gradients_average[n_features]);
        T gradients_average_j = gradients_average[n_features];
        while (!gradients_average[n_features].compare_exchange_weak(
            gradients_average_j, gradients_average_j + (grad_factor_diff * n_samples_inverse))) {
        }
        call_single(n_features, iterate, step, iterate);
      }
    }
    if constexpr (std::is_same<HISTORY, tick::solver::History<T>>::value) {
      if (n_thread == 0) {
        history += epoch_size;
        if ((history.last_record_epoch + epoch) == 1 ||
            ((history.last_record_epoch + epoch) % history.record_every == 0)) {
          auto end = std::chrono::steady_clock::now();
          double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                        static_cast<double>(std::chrono::steady_clock::period::den);
          history.save_history(time, epoch, iterate, n_features);
        }
      }
    }
  }
  if constexpr (std::is_same<HISTORY, tick::solver::History<T>>::value) {
    if (n_thread == 0) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                    static_cast<double>(std::chrono::steady_clock::period::den);
      history.last_record_time = time;
      history.last_record_epoch += n_epochs;
    }
  }
}

template <bool INTERCEPT = false, typename T, typename HISTORY = tick::solver::History<T>, typename Sparse2D,
          typename PROX, typename NEXT_I>
void solve(const Sparse2D &features, const T *labels, std::atomic<T> *gradients_average,
           std::atomic<T> *gradients_memory, T *iterate, T *steps_correction, PROX call_single, NEXT_I _next_i,
           size_t n_threads, HISTORY &&history = HISTORY()) {
  std::vector<std::thread> threads;
  for (size_t i = 0; i < n_threads; i++) {
    threads.emplace_back(
        [&](size_t n_thread) {
          threaded_solve<INTERCEPT>(features, labels, gradients_average, gradients_memory, iterate, steps_correction,
                                    call_single, _next_i, n_threads, n_thread, history);
        },
        i);
  }
  for (size_t i = 0; i < n_threads; i++) {
    threads[i].join();
  }
}
}  // namespace sparse
}  // namespace asaga
}  // namespace tick
#endif  // TICK_SOLVER_ASAGA_HPP_
