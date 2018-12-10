#ifndef TICK_SOLVER_ASAGA_HPP_
#define TICK_SOLVER_ASAGA_HPP_

#include "history.hpp"
#include <thread>
namespace tick {
namespace asaga {
template <typename T = double>
class DAO {
 public:
  DAO(size_t n_samples, size_t n_features)
      : gradients_average(n_features), gradients_memory(n_samples) {}
  DAO() {}
  T step{0};
  std::vector<std::atomic<T>> gradients_average, gradients_memory;
};
namespace sparse {

constexpr size_t N_ITER = 10;
template <typename MODEL, bool INTERCEPT, typename HISTORY, typename T, typename PROX,
          typename NEXT_I, typename ASAGA_DAO>
void threaded_solve(typename MODEL::DAO &modao, T *iterate, T *steps_correction, PROX call_single,
                    NEXT_I _next_i, size_t n_threads, size_t n_thread, HISTORY &history,
                    ASAGA_DAO &dao) {
  KLOG(INF) << n_threads;
  KLOG(INF) << n_thread;
  double step = 0.00257480411965l;
  auto &features = modao.features();
  auto *labels = modao.labels().data();
  auto *gradients_memory = dao.gradients_memory.data();
  auto *gradients_average = dao.gradients_average.data();
  const size_t n_samples = features.rows(), n_features = features.cols();
  constexpr size_t n_epochs = N_ITER;
  size_t epoch_size = n_samples;
  T n_samples_inverse = ((double)1 / (double)n_samples);
  T x_ij = 0, step_correction = 0;
  T grad_factor_diff = 0, grad_avg_j = 0, grad_i_factor = 0, grad_i_factor_old = 0;

  size_t idx_nnz = 0, thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);

  KLOG(INF) << n_samples;
  KLOG(INF) << (n_epochs + 1);
  KLOG(INF) << thread_epoch_size;

  auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    for (size_t t = 0; t < thread_epoch_size; ++t) {
      INDICE_TYPE i = _next_i();
      size_t x_i_size = features.row_size(i);
      const T *x_i = features.row_raw(i);
      const INDICE_TYPE *x_i_indices = features.row_indices(i);
      T grad_i_factor = MODEL::grad_i_factor(modao, iterate, i);
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
      if
        constexpr(INTERCEPT) {
          iterate[n_features] -= step * (grad_factor_diff + gradients_average[n_features]);
          T gradients_average_j = gradients_average[n_features];
          while (!gradients_average[n_features].compare_exchange_weak(
              gradients_average_j, gradients_average_j + (grad_factor_diff * n_samples_inverse))) {
          }
          call_single(n_features, iterate, step, iterate);
        }
    }
    if
      constexpr(std::is_same<HISTORY, tick::solver::History<T>>::value) {
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
  if
    constexpr(std::is_same<HISTORY, tick::solver::History<T>>::value) {
      if (n_thread == 0) {
        auto end = std::chrono::steady_clock::now();
        double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                      static_cast<double>(std::chrono::steady_clock::period::den);
        history.last_record_time = time;
        history.last_record_epoch += n_epochs;
      }
    }

}

template <typename MODEL, bool INTERCEPT = false, typename T,
          typename HISTORY = tick::solver::History<T>, typename PROX, typename NEXT_I,
          typename ASAGA_DAO = asaga::DAO<T>>
std::shared_ptr<ASAGA_DAO> solve(typename MODEL::DAO &modao, T *iterate, T *steps_correction, PROX call_single,
           NEXT_I _next_i, size_t n_threads, HISTORY &history,
           std::shared_ptr<ASAGA_DAO> &&p_dao = nullptr) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  if (p_dao == nullptr) p_dao = std::make_shared<ASAGA_DAO>(n_samples, n_features);
  auto &dao = *p_dao.get();
  std::vector<std::thread> threads;
  for (size_t i = 0; i < n_threads; i++) {
    threads.emplace_back(
        [&](size_t n_thread) {
          threaded_solve<MODEL, INTERCEPT>(modao, iterate, steps_correction, call_single, _next_i,
                                           n_threads, n_thread, history, dao);
        },
        i);
  }
  for (size_t i = 0; i < n_threads; i++) threads[i].join();

  return p_dao;
}
}  // namespace sparse
}  // namespace asaga
}  // namespace tick
#endif  // TICK_SOLVER_ASAGA_HPP_
