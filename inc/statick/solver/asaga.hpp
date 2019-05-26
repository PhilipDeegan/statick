#ifndef STATICK_SOLVER_ASAGA_HPP_
#define STATICK_SOLVER_ASAGA_HPP_

#include <thread>
#include "statick/solver/saga.hpp"
#include "statick/solver/history.hpp"
namespace statick {
namespace asaga {

template <typename MODAO, typename HISTOIR = statick::solver::NoHistory,
          bool INTERCEPT = false, typename T = typename MODAO::value_type>
class DAO {
 public:
  using HISTORY = HISTOIR;
  DAO(MODAO &modao, size_t _n_epochs, size_t _epoch_size, size_t _threads)
      : n_epochs(_n_epochs), epoch_size(_epoch_size), n_threads(_threads),
        iterate(modao.n_features() + static_cast<size_t>(INTERCEPT)),
        steps_corrections(statick::saga::sparse::compute_step_corrections(modao.features())),
        gradients_average(modao.n_features()), gradients_memory(modao.n_samples()) {
    for (size_t i = 0; i < modao.n_samples(); i++) gradients_memory[i].store(0);
    for (size_t i = 0; i < modao.n_features(); i++) gradients_average[i].store(0);
  }
  T step = 0.00257480411965l;
  size_t n_epochs = 200, epoch_size = 0, n_threads;
  HISTORY history;
  std::vector<T> iterate, steps_corrections;
  std::vector<std::atomic<T>> gradients_average, gradients_memory;
};
namespace sparse {

template <typename MODEL, bool INTERCEPT, typename PROX, typename NEXT_I, typename DAO>
void threaded_solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox, NEXT_I _next_i, size_t n_thread) {
  using T = typename MODEL::value_type;
  using TOL = typename DAO::HISTORY::TOLERANCE;
  auto &features = modao.features();
  auto *gradients_memory = dao.gradients_memory.data();
  auto *gradients_average = dao.gradients_average.data();
  const auto &step = dao.step;
  const auto &n_epochs = dao.n_epochs;
  const auto n_samples = features.rows(), n_features = features.cols();
  const auto epoch_size = dao.epoch_size != 0 ? dao.epoch_size : n_samples;
  auto &history = dao.history;
  const auto &record_every = history.record_every;
  auto &last_record_time = history.last_record_time;
  auto &last_record_epoch = history.last_record_epoch;
  auto * iterate = dao.iterate.data(), * steps_corrections = dao.steps_corrections.data();
  T n_samples_inverse = ((double)1 / (double)n_samples), x_ij = 0, step_correction = 0;
  T grad_factor_diff = 0, grad_avg_j = 0, grad_i_factor = 0, grad_i_factor_old = 0;
  auto n_threads = dao.n_threads;
  size_t idx_nnz = 0, thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);
  const auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    for (size_t t = 0; t < thread_epoch_size; ++t) {
      INDICE_TYPE i = _next_i();
      size_t x_i_size = features.row_size(i);
      const T *x_i = features.row_raw(i);
      const INDICE_TYPE *x_i_indices = features.row_indices(i);
      grad_i_factor = MODEL::grad_i_factor(modao, iterate, i);
      grad_i_factor_old = gradients_memory[i];
      while (!gradients_memory[i].compare_exchange_weak(grad_i_factor_old, grad_i_factor)) {}
      grad_factor_diff = grad_i_factor - grad_i_factor_old;
      for (idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
        const INDICE_TYPE &j = x_i_indices[idx_nnz];
        x_ij = x_i[idx_nnz];
        grad_avg_j = gradients_average[j];
        step_correction = steps_corrections[j];
        while (!gradients_average[j].compare_exchange_weak(
            grad_avg_j, grad_avg_j + (grad_factor_diff * x_ij * n_samples_inverse))) {
        }
        iterate[j] = PROX::call_single(prox,
            iterate[j] - (step * (grad_factor_diff * x_ij + step_correction * grad_avg_j)),
            step * step_correction);
      }
      if constexpr(INTERCEPT) {
        iterate[n_features] -= step * (grad_factor_diff + gradients_average[n_features]);
        T gradients_average_j = gradients_average[n_features];
        while (!gradients_average[n_features].compare_exchange_weak(
            gradients_average_j, gradients_average_j + (grad_factor_diff * n_samples_inverse))) {
        }
        iterate[n_features] = PROX::call_single(prox, iterate[n_features], step);
      }
    }
    if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T, TOL>>::value) {
      if (n_thread == 0) {
        if ((last_record_epoch + epoch) == 1 ||
            ((last_record_epoch + epoch) % record_every == 0)) {
          auto end = std::chrono::steady_clock::now();
          double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                        static_cast<double>(std::chrono::steady_clock::period::den);
          history.save_history(time, epoch, iterate, n_features);
        }
      }
    }
  }
  if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T, TOL>>::value) {
    if (n_thread == 0) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                    static_cast<double>(std::chrono::steady_clock::period::den);
      last_record_time = time;
      last_record_epoch += n_epochs;
    }
  }
}

template <typename MODEL, bool INTERCEPT = false, typename PROX, typename NEXT_I, typename DAO>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox, NEXT_I _next_i) {
  using T = typename MODEL::value_type;
  auto &history = dao.history;
  history.init(dao.n_epochs / history.record_every + 1, dao.iterate.size());
  std::vector<std::thread> threads;
  for (size_t i = 1; i < dao.n_threads; i++) {
    threads.emplace_back(
      [&](size_t n_thread) {
        threaded_solve<MODEL, INTERCEPT>(dao, modao, prox, _next_i, n_thread);
      }, i);
  }
  threaded_solve<MODEL, INTERCEPT>(dao, modao, prox, _next_i, 0);
  for (auto & thread : threads) thread.join();
}

}  // namespace sparse
}  // namespace asaga
}  // namespace statick

#endif  // STATICK_SOLVER_ASAGA_HPP_
