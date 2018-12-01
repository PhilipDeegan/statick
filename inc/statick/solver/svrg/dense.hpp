

namespace statick {
namespace svrg {
namespace dense {

template <typename MODAO, bool INTERCEPT = false,
          typename T = typename MODAO::value_type,
          typename HISTOIR = statick::solver::History<T>>
class DAO {
 public:
  using HISTORY = HISTOIR;
  DAO(MODAO &modao, size_t _n_epochs, size_t _epoch_size, size_t _threads)
      : n_epochs(_n_epochs),
        epoch_size(_epoch_size), n_threads(_threads),
        iterate(modao.n_features() + static_cast<size_t>(INTERCEPT)) {
  }
  T step = 0.00257480411965l;
  size_t n_epochs, epoch_size, rand_index = 0, n_threads, t = 0;
  HISTORY history;
  std::vector<T> iterate;
  std::vector<T> full_gradient, fixed_w, grad_i, grad_i_fixed_w, next_iterate;
};

template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename PROX,
          typename NEXT_I, typename DAO>
void solve_thread(DAO &dao, typename MODEL::DAO &modao, PROX call,
                  NEXT_I fn_next_i, size_t n_thread) {
  using T = typename MODEL::value_type;
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  size_t iterate_size = n_features + static_cast<size_t>(INTERCEPT);
  auto *full_gradient = dao.full_gradient.data();
  auto &next_iterate = dao.next_iterate;
  const auto &step = dao.step;
  const auto &n_epochs = dao.n_epochs;
  const auto epoch_size = dao.epoch_size != 0 ? dao.epoch_size : n_samples;
  auto &history = dao.history;
  const auto &record_every = history.record_every;
  auto &last_record_time = history.last_record_time;
  auto &last_record_epoch = history.last_record_epoch;
  auto * iterate = dao.iterate.data();

  T epoch_size_inverse = 1.0 / epoch_size;
  auto n_threads = dao.n_threads;
  size_t idx_nnz = 0, thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);
  const auto start = std::chrono::steady_clock::now();
  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    for (size_t t = 0; t < thread_epoch_size; ++t) {
      const size_t i = fn_next_i();
      MODEL::grad_i(modao, iterate, dao.grad_i.data(), iterate_size, i);
      MODEL::grad_i(modao, dao.fixed_w.data(), dao.grad_i_fixed_w.data(), iterate_size, i);
      for (size_t j = 0; j < iterate_size; ++j)
        iterate[j] =
            iterate[j] - dao.step * (dao.grad_i[j] - dao.grad_i_fixed_w[j] + dao.full_gradient[j]);
      call(iterate, dao.step, iterate, iterate_size);
      if constexpr(RM == VarianceReductionMethod::Random)
        copy(iterate, dao.next_iterate.data(), dao.next_iterate.size());
      if constexpr(RM == VarianceReductionMethod::Average)
        mult_incr(dao.next_iterate.data(), iterate, n_features, epoch_size_inverse);
    }
    if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T>>::value) {
      if (n_thread == 0) {
        auto end = std::chrono::steady_clock::now();
        double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                      static_cast<double>(std::chrono::steady_clock::period::den);
        last_record_time = time;
        last_record_epoch += n_epochs;
      }
    }
  }
  if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T>>::value) {
    if (n_thread == 0) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                    static_cast<double>(std::chrono::steady_clock::period::den);
      last_record_time = time;
      last_record_epoch += n_epochs;
    }
  }
}

template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename PROX,
          typename NEXT_I, typename DAO>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX call, NEXT_I fn_next_i) {
  using T = typename MODEL::value_type;
  const size_t iterate_size = dao.iterate.size();
  auto n_threads = dao.n_threads;
  auto &history = dao.history;
  if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T>>::value) {
    history.time_history.resize(dao.n_epochs / history.record_every + 1);
    history.epoch_history.resize(dao.n_epochs / history.record_every + 1);
    history.iterate_history.resize(dao.n_epochs / history.record_every + 1);
  }
  statick::svrg::prepare_solve<MODEL>(dao, modao, dao.t, fn_next_i);
  if (n_threads > 1) {
    std::vector<std::thread> threadsV;
    for (size_t i = 0; i < n_threads; i++)
      threadsV.emplace_back([&]() {
        solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, fn_next_i, i);
      });
    for (size_t i = 0; i < n_threads; i++) threadsV[i].join();
  } else {
    solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, fn_next_i, 0);
  }
  if constexpr(RM == VarianceReductionMethod::Last)
    for (size_t i = 0; i < iterate_size; i++) dao.next_iterate[i] = dao.iterate[i];
  dao.t += dao.epoch_size;
}

}  // namespace dense
}
}
