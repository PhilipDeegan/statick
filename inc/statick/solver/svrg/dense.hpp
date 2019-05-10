#ifndef STATICK_SOLVER_SVRG_DENSE_HPP_
#define STATICK_SOLVER_SVRG_DENSE_HPP_

namespace statick {
namespace svrg {
namespace dense {

template <typename MODAO, typename HISTOIR, bool INTERCEPT = false,
          typename T = typename MODAO::value_type>
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
  std::vector<T> iterate, full_gradient, fixed_w, grad_i, grad_i_fixed_w, next_iterate;
};

template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename PROX,
          typename NEXT_I, typename DAO>
void solve_thread(DAO &dao, typename MODEL::DAO &modao, PROX &prox,
                  NEXT_I fn_next_i, size_t n_thread) {
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  const auto &n_threads = dao.n_threads;
  const auto epoch_size = dao.epoch_size != 0 ? dao.epoch_size : n_samples;
  size_t iterate_size = n_features + static_cast<size_t>(INTERCEPT);
  size_t thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);
  const auto &step = dao.step, epoch_size_inverse = 1.0 / epoch_size;
  (void)  epoch_size_inverse;  // possibly unused if constexpr
  auto * iterate = dao.iterate.data(), *full_gradient = dao.full_gradient.data();
  auto * fixed_w = dao.fixed_w.data(), *grad_i_fixed_w = dao.grad_i_fixed_w.data();
  auto * grad_i = dao.grad_i.data();
  for (size_t t = 0; t < thread_epoch_size; ++t) {
    const size_t i = fn_next_i();
    MODEL::grad_i(modao, iterate, grad_i, iterate_size, i);
    MODEL::grad_i(modao, fixed_w, grad_i_fixed_w, iterate_size, i);
    for (size_t j = 0; j < iterate_size; ++j)
      iterate[j] = iterate[j] - step * (dao.grad_i[j] - dao.grad_i_fixed_w[j] + full_gradient[j]);
    PROX::call(prox, iterate, step, iterate, iterate_size);
    if constexpr(RM == VarianceReductionMethod::Random)
      copy(iterate, dao.next_iterate.data(), dao.next_iterate.size());
    if constexpr(RM == VarianceReductionMethod::Average)
      mult_incr(dao.next_iterate.data(), iterate, n_features, epoch_size_inverse);
  }
}

template <typename MODEL, uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false, typename PROX,
          typename NEXT_I, typename DAO>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX &prox, NEXT_I fn_next_i) {
  using T = typename MODEL::value_type;
  using TOL = typename DAO::HISTORY::TOLERANCE;
  auto &history = dao.history;
  const size_t n_samples = modao.n_samples(), n_features = modao.n_features();
  history.init(dao.n_epochs / history.record_every + 1, dao.iterate.size());
  const auto &n_epochs = dao.n_epochs, &n_threads = dao.n_threads;
  const auto epoch_size = dao.epoch_size != 0 ? dao.epoch_size : n_samples;
  const auto &record_every = history.record_every;
  auto &last_record_time = history.last_record_time;
  auto &last_record_epoch = history.last_record_epoch;
  auto * iterate = dao.iterate.data();
  statick::ThreadPool pool(n_threads);
  std::vector<std::function<void()>> funcs;
  for (size_t i = 1; i < n_threads; i++)
    funcs.emplace_back([&](){
      solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, prox, fn_next_i, i); });

  auto start = std::chrono::steady_clock::now();
  auto log_history = [&](size_t epoch){
    dao.t += epoch_size;
    if ((last_record_epoch + epoch) == 1 || ((last_record_epoch + epoch) % record_every == 0)) {
      auto end = std::chrono::steady_clock::now();
      double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
          static_cast<double>(std::chrono::steady_clock::period::den);
      history.save_history(time, epoch, iterate, n_features);
    }
  };

  for (size_t epoch = 1; epoch < (n_epochs + 1); ++epoch) {
    statick::svrg::prepare_solve<MODEL>(dao, modao, dao.t, fn_next_i);
    pool.async(funcs);
    solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, prox, fn_next_i, 0);
    pool.sync();
    log_history(epoch);
    if constexpr(RM == VarianceReductionMethod::Last)
      for (size_t i = 0; i < dao.iterate.size(); i++) dao.next_iterate[i] = dao.iterate[i];
  }

  dao.t += dao.epoch_size;
  if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T, TOL>>::value) {
    auto end = std::chrono::steady_clock::now();
    double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                  static_cast<double>(std::chrono::steady_clock::period::den);
    last_record_time = time;
    last_record_epoch += n_epochs;
  }
}

}  // namespace dense
}
}


#endif  // STATICK_SOLVER_SVRG_DENSE_HPP_
