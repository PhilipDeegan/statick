#ifndef STATICK_SOLVER_SVRG_SPARSE_HPP_
#define STATICK_SOLVER_SVRG_SPARSE_HPP_

namespace statick {
namespace svrg {
namespace sparse {

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

template <typename MODAO, bool INTERCEPT = false,
          typename T = typename MODAO::value_type,
          typename HISTOIR = statick::solver::History<T>>
class DAO {
 public:
  using HISTORY = HISTOIR;
  DAO(MODAO &modao, size_t _n_epochs, size_t _epoch_size, size_t _threads)
      : n_epochs(_n_epochs),
        epoch_size(_epoch_size), n_threads(_threads),
        iterate(modao.n_features() + static_cast<size_t>(INTERCEPT)),
        steps_corrections(statick::svrg::sparse::compute_step_corrections(modao.features())) {
  }
  T step = 0.00257480411965l;
  size_t n_epochs, epoch_size, rand_index = 0, n_threads, t = 0;
  HISTORY history;
  std::vector<T> iterate, steps_corrections;
  std::vector<T> full_gradient, fixed_w, grad_i, grad_i_fixed_w, next_iterate;
};

template <typename MODEL, uint16_t RM, uint16_t ST, bool INTERCEPT,
          typename PROX, typename NEXT_I,
          typename DAO, typename T = typename MODEL::value_type>
void solve_thread(DAO &dao, typename MODEL::DAO &modao, PROX call,
                  NEXT_I fn_next_i,size_t n_thread) {

  auto is_in_range = [](size_t){ return 1; }; // TODO - TODO
  auto & features = modao.features();
  auto & next_iterate = dao.next_iterate;
  auto * full_gradient = dao.full_gradient.data();
  const auto &step = dao.step;
  const auto n_samples = features.rows(), n_features = features.cols();
  (void) n_features;  // possibly unused if constexpr
  const auto epoch_size = dao.epoch_size != 0 ? dao.epoch_size : n_samples;
  auto * iterate = dao.iterate.data();
  auto * steps_corrections = dao.steps_corrections.data();

  auto n_threads = dao.n_threads;
  size_t thread_epoch_size = epoch_size / n_threads;
  thread_epoch_size += n_thread < (epoch_size % n_threads);

  for (size_t t = 0; t < thread_epoch_size; ++t) {
    const size_t i = fn_next_i();
    const size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDICE_TYPE *x_i_indices = features.row_indices(i);
    // // Gradients factors (model is a GLM)
    // // TODO: a grad_i_factor(i, array1, array2) to loop once on the features
    T grad_i_diff = MODEL::grad_i_factor(modao, iterate, i) - MODEL::grad_i_factor(modao, dao.fixed_w.data(), i);
    // T grad_i_diff = model->grad_i_factor(i, iterate) - model->grad_i_factor(i, fixed_w);
    // We update the iterate within the support of the features vector, with the probabilistic correction
    for (size_t idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      // Get the index of the idx-th sparse feature of x_i
      size_t j = x_i_indices[idx_nnz];
      T full_gradient_j = full_gradient[j];
      // Step-size correction for coordinate j
      T step_correction = steps_corrections[j];
      // Gradient descent with probabilistic step-size correction
      T descent_direction = step * (x_i[idx_nnz] * grad_i_diff +
          step_correction * full_gradient_j);
      if (is_in_range(j)) call(j, iterate[j] - descent_direction, step * step_correction, iterate);
      else iterate[j] -= descent_direction;
    }
    // And let's not forget to update the intercept as well. It's updated at each
    // step, so no step-correction. Note that we call the prox, in order to be
    // consistent with the dense case (in the case where the user has the weird
    // desire to to regularize the intercept)
    if constexpr (INTERCEPT) {
      T descent_direction = step * (grad_i_diff + full_gradient[n_features]);
      if (is_in_range(n_features))
        iterate[n_features] = call(
            iterate[n_features] - descent_direction, step, n_features);
      else
        iterate[n_features] -= descent_direction;
    }
    // Note that the average option for variance reduction with sparse data is a
    // very bad idea, but this is caught in the python class
    if constexpr (RM == VarianceReductionMethod::Random) if(t == dao.rand_index) next_iterate = iterate;
    if constexpr (RM == VarianceReductionMethod::Average) dao.next_iterate.mult_incr(iterate, 1.0 / epoch_size);
  }
}

template <typename MODEL,
          uint16_t RM = VarianceReductionMethod::Last,
          uint16_t ST = StepType::Fixed, bool INTERCEPT = false,
          typename PROX, typename NEXT_I, typename DAO>
void solve(DAO &dao, typename MODEL::DAO &modao, PROX call, NEXT_I fn_next_i) {
  using T = typename MODEL::value_type;
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
      solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, fn_next_i, i); });

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
    solve_thread<MODEL, RM, ST, INTERCEPT>(dao, modao, call, fn_next_i, 0);
    pool.sync();
    log_history(epoch);
    if constexpr(RM == VarianceReductionMethod::Last)
      for (size_t i = 0; i < dao.iterate.size(); i++)
        dao.next_iterate[i] = dao.iterate[i];
  }

  dao.t += epoch_size;
  if constexpr(std::is_same<typename DAO::HISTORY, statick::solver::History<T>>::value) {
    auto end = std::chrono::steady_clock::now();
    double time = ((end - start).count()) * std::chrono::steady_clock::period::num /
                  static_cast<double>(std::chrono::steady_clock::period::den);
    last_record_time = time;
    last_record_epoch += n_epochs;
  }
}

}  // namespace sparse
}
}

#endif  // STATICK_SOLVER_SVRG_SPARSE_HPP_