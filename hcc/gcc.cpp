
namespace statick {

template <class T>
T pow(const T &f, const long double &e = 2) __CPU__ {
  return ::pow(f, e);
}
template <class T>
T pow(const T &f, const long double &e = 2) __HC__ {
  return Kalmar::precise_math::pow(f, e);
}
template <class T>
T abs(const T &f) __CPU__ __HC__ {
  return f < 0 ? f * -1 : f;
}
template <class T>
T exp(const T f) __CPU__ {
  return ::exp(f);
}
template <class T>
T exp(const T f) __HC__ {
  return Kalmar::precise_math::exp(f);
}
template <class T>
T log(const T f) __CPU__ {
  return ::log(f);
}
template <class T>
T log(const T f) __HC__ {
  return Kalmar::precise_math::log(f);
}

template <class T>
T sigmoid(const T z) __CPU__ __HC__ {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <class T>
T logistic(const T z) __CPU__ __HC__ {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}

namespace saga {
using INDEX_TYPE = INDICE_TYPE;
namespace dense {
template <typename T>
void solve(T *features, T *labels, T *gradients_average, T *gradients_memory, T *iterate,
           INDEX_TYPE *next_i) __CPU__ __HC__ {
  size_t N_FEATURES = 200;
  size_t N_SAMPLES = 75000;
  T N_SAMPLES_inverse = ((double)1 / (double)N_SAMPLES);
  double step = 0.00257480411965l;
  ulong n_features = N_FEATURES;
  for (ulong t = 0; t < N_SAMPLES; ++t) {
    INDEX_TYPE &i = next_i[t];
    T grad_i_factor = statick::hcc::LogisticRegression<T>::grad_i_factor(i, N_FEATURES, N_SAMPLES,
                                                                      features, labels, iterate);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    const T *x_i = statick::hcc::LogisticRegression<T>::get_features_raw(features, N_FEATURES, i);
    for (ulong j = 0; j < n_features; ++j) {
      T grad_avg_j = gradients_average[j];
      iterate[j] -= step * (grad_factor_diff * x_i[j] + grad_avg_j);
      gradients_average[j] += grad_factor_diff * x_i[j] * N_SAMPLES_inverse;
    }
  }
}
}  // namespace dense

namespace sparse {
template <typename T, typename Sparse2D>
void solve(const Sparse2D &features, T *labels, T *gradients_average, T *gradients_memory,
           T *iterate, size_t *next_i, T *steps_correction) __CPU__ __HC__ {
  size_t n_samples = features.rows();
  T n_samples_inverse = ((double)1 / (double)n_samples);
  double step = 0.00257480411965l;

  for (ulong t = 0; t < n_samples; ++t) {
    size_t &i = next_i[t];
    size_t x_i_size = features.row_size(i);
    const T *x_i = features.row_raw(i);
    const INDEX_TYPE *x_i_indices = features.row_indices(i);
    T grad_i_factor = labels[i] * (hcc::sigmoid(labels[i] * features.row(i).dot(iterate)) - 1);
    T grad_i_factor_old = gradients_memory[i];
    gradients_memory[i] = grad_i_factor;
    T grad_factor_diff = grad_i_factor - grad_i_factor_old;
    for (ulong idx_nnz = 0; idx_nnz < x_i_size; ++idx_nnz) {
      const INDEX_TYPE &j = x_i_indices[idx_nnz];
      iterate[j] -=
          step * (grad_factor_diff * x_i[idx_nnz] + steps_correction[j] * gradients_average[j]);
      gradients_average[j] += grad_factor_diff * x_i[idx_nnz] * n_samples_inverse;
    }
  }
}
template <typename Sparse2D, class T = double>
std::vector<T> compute_columns_sparsity(const Sparse2D &features) {
  std::vector<T> column_sparsity(features.cols());
  std::fill(column_sparsity.begin(), column_sparsity.end(), 0);
  double samples_inverse = 1. / features.rows();
  for (ulong i = 0; i < features.rows(); ++i) {
    auto row = features.row_indices(i);
    for (ulong j = 0; j < features.row_size(i); ++j) column_sparsity[row[j]] += 1;
  }
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
}  // namespace statick
