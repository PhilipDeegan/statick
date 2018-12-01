
#ifndef TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
#define TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_

namespace tick {

template <typename T>
T abs(const T &f) {
  return f < 0 ? f * -1 : f;
}

template <typename T>
T sigmoid(const T z) {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <typename T>
T logistic(const T z) {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}

namespace logreg {
template <typename T>
T dot(const T *t1, const T *t2, size_t size) {
  T res{0};
  for (size_t i = 0; i < size; ++i) res += t1[i] * t2[i];
  return res;
}

template <typename T>
T loss(const Sparse2DRaw<T> &features, const T * const labels, T *coeffs) {
  const size_t &rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}

template <typename T, typename FEATURES>
T loss(const FEATURES &features, const T * const labels, T *coeffs) {
  const size_t &rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}

template <typename T>
T grad_i_factor(T *features, size_t cols, ulong i, T y_i, size_t coeffs_size, T *coeffs) {
  return y_i * (sigmoid(y_i * get_inner_prod(features, cols, i, coeffs_size, coeffs)) - 1);
}

template <typename T>
T get_inner_prod(const size_t i, const size_t cols, const size_t rows, T *features, T *coeffs) {
  return dot(coeffs, &features[i * cols], cols);
}
template <typename T>
T grad_i_factor(const size_t i, const size_t cols, const size_t rows, T *features, T *labels,
                T *coeffs) {
  const T y_i = labels[i];
  return y_i * (sigmoid(y_i * get_inner_prod(i, cols, rows, features, coeffs)) - 1);
}
}  // namespace logreg
}  // namespace tick

#endif  // TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_