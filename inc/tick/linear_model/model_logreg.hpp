#ifndef TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
#define TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
namespace tick {
namespace logreg {
template <typename T>
T dot(const T *t1, const T *t2, size_t size) {
  T res{0};
  for (size_t i = 0; i < size; ++i) res += t1[i] * t2[i];
  return res;
}
template <typename T>
T loss(const Sparse2DRaw<T> &features, const T *const labels, T *coeffs) {
  const size_t &rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}
template <typename T, typename FEATURES>
T loss(const FEATURES &features, const T *const labels, T *coeffs) {
  const size_t &rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}
template <typename T>
T grad_i_factor(T *features, size_t cols, size_t i, T y_i, size_t coeffs_size, T *coeffs) {
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
