#ifndef TICK_ARRAY_MATH_HPP_
#define TICK_ARRAY_MATH_HPP_
namespace tick {

template <typename T>
inline T sum(const T *x, const size_t size) {
  T t{0};
  for (size_t i = 0; i < size; i++) t += x[i];
  return t;
}

template <typename T>
inline void copy(const T *from, T *to, const size_t size) {
  for (size_t i = 0; i < size; i++) to[i] = from[i];
}

template <typename T, typename Y>
inline void set(T *x, Y y, const size_t size) {
  for (size_t i = 0; i < size; i++) x[i] = y;
}

template <typename T>
inline T dot(const T *x, const T *y, const size_t size) {
  T result{0};
  for (size_t i = 0; i < size; i++) result += x[i] * y[i];
  return result;
}

template <typename T, typename Y>
void mult_incr(T *x, const T *const y, const Y a, const size_t size) {
  for (size_t j = 0; j < size; j++) x[j] += y[j] * a;
}

template <typename T>
T max(T *x, size_t size) {
  T m{0};
  for (size_t i = 0; i < size; i++)
    if (x[i] > m) m = x[i];
  return m;
}

template <typename T>
T sumExpMinusMax(T *x, size_t size, T x_max) {
  T sum = 0;
  for (ulong i = 0; i < size; ++i) sum += exp(x[i] - x_max);  // overflow-proof
  return sum;
}

template <typename T>
T logSumExp(const T *x, size_t size) {
  T x_max = max(x, size);
  return x_max + log(sumExpMinusMax(x, size, x_max));
}

template <typename T>
void softMax(const T *x, T *out, size_t size) {
  T x_max = max(x, size);
  T sum = sumExpMinusMax(x, size, x_max);
  for (ulong i = 0; i < x.size(); i++) {
    out[i] = exp(x[i] - x_max) / sum;  // overflow-proof
  }
}

template <typename T>
inline T norm_sq(const T *array, const size_t size) {
  T n_sq{0};
  for (size_t i = 0; i < size; ++i) n_sq = (array[i] * array[i]) + n_sq;
  return n_sq;
}

template <typename T>
T abs(const T f) {
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

}  // namespace tick

#endif  //  TICK_ARRAY_MATH_HPP_
