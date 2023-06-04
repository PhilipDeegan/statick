#ifndef STATICK_ARRAY_MATH_HPP_
#define STATICK_ARRAY_MATH_HPP_

#include "mkn/kul/math.hpp"

namespace statick {

template <typename T>
static inline T sum(const T *x, const size_t size) {
  T t{0};
  for (size_t i = 0; i < size; i++) t += x[i];
  return t;
}

template <typename T>
static inline void copy(const T *from, T *to, const size_t size) {
  for (size_t i = 0; i < size; i++) to[i] = from[i];
}

template <typename T, typename Y>
inline void set(T *x, Y y, const size_t size) {
  for (size_t i = 0; i < size; i++) x[i] = y;
}

template <typename T>
static inline T dot(const T *const x, const T *const y, const size_t size) {
  return mkn::kul::math::dot(size, x, y);
}

template <typename T, typename Y>
static inline void mult_incr(T *x, const T *const y, const Y a, const size_t size) {
  return mkn::kul::math::mult_incr<T, Y, T>(size, a, y, x);
}

template <typename T, typename Y>
void mult_fill(T *x, const T *const y, const Y a, const size_t size) {
  for (size_t j = 0; j < size; ++j) x[j] = y[j] * a;
}

template <typename T>
static inline T max(const T *x, size_t size) {
  T _m{0};
  for (size_t i = 0; i < size; i++)
    if (x[i] > _m) _m = x[i];
  return _m;
}

template <typename T>
static inline T sumExpMinusMax(const T *x, size_t size, T x_max) {
  T sum = 0;
  for (size_t i = 0; i < size; ++i) sum += exp(x[i] - x_max);
  return sum;
}

template <typename T>
static inline T logSumExp(const T *x, size_t size) {
  T x_max = max(x, size);
  return x_max + log(sumExpMinusMax(x, size, x_max));
}

template <typename T>
static inline void softMax(const T *x, T *out, size_t size) {
  T x_max = max(x, size);
  T sum = sumExpMinusMax(x, size, x_max);
  for (size_t i = 0; i < size; i++) out[i] = exp(x[i] - x_max) / sum;
}

template <typename T>
static inline T norm_sq(const T *array, const size_t size) {
  T n_sq{0};
  for (size_t i = 0; i < size; ++i) n_sq = (array[i] * array[i]) + n_sq;
  return n_sq;
}

template <typename T>
static inline T abs(const T f) {
  return f < 0 ? f * -1 : f;
}

template <typename T>
static inline T sigmoid(const T z) {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <typename T>
static inline T logistic(const T z) {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}

}  // namespace statick

#endif  //  STATICK_ARRAY_MATH_HPP_
