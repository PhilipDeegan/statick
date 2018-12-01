#ifndef TICK_PROX_PROX_L2_HPP_
#define TICK_PROX_PROX_L2_HPP_
namespace statick {
namespace prox_l2 {
namespace np {
template <class T, class K>
void call(const K *coeffs, T step, K *out, size_t start, size_t end, T strength) {
  size_t size = end - start;
  const T thresh = step * strength * std::sqrt(end - start);
  T n_sq = std::sqrt(norm_sq(&coeffs[start], size));
  if (n_sq <= thresh) {
    for (size_t i = start; i < end; i++) out[i] = 0;
  } else {
    T t = 1. - thresh / n_sq;
    for (size_t i = start; i < end; i++) out[i] = out[i] * t;
  }
}
template <class T, class K>
void call(const K *coeffs, T step, K *out, size_t size, T strength) {
  call(coeffs, step, out, 0, size, strength);
}
}  // namespace np
template <class T, class K>
void call(const K *coeffs, T step, K *out, size_t start, size_t end, T strength) {
  np::call(coeffs, step, out, start, end, strength);
  size_t size = end - start;
  for (size_t i = start; i < size; ++i)
    if (out[i] < 0) out[i] = 0;
}
template <class T, class K>
void call(const K *coeffs, T step, K *out, size_t size, T strength) {
  call(coeffs, step, out, 0, size, strength);
}
template <class T, class K>
T value(const K *coeffs, size_t start, size_t end, T strength) {
  T n_sq = norm_sq(coeffs[start], end - start);
  return strength * std::sqrt((end - start) * n_sq);
}
}  // namespace prox_l2
}  // namespace statick
#endif  // TICK_PROX_PROX_L2_HPP_
