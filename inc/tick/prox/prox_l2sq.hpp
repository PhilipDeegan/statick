

#ifndef TICK_PROX_PROX_L2SQ_HPP_
#define TICK_PROX_PROX_L2SQ_HPP_

namespace tick {
namespace prox_l2sq {
namespace np {
template <typename T>
void call_single(size_t i, const T *coeffs, T step, T *out, T strength) {
  out[i] = coeffs[i] / (1 + step * strength);
}
template <typename T>
void call_single(size_t i, const T *coeffs, T step, T *out, T strength, size_t n_times) {
  out[i] = coeffs[i] / std::pow(1 + step * strength, n_times);
}
template <typename T>
void call(const T *coeffs, T step, T *out, size_t size, T strength) {
  for (size_t i = 0; i < size; i++) call_single(i, coeffs, step, out, strength);
}
template <typename T>
void call(const T *coeffs, T step, T *out, size_t start, size_t end, T strength) {
  for (size_t i = start; i < end; i++) call_single(i, coeffs, step, out, strength);
}
}  // namespace np

template <typename T>
void call_single(size_t i, const T *coeffs, T step, T *out, T strength) {
  if (coeffs[i] < 0)
    out[i] = 0;
  else
    np::call_single(i, coeffs, step, out, strength);
}
template <typename T>
void call_single(size_t i, const T *coeffs, T step, T *out, T strength, size_t n_times) {
  if (coeffs[i] < 0)
    out[i] = 0;
  else
    np::call_single(i, coeffs, step, out, strength, n_times);
}
template <typename T>
void call(const T *coeffs, T step, T *out, size_t size, T strength) {
  for (size_t i = 0; i < size; i++) call_single(i, coeffs, step, out, strength);
}

template <typename T>
void call(const T *coeffs, T step, T *out, size_t start, size_t end, T strength) {
  for (size_t i = start; i < end; i++) call_single(i, coeffs, step, out, strength);
}

template <typename T>
T value_single(T x) {
  return x * x / 2;
}

}  // namespace prox_l2sq

}  // namespace tick

#endif  // TICK_PROX_PROX_L2SQ_HPP_
