#ifndef STATICK_PROX_PROX_L2_HPP_
#define STATICK_PROX_PROX_L2_HPP_
namespace statick {
namespace prox_l2 {
namespace p0 { // > 0
template <class T>
void call(const T *coeffs, T step, T *out, size_t size, T strength) {
  const T thresh = step * strength * std::sqrt(size);
  T norm = std::sqrt(norm_sq(coeffs, size));
  if (norm <= thresh) {
    for (size_t i = 0; i < size; i++) out[i] = 0;
  } else {
    T t = 1. - thresh / norm;
    kul::math::scale(size, t, out);
  }
}
}  // namespace p0
template <class T, class K>
void call(const K *coeffs, T step, K *out, size_t size, T strength) {
  p0::call(coeffs, step, out, size, strength);
  for (size_t i = 0; i < size; ++i) if (out[i] < 0) out[i] = 0;
}
template <class T, class K>
T value(const K *coeffs, size_t size, T strength) {
  return strength * std::sqrt((size) * norm_sq(coeffs, size));
}
}  // namespace prox_l2
template <typename T, bool POSITIVE = 0>
class ProxL2 {
 public:
  ProxL2(T _strength) : strength(_strength){}
  static inline void call(ProxL2 &prox, const T* coeffs, T step, T *out, size_t size) {
    if constexpr (POSITIVE) prox_l2::call(coeffs, step, out, size, prox.strength);
    else prox_l2::p0::call(coeffs, step, out, size, prox.strength);
  }
  static inline T value(ProxL2 &prox, const T *coeffs, size_t size) {
    return prox_l2::value(coeffs, size, prox.strength);
  }
  T strength {0};
};
}  // namespace statick
#endif  // STATICK_PROX_PROX_L2_HPP_
