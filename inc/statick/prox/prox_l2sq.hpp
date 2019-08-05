#ifndef STATICK_PROX_PROX_L2SQ_HPP_
#define STATICK_PROX_PROX_L2SQ_HPP_
namespace statick {
namespace prox_l2sq{
namespace p0 { // > 0
template <class T> static inline
T call_single(T x, T step, T strength) {
  return x / (1 + step * strength);
}
}  // namespace p0 { // > 0
template <class T>
T call_single(T x, T step, T strength) {
  if (x < 0) return 0;
  return p0::call_single(x, step, strength);
}
template <typename T> static inline
T value_single(T x) {
  return x * x / 2;
}
template <class T> static inline
T value(const T *coeffs, size_t size, T strength) {
  T val = 0;
  for (size_t i = 0; i < size; i++) val += value_single(coeffs[i]);
  return strength * val;
}
}  // namespace prox_l2sq
template <typename T, bool POSITIVE = 0>
class ProxL2Sq {
 public:
  static constexpr std::string_view NAME = "l2sq";
  using value_type = T;

  ProxL2Sq(T _strength) : strength(_strength){}
  static inline T value(ProxL2Sq &prox, const T *coeffs, const size_t size) {
    return statick::prox_l2sq::value(coeffs, size, prox.strength);
  }
  static inline T call_single(ProxL2Sq &prox, const T x, T step) {
    if constexpr (POSITIVE)
      return statick::prox_l2sq::call_single(x, step, prox.strength);
    else
      return statick::prox_l2sq::p0::call_single(x, step, prox.strength);
  }

  static inline void call(ProxL2Sq &prox, const T* coeffs, T step, T *out, size_t size) {
    if constexpr (POSITIVE)
      for (size_t i = 0; i < size; i++)
        out[i] = prox_l2sq::call_single(coeffs[i], step, prox.strength);
    else
      for (size_t i = 0; i < size; i++)
        out[i] = prox_l2sq::p0::call_single(coeffs[i], step, prox.strength);
  }
  static inline void call(const T *coeffs, T step, T *out, size_t size) {}

  T strength {0};
};
}  // namespace statick
#endif  // STATICK_PROX_PROX_L2SQ_HPP_
