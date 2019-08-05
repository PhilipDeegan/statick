#ifndef STATICK_PROX_PROX_ZERO_HPP_
#define STATICK_PROX_PROX_ZERO_HPP_

namespace statick {

template <typename T, bool POSITIVE = 0>
class ProxZero {
 public:
  static constexpr bool is_separable = true;
  static constexpr std::string_view NAME = "zero";
  using value_type = T;

  static inline T value(ProxZero &prox, const T *coeffs, const size_t size) {
    return 0;
  }
  static inline T call_single(ProxZero &prox, const T x, T step) {
    return x;
  }

  static inline void call(ProxZero &prox, const T* coeffs, T step, T *out, size_t size) {}
};

}  // namespace statick

#endif  // STATICK_PROX_PROX_ZERO_HPP_
