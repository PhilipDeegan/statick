#ifndef STATICK_PROX_PROX_L2SQ_HPP_
#define STATICK_PROX_PROX_L2SQ_HPP_
namespace statick {
template <typename T, bool POSITIVE = 0>
class ProxZero {
 public:
  static inline T value(ProxZero &prox, const T *coeffs, const size_t size) {
    return 0;
  }
  static inline T call_single(ProxZero &prox, const T x, T step) {
    return x;
  }
};
}  // namespace statick
#endif  // STATICK_PROX_PROX_L2SQ_HPP_
