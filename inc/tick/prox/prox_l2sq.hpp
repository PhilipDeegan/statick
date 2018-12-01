


#ifndef TICK_PROX_PROX_L2SQ_HPP_
#define TICK_PROX_PROX_L2SQ_HPP_

namespace tick {
namespace prox_l2sq {

template <typename T>
inline
void call_single(ulong i, const T *coeffs, T step, T *out, T strength) {
  if (coeffs[i] < 0) out[i] = 0;
  out[i] = coeffs[i] / (1 + step * strength);
}
template <typename T>
inline
void call(const T *coeffs, T step, T *out, size_t size, T strength) {
  for (size_t i = 0; i < size; i++) call_single(i, coeffs, step, out, strength);
};

}
}

#endif  // TICK_PROX_PROX_L2SQ_HPP_