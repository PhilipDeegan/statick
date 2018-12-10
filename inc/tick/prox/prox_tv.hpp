#ifndef TICK_PROX_PROX_TV_HPP_
#define TICK_PROX_PROX_TV_HPP_
namespace tick {
namespace prox_tv {
template <bool POSITIVE = true, class T>
void call(const T *coeffs, T step, T *out, size_t start, size_t end, T strength) {
  size_t width = end - start;
  const T *sub_coeffs = &coeffs[start];
  T *sub_out = &out[start];
  const T thresh = step * strength;
  int k = 0, k0 = 0;               /*k: current sample location, k0: beginning of current segment*/
  T umin = thresh, umax = -thresh; /*u is the dual variable*/
  T vmin = sub_coeffs[0] - thresh, vmax = sub_coeffs[0] + thresh; /*bounds for the segment's value*/
  int kplus = 0, kminus = 0;        /*last positions where umax=-lambda, umin=lambda, respectively*/
  const T twolambda = 2.0 * thresh; /*auxiliary variable*/
  const T minlambda = -thresh;      /*auxiliary variable*/
  for (;;) {                        /*simple loop, the exit test is inside*/
    while (k == width - 1) {        /*we use the right boundary condition*/
      if (umin < 0.0) {             /*vmin is too high -> negative jump necessary*/
        do {
          sub_out[k0++] = vmin;
        } while (k0 <= kminus);
        umax = (vmin = sub_coeffs[kminus = k = k0]) + (umin = thresh) - vmax;
      } else if (umax > 0.0) { /*vmax is too low -> positive jump necessary*/
        do {
          sub_out[k0++] = vmax;
        } while (k0 <= kplus);
        umin = (vmax = sub_coeffs[kplus = k = k0]) + (umax = minlambda) - vmin;
      } else {
        vmin += umin / (k - k0 + 1);
        do {
          sub_out[k0++] = vmin;
        } while (k0 <= k);
        if
          constexpr(POSITIVE) {
            for (size_t i = start; i < end; i++) {
              if (out[i] < 0) {
                out[i] = 0;
              }
            }
          }
        return;
      }
    }
    /*negative jump necessary*/
    if ((umin += sub_coeffs[k + 1] - vmin) < minlambda) {
      do {
        sub_out[k0++] = vmin;
      } while (k0 <= kminus);
      vmax = (vmin = sub_coeffs[kplus = kminus = k = k0]) + twolambda;
      umin = thresh;
      umax = minlambda;
    } else if ((umax += sub_coeffs[k + 1] - vmax) > thresh) { /*positive jump necessary*/
      do {
        sub_out[k0++] = vmax;
      } while (k0 <= kplus);
      vmin = (vmax = sub_coeffs[kplus = kminus = k = k0]) - twolambda;
      umin = thresh;
      umax = minlambda;
    } else { /*no jump necessary, we continue*/
      k++;
      if (umin >= thresh) { /*update of vmin*/
        vmin += (umin - thresh) / ((kminus = k) - k0 + 1);
        umin = thresh;
      }
      if (umax <= minlambda) { /*update of vmax*/
        vmax += (umax + thresh) / ((kplus = k) - k0 + 1);
        umax = minlambda;
      }
    }
  }
}
template <bool POSITIVE = true, class T>
void call(const T *coeffs, T step, T *out, size_t size, T strength) {
  call(coeffs, step, out, 0, size, strength);
}

template <bool POSITIVE = true, class RAWrray, class T>
void call(const RAWrray &coeffs, T step, RAWrray &out, size_t start, size_t end, T strength) {
  size_t size = end - start;
  if (coeffs.size() < size) return;
  call<POSITIVE>(coeffs.data(), step, out.data(), start, end, strength);
}
template <bool POSITIVE = true, class RAWrray, class T>
void call(const RAWrray &coeffs, T step, RAWrray &out, size_t size, T strength) {
  call<POSITIVE>(coeffs, step, out.data(), 0, size, strength);
}

template <class RAWrray, class T>
T value(const RAWrray &coeffs, size_t start, size_t end, T strength) {
  T diff, tv_norm = 0;
  for (size_t i = start + 1; i < end; i++) {
    diff = coeffs[i] - coeffs[i - 1];
    if (diff > 0) tv_norm += diff;
    if (diff < 0) tv_norm -= diff;
  }
  return strength * tv_norm;
}
}  // namespace prox_tv
}  // namespace tick
#endif  // TICK_PROX_PROX_TV_HPP_
