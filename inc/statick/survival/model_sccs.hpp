#ifndef STATICK_SURVIVAL_MODEL_SCCS_H_
#define STATICK_SURVIVAL_MODEL_SCCS_H_

#include "statick/survival/dao/model_sccs.hpp"

namespace statick {
namespace sccs {}

template <typename MODAO>
class TModelSCCS {
 public:
  using DAO = MODAO;
  using T = typename DAO::value_type;
  using value_type = T;

  static size_t get_max_interval(DAO &dao, size_t i) {
    return std::min(dao.censoring[i], dao.n_intervals());
  }

  static const T *get_longitudinal_features(DAO &dao, size_t i, size_t t) {
    return dao.features[i]->row_raw(t);
  }

  static size_t get_longitudinal_label(DAO &dao, size_t i, size_t t) {
    return (*dao.labels[i])[t];
  }

  static T get_inner_prod(DAO &dao, const T *const coeffs, const size_t i, const size_t t) {
    return dao.features[i]->row(t).dot(coeffs);
  }

  static void grad_i(DAO &dao, const T *const coeffs, T *out, const size_t size, const size_t i) {
    set(out, T{0}, size);
    std::vector<T> inner_prod(dao.n_intervals());
    std::vector<T> buffer(dao.n_lagged_features(), 0);
    size_t max_interval = get_max_interval(dao, i);
    for (size_t t = 0; t < max_interval; t++) inner_prod[t] = get_inner_prod(dao, coeffs, i, t);
    if (max_interval < dao.n_intervals())
      set(inner_prod.data() + max_interval, dao.n_intervals() - max_interval, 0);
    T x_max = max(inner_prod.data(), inner_prod.size());
    T sum_exp = sumExpMinusMax(inner_prod.data(), inner_prod.size(), x_max);
    T multiplier = 0;  // need a double instead of long double for mult_incr
    for (size_t t = 0; t < max_interval; t++) {
      multiplier = exp(inner_prod[t] - x_max) / sum_exp;  // overflow-proof
      mult_incr(buffer.data(), get_longitudinal_features(dao, i, t), multiplier, buffer.size());
    }
    T label = 0;
    for (size_t t = 0; t < max_interval; t++) {
      label = get_longitudinal_label(dao, i, t);
      if (label != 0) {
        mult_incr(out, get_longitudinal_features(dao, i, t), -label, size);
        mult_incr(out, buffer.data(), label, size);
      }
    }
  }

  template <bool INTERCEPT, class K>
  static void grad(DAO &dao, const K *coeffs, T *out, const size_t size) {
    set(out, T{0}, size);
    std::vector<T> buffer(size, 0);
    for (size_t i = 0; i < dao.n_samples(); ++i) {
      grad_i(dao, coeffs, buffer.data(), size, i);
      mult_incr(out, buffer.data(), 1, size);
    }
    for (size_t j = 0; j < size; ++j) out[j] /= dao.n_samples();
  }

  static T loss(DAO &dao, const T *coeffs) {
    T loss = 0;
    for (size_t i = 0; i < dao.n_samples(); ++i) loss += loss_i(dao, coeffs, i);

    return loss / dao.n_samples();
  }

  static T loss_i(DAO &dao, const T *coeffs, const size_t i) {
    T loss{0};
    std::vector<T> inner_prod(dao.n_intervals()), softmax(dao.n_intervals());
    size_t max_interval = get_max_interval(dao, i);

    for (size_t t = 0; t < max_interval; t++) inner_prod[t] = get_inner_prod(dao, coeffs, i, t);
    if (max_interval < dao.n_intervals())
      set(inner_prod.data() + max_interval, dao.n_intervals() - max_interval, 0);

    softMax(inner_prod.data(), softmax.data(), dao.n_intervals());

    for (size_t t = 0; t < max_interval; t++)
      loss -= get_longitudinal_label(dao, i, t) * log(softmax[t]);

    return loss;
  }
};
}  // namespace statick
#endif  // STATICK_SURVIVAL_MODEL_SCCS_H_
