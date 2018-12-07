#ifndef TICK_SURVIVAL_MODEL_SCCS_H_
#define TICK_SURVIVAL_MODEL_SCCS_H_

#include "tick/survival/dao/model_sccs.hpp"

namespace tick {
namespace sccs {
// template <typename T>
// // T dot(const T *t1, const T *t2, size_t size) {
// //   T res{0};
// //   for (size_t i = 0; i < size; ++i) res += t1[i] * t2[i];
// //   return res;
// // }
// // template <typename T>
// // T loss(const Sparse2DRaw<T> &features, const T *const labels, T *coeffs) {
// //   const size_t &rows = features.rows();
// //   T t{0};
// //   for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
// //   return t / rows;
// // }
// // template <typename T, typename FEATURES>
// T loss(const FEATURES &features, const T *const labels, T *coeffs) {
//   const size_t &rows = features.rows();
//   T t{0};
//   for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
//   return t / rows;
// }

// double ModelSCCS::loss(const ArrayDouble &coeffs) {
//   double loss = 0;
//   for (ulong i = 0; i < n_samples; ++i) loss += loss_i(i, coeffs);

//   return loss / n_samples;
// }


// double ModelSCCS::loss_i(const ulong i, const ArrayDouble &coeffs) {
//   double loss = 0;
//   ArrayDouble inner_prod(n_intervals), softmax(n_intervals);
//   ulong max_interval = get_max_interval(i);

//   for (ulong t = 0; t < max_interval; t++)
//     inner_prod[t] = get_inner_prod(i, t, coeffs);
//   if (max_interval < n_intervals)
//     view(inner_prod, max_interval, n_intervals).fill(0);

//   softMax(inner_prod, softmax);

//   for (ulong t = 0; t < max_interval; t++)
//     loss -= get_longitudinal_label(i, t) * log(softmax[t]);

//   return loss;
// }

// // template <typename T>
// // T grad_i_factor(T *features, size_t cols, size_t i, T y_i, size_t coeffs_size, T *coeffs) {
// //   return y_i * (sigmoid(y_i * get_inner_prod(features, cols, i, coeffs_size, coeffs)) - 1);
// // }
// // template <typename T>
// // T get_inner_prod(const size_t i, const size_t cols, const size_t rows, T *features, T *coeffs) {
// //   return dot(coeffs, &features[i * cols], cols);
// // }

}

template <typename T>
class TModelSCCS {
 public:
  using DAO = sccs::DAO<T>;

  static size_t get_max_interval(DAO &dao, size_t i) { return std::min(dao.censoring[i], dao.n_intervals()); }

  static const T *get_longitudinal_features(DAO &dao, size_t i, size_t t) { return dao.features[i]->row_raw(t); }

  static const size_t get_longitudinal_label(DAO &dao, size_t i, size_t t) { return (*dao.labels[i])[t]; }

  static T get_inner_prod(DAO &dao, const T *const coeffs, const size_t i, const size_t t) {
    return dao.features[i]->row(t).dot(coeffs);
  }

  static void grad_i(DAO &dao, const T *const coeffs, T *out, const size_t size, const size_t i) {
    set(out, T{0}, size);
    std::vector<T> inner_prod(dao.n_intervals());
    std::vector<T> buffer(dao.n_intervals(), 0);
    size_t max_interval = get_max_interval(dao, i);

    for (size_t t = 0; t < max_interval; t++) inner_prod[t] = get_inner_prod(dao, coeffs, i, t);

    if (max_interval < dao.n_intervals()) set(inner_prod.data() + max_interval, dao.n_intervals() - max_interval, 0);

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

  static void grad(DAO &dao, const T *coeffs, T *out, const size_t size) {
    set(out, T{0}, size);
    std::vector<T> buffer(size, 0);
    for (size_t i = 0; i < dao.n_samples(); ++i) {
      grad_i(dao, coeffs, buffer.data(), size, i);
      mult_incr(out, buffer.data(), 1, size);
    }

    for (size_t j = 0; j < size; ++j) {
      out[j] /= dao.n_samples();
    }
  }
};

}  // namespace tick
#endif  // TICK_SURVIVAL_MODEL_SCCS_H_
