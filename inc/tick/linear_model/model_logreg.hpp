#ifndef TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
#define TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
namespace tick {
namespace logreg {

template <typename T, typename FEATURES>
T loss(const FEATURES &features, const T *const labels, T *coeffs) {
  const size_t &rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}
template <typename T, typename FEATURES>
T get_inner_prod(const FEATURES &features, T *coeffs, const size_t cols, const size_t rows,
                 const size_t i) {
  return features.row(i).dot(coeffs);
}
template <typename T, typename FEATURES>
T grad_i_factor(const FEATURES &features, const T *labels, T *coeffs, const size_t cols,
                const size_t rows, const size_t i) {
  const T y_i = labels[i];
  return y_i * (sigmoid(y_i * get_inner_prod(features, coeffs, cols, rows, i)) - 1);
}

template <typename T, typename FEATURES, typename LABELS>
class DAO {
 public:
  DAO(FEATURES &&_features, LABELS &&_labels) : m_features(_features), m_labels(_labels) {
    vars[0] = m_features->cols();
    vars[1] = m_features->rows();
  }
  FEATURES m_features;
  LABELS m_labels;
  size_t vars[2];  // cols, rows;

  inline const size_t &n_features() const { return vars[0]; }
  inline const size_t &n_samples() const { return vars[1]; }

  auto &features() const { return *m_features; }
  auto &labels() const { return *m_labels; }
};

namespace dense {
template <typename T>
using DAO = logreg::DAO<T, std::shared_ptr<Array2D<T>>, std::shared_ptr<Array<T>>>;
}  // namespace dense
namespace sparse {
template <typename T>
using DAO = logreg::DAO<T, std::shared_ptr<Sparse2D<T>>, std::shared_ptr<Array<T>>>;
}  // namespace sparse
}  // namespace logreg

template <typename T = double, typename LOGREG_DAO = logreg::dense::DAO<T>>
class TModelLogReg {
 public:
  using DAO = LOGREG_DAO;
  static T grad_i_factor(DAO &dao, T *coeffs, const size_t i) {
    return logreg::grad_i_factor(dao.features(), dao.labels().data(), coeffs, dao.vars[0],
                                 dao.vars[1], i);
  }

  template <bool INTERCEPT = false, bool FILL = true, class K>
  static void grad_i(DAO &dao, const K *coeffs, T *out, const size_t i) {
    compute_grad_i<INTERCEPT, true, T, K>(dao, coeffs, out, i);
  }

  template <bool INTERCEPT = false, bool FILL = true, class K>
  static void compute_grad_i(DAO &dao, const size_t i, const K *coeffs, T *out, const bool fill) {
    auto *x_i = dao.get_features(i);
    const T alpha_i = grad_i_factor(i, coeffs);
    if
      constexpr(FILL) mult_fill(out, x_i, alpha_i, dao.n_features());
    else
      mult_incr(out, x_i, alpha_i, dao.n_features());
    if
      constexpr(INTERCEPT) {
        if
          constexpr(FILL) out[dao.n_features()] = alpha_i;
        else
          out[dao.n_features()] += alpha_i;
      }
  }

  static void grad(DAO &dao, const T *coeffs, T *out, const size_t size) {
    set(out, T{0}, size);
    std::vector<T> buffer(size, 0);
    for (size_t i = 0; i < dao.n_samples(); ++i) {
      grad_i(dao, coeffs, buffer.data(), size, i);
      mult_incr(out, buffer.data(), 1, size);
    }
    for (size_t j = 0; j < size; ++j) out[j] /= dao.n_samples();
  }
};

}  // namespace tick
#endif  // TICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
