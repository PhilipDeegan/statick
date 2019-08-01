#ifndef STATICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
#define STATICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
namespace statick {
namespace logreg {

template <typename T, typename FEATURES>
T loss(const FEATURES &features, const T *const labels, T *coeffs) {
  const size_t rows = features.rows();
  T t{0};
  for (size_t i = 0; i < rows; i++) t += logistic(features.row(i).dot(coeffs) * labels[i]);
  return t / rows;
}
template <typename T, typename FEATURES>
T get_inner_prod(const FEATURES &features, T *coeffs, const size_t i) {
  return features.row(i).dot(coeffs);
}
template <typename T, typename K, typename FEATURES>
T grad_i_factor(const FEATURES &features, const T *labels, const K *coeffs, const size_t i) {
  return labels[i] * (sigmoid(labels[i] * get_inner_prod(features, coeffs, i)) - 1);
}

template <typename _F, typename _L>
class DAO {
 private:
 public:
  using FEATURE = typename std::conditional<is_shared_ptr<_F>::value, typename _F::element_type, _F>::type;
  using FEATURES = _F;
  using LABELS = _L;
  using LABEL = typename std::conditional<is_shared_ptr<_L>::value, typename _L::element_type, _L>::type;
  using T = typename FEATURE::value_type;
  using value_type = T;

  DAO(){}
  DAO(FEATURES &&_features, LABELS &&_labels) : m_features(_features), m_labels(_labels) { init(); }
  DAO(FEATURES &_features, LABELS &_labels) : m_features(_features), m_labels(_labels)   { init(); }

  void init() { vars[0] = m_features->cols(); vars[1] = m_features->rows(); }

  auto &features() const { if constexpr(is_shared_ptr<_F>::value) return *m_features; else return m_features; }
  auto &labels()   const { if constexpr(is_shared_ptr<_L>::value) return *m_labels; else return m_labels; }

  inline const size_t &n_features() const { return vars[0]; }
  inline const size_t &n_samples() const { return vars[1]; }

  void print(){
    std::cout << vars[0] << std::endl;
    std::cout << vars[1] << std::endl;
  }

  FEATURES m_features;
  LABELS m_labels;
  size_t vars[2];  // cols, rows;
};

namespace dense {
template <typename FEATURES, typename LABELS>
using DAO = logreg::DAO<std::shared_ptr<FEATURES>,
                        std::shared_ptr<LABELS>>;
}  // namespace dense
namespace sparse {
template <typename FEATURES, typename LABELS>
using DAO = logreg::DAO<std::shared_ptr<FEATURES>,
                        std::shared_ptr<LABELS>>;
}  // namespace sparse
}  // namespace logreg

template <typename _F, typename _L>
class ModelLogReg {
 public:
  using DAO = logreg::DAO<_F, _L>;
  using value_type = typename DAO::value_type;
  using T = value_type;

  template <bool INTERCEPT = false, bool FILL = true, class K>
  static T grad_i_factor(DAO &dao, K*coeffs, const size_t i) {
    return logreg::grad_i_factor(dao.features(), dao.labels().data(), coeffs, i);
  }

  template <bool INTERCEPT = false, bool FILL = true, class K>
  static void compute_grad_i(DAO &dao, const K *coeffs, T *out, const size_t i) {
    auto *x_i = dao.features().row_raw(i);
    const T alpha_i = grad_i_factor(dao, coeffs, i);
    if constexpr(FILL) mult_fill(out, x_i, alpha_i, dao.n_features());
    else mult_incr(out, x_i, alpha_i, dao.n_features());
    if constexpr(INTERCEPT) {
      if constexpr(FILL) out[dao.n_features()] = alpha_i;
      else out[dao.n_features()] += alpha_i;
    }
  }

  template <bool INTERCEPT = false, bool FILL = true, class K>
  static void grad_i(DAO &dao, const K *coeffs, T *out, const size_t i) {
    compute_grad_i<INTERCEPT, true, K>(dao, coeffs, out, i);
  }

  template <bool INTERCEPT, class K>
  static void grad(DAO &dao, const K *coeffs, T *out, const size_t size) {
    set(out, T{0}, size);
    std::vector<T> buffer(size, 0);
    for (size_t i = 0; i < dao.n_samples(); ++i) {
      grad_i<INTERCEPT, false>(dao, coeffs, buffer.data(), i);
      mult_incr(out, buffer.data(), 1, size);
    }
    for (size_t j = 0; j < size; ++j) out[j] /= dao.n_samples();
  }

  static T LOSS(DAO &dao, T *coeffs) {
    return logreg::loss(dao.features(), dao.labels().data(), coeffs);
  }
};
}  // namespace statick
#endif  // STATICK_LINEAR_MODEL_MODEL_LOGREG_HPP_
