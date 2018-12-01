

namespace statick {
namespace hcc {

template <class T>
class LogisticRegression {
 public:
  static T *get_features_raw(T *features, size_t cols, size_t i) __CPU__ __HC__ {
    return &features[cols * i];
  }
  static T dot(T *t1, T *t2, size_t size) __CPU__ __HC__ {
    T result{0};
    for (size_t i = 0; i < size; ++i) result += t1[i] * t2[i];
    return result;
  }
  static T get_inner_prod(T *features, size_t cols, size_t i, size_t coeffs_size,
                          T *coeffs) __CPU__ __HC__ {
    return dot(coeffs, get_features_raw(features, cols, i), coeffs_size);
  }
  static T loss_i(T *features, size_t cols, ulong i, T y_i, size_t coeffs_size,
                  T *coeffs) __CPU__ __HC__ {
    return logistic(get_inner_prod(features, cols, i, coeffs) * y_i);
  }
  static T grad_i_factor(T *features, size_t cols, ulong i, T y_i, size_t coeffs_size,
                         T *coeffs) __CPU__ __HC__ {
    return y_i * (sigmoid(y_i * get_inner_prod(features, cols, i, coeffs_size, coeffs)) - 1);
  }

  static T get_inner_prod(const size_t i, const size_t cols, const size_t rows, T *features,
                          T *coeffs) __CPU__ __HC__ {
    return dot(coeffs, &features[i * cols], cols);
  }
  static T grad_i_factor(const size_t i, const size_t cols, const size_t rows, T *features,
                         T *labels, T *coeffs) __CPU__ __HC__ {
    const T y_i = labels[i];
    return y_i * (sigmoid(y_i * get_inner_prod(i, cols, rows, features, coeffs)) - 1);
  }

  static T pow(const T &f, const long double &e = 2) __CPU__ { return ::pow(f, e); }
  static T pow(const T &f, const long double &e = 2) __HC__ {
    return Kalmar::precise_math::pow(f, e);
  }

  static T abs(const T &f) __CPU__ __HC__ { return f < 0 ? f * -1 : f; }
  static T exp(const T f) __CPU__ { return ::exp(f); }
  static T exp(const T f) __HC__ { return Kalmar::precise_math::exp(f); }
  static T log(const T f) __CPU__ { return ::log(f); }
  static T log(const T f) __HC__ { return Kalmar::precise_math::log(f); }

  static inline T sigmoid(const T z) __CPU__ __HC__ {
    if (z > 0) return 1 / (1 + exp(-z));
    const T exp_z = exp(z);
    return exp_z / (1 + exp_z);
  }
  static inline T logistic(const T z) __CPU__ __HC__ {
    if (z > 0) return log(1 + exp(-z));
    return -z + log(1 + exp(z));
  }
  static void sigmoid(const std::vector<T> &x, const std::vector<T> &out) __CPU__ __HC__ {
    for (ulong i = 0; i < x.size(); ++i) out[i] = sigmoid(x[i]);
  }
  static void logistic(const std::vector<T> &x, const std::vector<T> &out) __CPU__ __HC__ {
    for (ulong i = 0; i < x.size(); ++i) out[i] = logistic(x[i]);
  }
};

}  // namespace hcc
}  // namespace statick
