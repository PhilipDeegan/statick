
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE ulong
#else
#define INDICE_TYPE std::uint32_t
#endif

namespace statick {
namespace hcc {

template <class T>
class Sparse {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Sparse(const T *data, const size_t _size, const INDEX_TYPE *indices) __CPU__ __HC__
      : v_data(data),
        _size(_size),
        indices(indices) {}
  Sparse(const Sparse &that) __CPU__ __HC__ : v_data(that.data),
                                              _size(that._size),
                                              indices(that.indices) {}
  Sparse(Sparse &&that) __CPU__ __HC__ : _size(that._size) {
    this->v_data = that.v_data;
    this->indices = that.indices;
  };
  T &operator[](const Kalmar::index<1> &idx) const __CPU__ __HC__ { return v_data[idx]; }
  const T &operator[](int i) const __CPU__ __HC__ { return v_data[i]; }
  const T *data() const __CPU__ __HC__ { return v_data; }
  size_t size() const __CPU__ __HC__ { return _size; }
  T dot(const Sparse<T> &that) const __CPU__ __HC__ {
    T result = 0;
    uint64_t i1 = 0, i2 = 0;
    while (true) {
      if (i1 >= that._size) break;
      while (i2 < this->_size && this->indices()[i2] < that.indices[i1]) {
        i2++;
      }
      if (i2 >= this->_size) break;
      if (this->indices()[i2] == that.indices[i1]) {
        result += that.v_data[i2] * this->v_data[i1++];
      } else {
        while (i1 < that._size && this->indices()[i2] > that.indices[i1]) {
          i1++;
        }
      }
    }
  }
  T dot(const T *const that) const __CPU__ __HC__ {
    T result = 0;
    for (uint64_t i = 0; i < this->_size; i++) result += this->v_data[i] * that[this->indices[i]];
    return result;
  }

 private:
  const T *v_data;
  const size_t _size;
  const INDEX_TYPE *indices;
  Sparse() = delete;
  Sparse(Sparse &that) __CPU__ __HC__ = delete;
  Sparse(const Sparse &&that) __CPU__ __HC__ = delete;
  Sparse &operator=(Sparse &that) __CPU__ __HC__ = delete;
  Sparse &operator=(Sparse &&that) __CPU__ __HC__ = delete;
  Sparse &operator=(const Sparse &that) __CPU__ __HC__ = delete;
  Sparse &operator=(const Sparse &&that) __CPU__ __HC__ = delete;
};

template <class T>
class Sparse2D {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Sparse2D(T *data, size_t size, INDEX_TYPE *_indices, INDEX_TYPE *_row_indices, size_t cols,
           size_t rows)
      : v_data(size, data),
        a_info(3, 0),
        v_info(3, a_info.data()),
        v_indices(size, _indices),
        v_row_indices(rows + 1, _row_indices) {
    a_info[0] = cols;
    a_info[1] = rows;
    a_info[2] = size;
  }
  Sparse2D(Sparse2D &that) __CPU__ __HC__ : v_data(that.v_data),
                                            a_info(that.a_info),
                                            v_info(that.v_info),
                                            v_indices(that.v_indices),
                                            v_row_indices(that.v_row_indices) {}
  Sparse2D(Sparse2D &&that) __CPU__ __HC__ : v_data(that.v_data),
                                             a_info(that.a_info),
                                             v_info(that.v_info),
                                             v_indices(that.v_indices),
                                             v_row_indices(that.v_row_indices) {}

  T &operator[](const Kalmar::index<1> &idx) const __CPU__ __HC__ { return v_data[idx]; }
  T &operator[](int i) const __HC__ { return v_data[i]; }
  const T &operator[](int i) const { return v_data[i]; }
  T &operator[](int i) __CPU__ { return v_data[i]; }
  Sparse<T> row(size_t i) const __CPU__ __HC__ {
    return Sparse<T>(v_data.data() + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                     v_indices.data() + v_row_indices[i]);
  }
  const T *row_raw(size_t i) const __CPU__ __HC__ { return v_data.data() + v_row_indices[i]; }
  INDEX_TYPE row_size(size_t i) const __CPU__ __HC__ {
    size_t _rw = 0;
    if (i >= a_info[1])
      _rw = a_info[0] / 3;  // TODO FIX
    else
      _rw = v_row_indices[i + 1] - v_row_indices[i];
    // KLOG(INF) << _rw;
    return _rw;
  }
  size_t size() const __CPU__ __HC__ { return v_info[2]; }

  const INDEX_TYPE *row_indices(size_t i) const __CPU__ __HC__ {
    return v_indices.data() + v_row_indices[i];
  }
  const INDEX_TYPE *row_indices() const __CPU__ __HC__ { return v_row_indices.data(); }
  const INDEX_TYPE *indices() const __CPU__ __HC__ { return v_indices.data(); }

  size_t cols() const __CPU__ __HC__ { return v_info[0]; }
  size_t rows() const __CPU__ __HC__ { return v_info[1]; }
  const T *data() const __CPU__ __HC__ { return v_data.data(); }

 private:
  hc::array_view<T, 1> v_data;
  std::vector<size_t> a_info;
  hc::array_view<size_t, 1> v_info;
  hc::array_view<INDEX_TYPE, 1> v_indices, v_row_indices;

  Sparse2D() = delete;
  Sparse2D(const Sparse2D &&that) __CPU__ __HC__ = delete;
  Sparse2D &operator=(Sparse2D &that) __CPU__ __HC__ = delete;
  Sparse2D &operator=(Sparse2D &&that) __CPU__ __HC__ = delete;
  Sparse2D &operator=(const Sparse2D &that) __CPU__ __HC__ = delete;
  Sparse2D &operator=(const Sparse2D &&that) __CPU__ __HC__ = delete;
};

template <class T>
class Sparse2DView {
 private:
  using INDEX_TYPE = INDICE_TYPE;

 public:
  Sparse2DView(const T *_data, const size_t *_info, const INDEX_TYPE *_indices,
              const INDEX_TYPE *_row_indices) __CPU__ __HC__ : v_data(_data),
                                                               v_indices(_indices),
                                                               v_row_indices(_row_indices) {
    m_info[0] = _info[0]; m_info[1] = _info[1]; m_info[2] = _info[2];
  }
  Sparse2DView(const Sparse2DView &that) __CPU__ __HC__ : v_data(that.v_data),
                                                        v_indices(that.v_indices),
                                                        v_row_indices(that.v_row_indices) {
    m_info[0] = that.m_info[0]; m_info[1] = that.m_info[1]; m_info[2] = that.m_info[2];
  }

  Sparse2DView(const Sparse2DView &&that) __CPU__ __HC__ : v_data(that.v_data),
                                                         v_indices(that.v_indices),
                                                         v_row_indices(that.v_row_indices) {
    m_info[0] = that.m_info[0]; m_info[1] = that.m_info[1]; m_info[2] = that.m_info[2];
  }

  T &operator[](const Kalmar::index<1> &idx) const __CPU__ __HC__ { return v_data[idx]; }
  T &operator[](int i) __CPU__ __HC__ { return v_data[i]; }
  Sparse<T> row(size_t i) const __CPU__ __HC__ {
    return Sparse<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                     v_indices + v_row_indices[i]);
  }
  const T *row_raw(size_t i) const __CPU__ __HC__ { return v_data + v_row_indices[i]; }
  INDEX_TYPE row_size(size_t i) const __CPU__ __HC__ {
    return v_row_indices[i + 1] - v_row_indices[i];
  }

  const INDEX_TYPE *row_indices(size_t i) const __CPU__ __HC__ {
    return v_indices + v_row_indices[i];
  }
  const INDEX_TYPE *row_indices() const __CPU__ __HC__ { return v_row_indices; }
  const INDEX_TYPE *indices() const __CPU__ __HC__ { return v_indices; }

  size_t cols() const __CPU__ __HC__ { return m_info[0]; }
  size_t rows() const __CPU__ __HC__ { return m_info[1]; }
  size_t size() const __CPU__ __HC__ { return m_info[2]; }
  const T *data() const __CPU__ __HC__ { return v_data; }

 private:
  const T *v_data;
  size_t m_info[3];
  const INDEX_TYPE *v_indices, *v_row_indices;

  Sparse2DView() = delete;
  Sparse2DView(Sparse2DView &that) __CPU__ __HC__ = delete;
  Sparse2DView &operator=(Sparse2DView &that) __CPU__ __HC__ = delete;
  Sparse2DView &operator=(Sparse2DView &&that) __CPU__ __HC__ = delete;
  Sparse2DView &operator=(const Sparse2DView &that) __CPU__ __HC__ = delete;
  Sparse2DView &operator=(const Sparse2DView &&that) __CPU__ __HC__ = delete;
};

template <class T>
class Sparse2DList {
 private:
  using INDEX_TYPE = INDICE_TYPE;
  static constexpr size_t INFO_SIZE = 5;

 public:
  Sparse2DList(std::vector<T> &data, std::vector<size_t> &info, std::vector<INDEX_TYPE> &_indices,
               std::vector<INDEX_TYPE> &_rows_indices)
      : v_data(data.data()),
        av_data(data.size(), v_data),
        v_info(info.data()),
        av_info(info.size(), v_info),
        v_indices(_indices.data()),
        av_indices(_indices.size(), v_indices),
        v_row_indices(_rows_indices.data()),
        av_row_indices(_rows_indices.size(), v_row_indices) {}

  Sparse2DList(Sparse2DList &that) __CPU__ __HC__ : v_data(that.v_data),
                                                    av_data(that.av_data),
                                                    v_info(that.v_info),
                                                    av_info(that.av_info),
                                                    v_indices(that.v_indices),
                                                    av_indices(that.av_indices),
                                                    v_row_indices(that.v_row_indices),
                                                    av_row_indices(that.av_row_indices) {}

  Sparse2DList(Sparse2DList &&that) __CPU__ __HC__ : v_data(that.v_data),
                                                     av_data(that.av_data),
                                                     v_info(that.v_info),
                                                     av_info(that.av_info),
                                                     v_indices(that.v_indices),
                                                     av_indices(that.av_indices),
                                                     v_row_indices(that.v_row_indices),
                                                     av_row_indices(that.av_row_indices) {}

  Sparse2DView<T> operator[](size_t i) const __HC__ {
    return Sparse2DView<T>(av_data.data() + (av_info[(i * INFO_SIZE) + 3]), &av_info[i * INFO_SIZE],
                          av_indices.data() + (av_info[(i * INFO_SIZE) + 3]),
                          av_row_indices.data() + (av_info[(i * INFO_SIZE) + 4]));
  }

  Sparse2DView<T> operator[](size_t i) const __CPU__ {
    return Sparse2DView<T>(v_data + (v_info[(i * INFO_SIZE) + 3]), &v_info[i * INFO_SIZE],
                          v_indices + (v_info[(i * INFO_SIZE) + 3]),
                          v_row_indices + (v_info[(i * INFO_SIZE) + 4]));
  }

  const INDEX_TYPE *indices() const __CPU__ { return v_indices; }
  const INDEX_TYPE *indices() const __HC__ { return av_indices.data(); }
  const INDEX_TYPE *row_indices() const __CPU__ { return v_row_indices; }
  const INDEX_TYPE *row_indices() const __HC__ { return av_row_indices.data(); }
  const size_t *const info() const __CPU__ { return v_info; }
  const size_t *const info() const __HC__ { return av_info.data(); }

 private:
  T *v_data;
  hc::array_view<T, 1> av_data;
  size_t *v_info;
  hc::array_view<size_t, 1> av_info;
  INDEX_TYPE *v_indices, *v_row_indices;
  hc::array_view<INDEX_TYPE, 1> av_indices, av_row_indices;

  Sparse2DList() __CPU__ __HC__ = delete;
  Sparse2DList(const Sparse2DList &that) __CPU__ __HC__ = delete;
  Sparse2DList(const Sparse2DList &&that) __CPU__ __HC__ = delete;
  Sparse2DList &operator=(Sparse2DList &that) __CPU__ __HC__ = delete;
  Sparse2DList &operator=(Sparse2DList &&that) __CPU__ __HC__ = delete;
  Sparse2DList &operator=(const Sparse2DList &that) __CPU__ __HC__ = delete;
  Sparse2DList &operator=(const Sparse2DList &&that) __CPU__ __HC__ = delete;
};

template <class T>
T pow(const T &f, const long double &e = 2) __CPU__ {
  return ::pow(f, e);
}
template <class T>
T pow(const T &f, const long double &e = 2) __HC__ {
  return Kalmar::precise_math::pow(f, e);
}
template <class T>
T abs(const T &f) __CPU__ __HC__ {
  return f < 0 ? f * -1 : f;
}
template <class T>
T exp(const T f) __CPU__ {
  return ::exp(f);
}
template <class T>
T exp(const T f) __HC__ {
  return Kalmar::precise_math::exp(f);
}
template <class T>
T log(const T f) __CPU__ {
  return ::log(f);
}
template <class T>
T log(const T f) __HC__ {
  return Kalmar::precise_math::log(f);
}

template <class T>
T sigmoid(const T z) __CPU__ __HC__ {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <class T>
T logistic(const T z) __CPU__ __HC__ {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}
template <class T>
void sigmoid(const std::vector<T> &x, const std::vector<T> &out) __CPU__ __HC__ {
  for (ulong i = 0; i < x.size(); ++i) out[i] = sigmoid(x[i]);
}
template <class T>
void logistic(const std::vector<T> &x, const std::vector<T> &out) __CPU__ __HC__ {
  for (ulong i = 0; i < x.size(); ++i) out[i] = logistic(x[i]);
}

}  // namespace hcc
}  // namespace statick
