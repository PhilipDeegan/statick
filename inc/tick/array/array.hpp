#ifndef TICK_ARRAY_ARRAY_HPP_
#define TICK_ARRAY_ARRAY_HPP_

#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif

namespace tick {

template <typename T>
T norm_sq(const T *array, size_t size) {
  T n_sq{0};
  for (size_t i = 0; i < size; ++i) n_sq = (array[i] * array[i]) + n_sq;
  return n_sq;
}

template <typename T>
T abs(const T f) {
  return f < 0 ? f * -1 : f;
}

template <typename T>
T sigmoid(const T z) {
  if (z > 0) return 1 / (1 + exp(-z));
  const T exp_z = exp(z);
  return exp_z / (1 + exp_z);
}

template <typename T>
T logistic(const T z) {
  if (z > 0) return log(1 + exp(-z));
  return -z + log(1 + exp(z));
}

template <class Archive, class T>
bool load_array_with_raw_data(Archive &ar, std::vector<T> &data) {
  bool is_sparse = false;
  ar(CEREAL_NVP(is_sparse));
  if (is_sparse) return false;
  size_t vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));
  if (data.size() < vectorSize) data.resize(vectorSize);
  ar(cereal::binary_data(data.data(), static_cast<std::size_t>(vectorSize) * sizeof(T)));
  return true;
}

template <class T>
class Array {
 public:
  Array() {}
  Array(Array &&that) : m_data(that.m_data) {}

  const T *data() const { return m_data.data(); }
  const size_t &size() const { return m_data.size(); }
  const T &operator[](int i) const { return m_data[i]; }

  T dot(const T *const that) const {
    T result = 0;
    for (size_t i = 0; i < this->m_data.size(); i++) result += this->m_data[i] * that[i];
    return result;
  }
  T dot(const Array<T> &that) const { return dot(that.m_data); }

  static std::shared_ptr<Array<T>> FROM_CEREAL(const std::string &file) {
    auto array = std::make_shared<Array<T>>();
    {
      std::ifstream bin_data(file, std::ios::in | std::ios::binary);
      cereal::PortableBinaryInputArchive iarchive(bin_data);
      if (tick::load_array_with_raw_data(iarchive, array->m_data)) return std::move(array);
    }
    return nullptr;
  }

 private:
  std::vector<T> m_data;

  Array(Array &that) = delete;
  Array(const Array &that) = delete;
  Array(const Array &&that) = delete;
  Array &operator=(Array &that) = delete;
  Array &operator=(Array &&that) = delete;
  Array &operator=(const Array &that) = delete;
  Array &operator=(const Array &&that) = delete;
};

template <class T>
class ArrayRaw {
 public:
  ArrayRaw(const T *_data, const size_t _size) : _size(_size), v_data(_data) {}
  ArrayRaw(ArrayRaw &&that) : _size(that._size), v_data(that.v_data) {}

  const T *data() const { return v_data; }
  const size_t &size() const { return _size; }
  const T &operator[](int i) const { return v_data[i]; }

  T dot(const T *const that) const {
    T result = 0;
    for (size_t i = 0; i < this->_size; i++) result += this->v_data[i] * that[i];
    return result;
  }
  T dot(const ArrayRaw<T> &that) const { return dot(that.v_data); }

 private:
  const size_t _size;
  const T *v_data;
  ArrayRaw() = delete;
  ArrayRaw(ArrayRaw &that) = delete;
  ArrayRaw(const ArrayRaw &that) = delete;
  ArrayRaw(const ArrayRaw &&that) = delete;
  ArrayRaw &operator=(ArrayRaw &that) = delete;
  ArrayRaw &operator=(ArrayRaw &&that) = delete;
  ArrayRaw &operator=(const ArrayRaw &that) = delete;
  ArrayRaw &operator=(const ArrayRaw &&that) = delete;
};

template <class Archive, class T>
bool load_array2d_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info) {
  bool is_sparse = false;
  size_t cols = 0, rows = 0, vectorSize = 0;
  ar(is_sparse);
  if (is_sparse) return false;
  ar(cols, rows);
  ar(cereal::make_size_tag(vectorSize));
  if ((cols * rows) != vectorSize) return false;
  if (data.size() < vectorSize) data.resize(vectorSize);
  ar(cereal::binary_data(data.data(), static_cast<std::size_t>(vectorSize) * sizeof(T)));
  info.resize(info.size() + 3);
  info[0] = cols;
  info[1] = rows;
  info[2] = vectorSize;
  return true;
}

template <class T>
class Array2D {
 public:
  Array2D() {}
  Array2D(Array2D &&that) : m_data(that.m_data), m_info(that.m_info) {}

  const T *data() const { return m_data.data(); }
  ArrayRaw<T> row(size_t i) const { return ArrayRaw<T>(&m_data[i * m_info[0]], m_info[0]); }
  const T *row_raw(size_t i) const { return &m_data[i * m_info[0]]; }
  const T &operator[](int i) const { return m_data[i]; }

  const size_t &cols() const { return m_info[0]; }
  const size_t &rows() const { return m_info[1]; }
  const size_t &size() const { return m_info[2]; }

  static std::shared_ptr<Array2D<T>> FROM_CEREAL(const std::string &file) {
    auto array = std::make_shared<Array2D<T>>();
    {
      std::ifstream bin_data(file, std::ios::in | std::ios::binary);
      cereal::PortableBinaryInputArchive iarchive(bin_data);
      if (tick::load_array2d_with_raw_data(iarchive, array->m_data, array->m_info))
        return std::move(array);
    }
    return nullptr;
  }

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;

  Array2D(Array2D &that) = delete;
  Array2D(const Array2D &that) = delete;
  Array2D(const Array2D &&that) = delete;
  Array2D &operator=(Array2D &that) = delete;
  Array2D &operator=(Array2D &&that) = delete;
  Array2D &operator=(const Array2D &that) = delete;
  Array2D &operator=(const Array2D &&that) = delete;
};

template <class T>
class Array2DRaw {
 public:
  Array2DRaw(const T *_data, const size_t *_info)
      : v_data(_data), m_cols(&_info[0]), m_rows(&_info[1]), m_size(&_info[2]) {}
  Array2DRaw(Array2DRaw &&that)
      : v_data(that.v_data), m_cols(that.m_cols), m_rows(that.m_rows), m_size(that.m_size) {}

  const T &operator[](int i) { return v_data[i]; }
  const T *data() const { return v_data; }
  const T *row_raw(size_t i) const { return &v_data[i * (*m_cols)]; }
  ArrayRaw<T> row(size_t i) const { return ArrayRaw<T>(&v_data[i * (*m_cols)], *m_cols); }

  const size_t &cols() const { return *m_cols; }
  const size_t &rows() const { return *m_rows; }
  const size_t &size() const { return *m_size; }

 private:
  const T *v_data;
  const size_t *m_cols, *m_rows, *m_size;

  Array2DRaw() = delete;
  Array2DRaw(Array2DRaw &that) = delete;
  Array2DRaw(const Array2DRaw &that) = delete;
  Array2DRaw(const Array2DRaw &&that) = delete;
  Array2DRaw &operator=(Array2DRaw &that) = delete;
  Array2DRaw &operator=(Array2DRaw &&that) = delete;
  Array2DRaw &operator=(const Array2DRaw &that) = delete;
  Array2DRaw &operator=(const Array2DRaw &&that) = delete;
};

template <class T>
class Array2DList {
  static constexpr size_t INFO_SIZE = 3;

 public:
  Array2DRaw<T> operator[](size_t i) const {
    return Array2DRaw<T>(m_data.data() + (m_info[(i * INFO_SIZE) + 2]), &m_info[i * INFO_SIZE]);
  }

  const size_t *info() const { return m_info.data(); }
  const T *data() const { return m_data.data(); }
  size_t size() const { return m_data.size(); }

  void add_from_cereal(const std::string &file) {}

  static std::shared_ptr<Array2DList<T>> FROM_CEREAL(const std::string &file) {}

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;

  Array2DList() = delete;
  Array2DList(Array2DList &that) = delete;
  Array2DList(const Array2DList &that) = delete;
  Array2DList(Array2DList &&that) = delete;
  Array2DList(const Array2DList &&that) = delete;
  Array2DList &operator=(Array2DList &that) = delete;
  Array2DList &operator=(Array2DList &&that) = delete;
  Array2DList &operator=(const Array2DList &that) = delete;
  Array2DList &operator=(const Array2DList &&that) = delete;
};

template <class T>
class Sparse {
 public:
  Sparse(const T *data, const size_t _size, const INDICE_TYPE *indices)
      : v_data(data), _size(_size), indices(indices) {}
  Sparse(Sparse &&that) : _size(that._size) {
    this->v_data = that.v_data;
    this->indices = that.indices;
  };

  const T &operator[](int i) const { return v_data[i]; }
  const T *data() const { return v_data; }
  const size_t &size() const { return _size; }

  T dot(const Sparse<T> &that) const {
    T result = 0;
    size_t i1 = 0, i2 = 0;
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
    return result;
  }
  T dot(const T *const that) const {
    T result = 0;
    for (size_t i = 0; i < this->_size; i++) result += this->v_data[i] * that[this->indices[i]];
    return result;
  }

 private:
  const T *v_data;
  const size_t _size;
  const INDICE_TYPE *indices;
  Sparse() = delete;
  Sparse(Sparse &that) = delete;
  Sparse(const Sparse &that) = delete;
  Sparse(const Sparse &&that) = delete;
  Sparse &operator=(Sparse &that) = delete;
  Sparse &operator=(Sparse &&that) = delete;
  Sparse &operator=(const Sparse &that) = delete;
  Sparse &operator=(const Sparse &&that) = delete;
};

template <class Archive, class T>
bool load_sparse2d_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                 std::vector<INDICE_TYPE> &indices,
                                 std::vector<INDICE_TYPE> &row_indices) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse, rows, cols, size);

  if (data.size() < 3) data.resize(size_sparse);
  if (info.size() < 3) info.resize(3);
  if (indices.size() < size_sparse) indices.resize(size_sparse);
  if (row_indices.size() < (rows + 1)) row_indices.resize(rows + 1);

  ar(cereal::binary_data(data.data(), sizeof(T) * size_sparse));
  ar(cereal::binary_data(indices.data(), sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(row_indices.data(), sizeof(INDICE_TYPE) * (rows + 1)));

  info[0] = cols;
  info[1] = rows;
  info[2] = size_sparse;
  return true;
}

template <class T>
class Sparse2D {
 public:
  Sparse2D() {}
  Sparse2D(Sparse2D &&that)
      : m_data(that.m_data),
        m_info(that.m_info),
        m_indices(that.m_indices),
        m_row_indices(that.m_row_indices) {}

  const T *data() const { return m_data.data(); }
  const T *row_raw(size_t i) const { return m_data.data() + m_row_indices[i]; }
  INDICE_TYPE row_size(size_t i) const { return m_row_indices[i + 1] - m_row_indices[i]; }
  const INDICE_TYPE *indices() const { return m_indices.data(); }
  const INDICE_TYPE *row_indices(size_t i) const { return m_indices.data() + m_row_indices[i]; }
  const INDICE_TYPE *row_indices() const { return m_row_indices.data(); }

  Sparse<T> row(size_t i) const {
    return Sparse<T>(m_data.data() + m_row_indices[i], m_row_indices[i + 1] - m_row_indices[i],
                     m_indices.data() + m_row_indices[i]);
  }

  const size_t &cols() const { return m_info[0]; }
  const size_t &rows() const { return m_info[1]; }
  const size_t &size() const { return m_info[2]; }

  static std::shared_ptr<Sparse2D<T>> FROM_CEREAL(const std::string &file) {
    auto array = std::make_shared<Sparse2D<T>>();
    {
      std::ifstream bin_data(file, std::ios::in | std::ios::binary);
      cereal::PortableBinaryInputArchive iarchive(bin_data);
      if (tick::load_sparse2d_with_raw_data(iarchive, array->m_data, array->m_info,
                                            array->m_indices, array->m_row_indices))
        return std::move(array);
    }
    return nullptr;
  }

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;
  std::vector<INDICE_TYPE> m_indices, m_row_indices;

  Sparse2D(Sparse2D &that) = delete;
  Sparse2D(const Sparse2D &that) = delete;
  Sparse2D(const Sparse2D &&that) = delete;
  Sparse2D &operator=(Sparse2D &that) = delete;
  Sparse2D &operator=(Sparse2D &&that) = delete;
  Sparse2D &operator=(const Sparse2D &that) = delete;
  Sparse2D &operator=(const Sparse2D &&that) = delete;
};

template <class T>
class Sparse2DRaw {
 public:
  Sparse2DRaw(const T *_data, const size_t *_info, const INDICE_TYPE *_indices,
              const INDICE_TYPE *_row_indices)
      : v_data(_data),
        m_cols(&_info[0]),
        m_rows(&_info[1]),
        m_size(&_info[2]),
        v_indices(_indices),
        v_row_indices(_row_indices) {}
  Sparse2DRaw(Sparse2DRaw &&that)
      : v_data(that.v_data),
        m_cols(that.m_cols),
        m_rows(that.m_rows),
        m_size(that.m_size),
        v_indices(that.v_indices),
        v_row_indices(that.v_row_indices) {}
  T &operator[](int i) { return v_data[i]; }
  const T *data() const { return v_data; }
  const T *row_raw(size_t i) const { return v_data + v_row_indices[i]; }
  INDICE_TYPE row_size(size_t i) const { return v_row_indices[i + 1] - v_row_indices[i]; }
  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices(size_t i) const { return v_indices + v_row_indices[i]; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }

  Sparse<T> row(size_t i) const {
    return Sparse<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                     v_indices + v_row_indices[i]);
  }

  const size_t &cols() const { return *m_cols; }
  const size_t &rows() const { return *m_rows; }
  const size_t &size() const { return *m_size; }

 private:
  const T *v_data;
  const size_t *m_cols, *m_rows, *m_size;
  const INDICE_TYPE *v_indices, *v_row_indices;

  Sparse2DRaw() = delete;
  Sparse2DRaw(Sparse2DRaw &that) = delete;
  Sparse2DRaw(const Sparse2DRaw &that) = delete;
  Sparse2DRaw(const Sparse2DRaw &&that) = delete;
  Sparse2DRaw &operator=(Sparse2DRaw &that) = delete;
  Sparse2DRaw &operator=(Sparse2DRaw &&that) = delete;
  Sparse2DRaw &operator=(const Sparse2DRaw &that) = delete;
  Sparse2DRaw &operator=(const Sparse2DRaw &&that) = delete;
};

template <class Archive, class T>
bool load_sparse2dlist_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                     std::vector<INDICE_TYPE> &indices,
                                     std::vector<INDICE_TYPE> &row_indices) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse);
  ar(rows, cols, size);

  data.resize(data.size() + size_sparse);
  info.resize(info.size() + 5);
  indices.resize(indices.size() + size_sparse);
  row_indices.resize(row_indices.size() + rows + 1);

  T *s_data = &data[data.size()] - size_sparse;
  size_t *s_info = &info[info.size()] - 5;
  INDICE_TYPE *s_indices = &indices[indices.size()] - size_sparse;
  INDICE_TYPE *s_row_indices = &row_indices[row_indices.size()] - (rows + 1);

  ar(cereal::binary_data(s_data, sizeof(T) * size_sparse));
  ar(cereal::binary_data(s_indices, sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(s_row_indices, sizeof(INDICE_TYPE) * (rows + 1)));

  s_info[0] = cols;
  s_info[1] = rows;
  s_info[2] = size_sparse;
  s_info[3] = data.size() - size_sparse;
  s_info[4] = row_indices.size() - (rows + 1);
  return true;
}

template <class T>
class Sparse2DList {
  static constexpr size_t INFO_SIZE = 5;

 public:
  Sparse2DList(std::vector<T> &data, std::vector<size_t> &info, std::vector<INDICE_TYPE> &_indices,
               std::vector<INDICE_TYPE> &_rows_indices)
      : v_data(data.data()),
        v_info(info.data()),
        v_indices(_indices.data()),
        v_row_indices(_rows_indices.data()) {}

  Sparse2DRaw<T> operator[](size_t i) const {
    return Sparse2DRaw<T>(v_data + (v_info[(i * INFO_SIZE) + 3]), &v_info[i * INFO_SIZE],
                          v_indices + (v_info[(i * INFO_SIZE) + 3]),
                          v_row_indices + (v_info[(i * INFO_SIZE) + 4]));
  }

  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }
  const size_t *info() const { return v_info; }

 private:
  T *v_data;
  size_t *v_info;
  INDICE_TYPE *v_indices, *v_row_indices;

  Sparse2DList() = delete;
  Sparse2DList(Sparse2DList &that) = delete;
  Sparse2DList(const Sparse2DList &that) = delete;
  Sparse2DList(Sparse2DList &&that) = delete;
  Sparse2DList(const Sparse2DList &&that) = delete;
  Sparse2DList &operator=(Sparse2DList &that) = delete;
  Sparse2DList &operator=(Sparse2DList &&that) = delete;
  Sparse2DList &operator=(const Sparse2DList &that) = delete;
  Sparse2DList &operator=(const Sparse2DList &&that) = delete;
};

}  // namespace tick

#endif  //  TICK_ARRAY_ARRAY_HPP_
