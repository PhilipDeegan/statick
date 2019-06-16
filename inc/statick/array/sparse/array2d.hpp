#ifndef STATICK_ARRAY_SPARSE_ARRAY2D_HPP_
#define STATICK_ARRAY_SPARSE_ARRAY2D_HPP_

namespace statick {
namespace sparse_2d {

class Exception : public kul::Exception {
 public:
  Exception(const char *f, const size_t &l, const std::string &s) : kul::Exception(f, l, s) {}
};

// Non-atomic data, BinaryInputArchive
template <typename T, class Archive, typename S2D>
void inner_save(Archive &ar, const S2D &s2d) {
  ar(cereal::binary_data(s2d.data(), sizeof(T) * s2d.size()));
  ar(cereal::binary_data(s2d.indices(), sizeof(INDICE_TYPE) * s2d.size()));
  ar(cereal::binary_data(s2d.row_indices(), sizeof(INDICE_TYPE) * (s2d.rows() + 1)));
}
template <class Archive, class S2D>
void save_to(Archive &ar, const S2D &s2d) {
  ar(s2d.size());
  ar(s2d.rows());
  ar(s2d.cols());
  ar(s2d.cols() * s2d.rows());
  inner_save<typename S2D::value_type>(ar,s2d);
}
template <class S2D>
void save(const S2D &s2d, const std::string &_file) {
  std::ofstream ss(_file, std::ios::out | std::ios::binary);
  cereal::PortableBinaryOutputArchive ar(ss);
  save_to(ar, s2d);
}
}

template <class Archive, class T>
bool load_sparse2d_with_presized_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                 std::vector<INDICE_TYPE> &indices,
                                 std::vector<INDICE_TYPE> &row_indices) {
  size_t rows, cols, size_sparse, size;
  ar(size_sparse, rows, cols, size);
  ar(cereal::binary_data(data.data(), sizeof(T) * size_sparse));
  ar(cereal::binary_data(indices.data(), sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(row_indices.data(), sizeof(INDICE_TYPE) * (rows + 1)));
  info[0] = cols;
  info[1] = rows;
  info[2] = size_sparse;
  return true;
}

template <class Archive, class T>
bool load_sparse2d_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                 std::vector<INDICE_TYPE> &indices,
                                 std::vector<INDICE_TYPE> &row_indices) {
  size_t rows, cols, size_sparse, size;
  ar(size_sparse, rows, cols, size);
  if (info.size() < 3) info.resize(3);
  if (data.size() < size_sparse) data.resize(size_sparse);
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

template <class T> class RawSparse2D;

template <class T>
class Sparse2D {
 public:
  using value_type = T;
  using real_type  = Sparse2D<T>;
  using raw_type   = RawSparse2D<T>;
  using raw1d_type = Sparse<T>;
  static constexpr bool is_sparse =  1;

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
      if (statick::load_sparse2d_with_raw_data(iarchive, array->m_data, array->m_info,
                                            array->m_indices, array->m_row_indices))
        return std::move(array);
    }
    return nullptr;
  }

  template <class Archive> void load(Archive &ar) {
    load_sparse2d_with_raw_data(ar, m_data, m_info, m_indices, m_row_indices);
  }
  template <class Archive> void save(Archive &ar) const { sparse_2d::save<T>(ar, *this); }


  statick::Array2D<T> toArray2D() const;

  static std::shared_ptr<Sparse2D<T>> RANDOM(size_t rows, size_t cols, T density, T seed = -1) {
    if(density < 0 || density > 1)
      KEXCEPT(sparse_2d::Exception, "Invalid sparse density, must be between 0 and 1");

    std::mt19937_64 generator;
    if(seed > 0) generator = std::mt19937_64(seed);
    else{
      std::random_device r;
      std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
      generator = std::mt19937_64(seed_seq);
    }
    std::uniform_real_distribution<T> dist;

    auto arr = std::make_shared<Sparse2D<T>>();
    size_t size = std::floor(rows * cols * density);

    arr->m_data.resize(size);
    arr->m_indices.resize(size);

    for (size_t i = 0; i < size; i++) arr->m_data[i] = dist(generator);

    size_t nnz = size;
    std::vector<size_t> nnz_row(rows, 0);

    size_t index = 0;
    while (nnz > 0) {
      std::uniform_int_distribution<size_t> dist_int(1, 100);//to do 50 50
      if (dist_int(generator) > 50) {
        nnz_row[index]++;
        nnz--;
      }
      index++;
      if (index >= rows) index = 0;
    }

    index = 0;
    for (size_t i : nnz_row) {
      std::vector<size_t> indice_comb;
      for (size_t j = 0; j < cols; j++) indice_comb.emplace_back(j);
      std::shuffle(indice_comb.begin(), indice_comb.end(), generator);
      for (size_t j = 0; j < i; j++) {
        arr->m_indices[index] = indice_comb[j];
        index++;
      }
    }

    if (index != arr->m_indices.size() - 1)
      std::runtime_error("Uh something is wrong");

    arr->m_row_indices.resize(rows + 1);
    arr->m_row_indices[0] = 0;
    for (size_t i = 1; i < rows + 1; i++)
      arr->m_row_indices[i] = arr->m_row_indices[i - 1] + nnz_row[i - 1];

    return arr;
  }

  std::vector<T> m_data;
  std::vector<size_t> m_info;
  std::vector<INDICE_TYPE> m_indices, m_row_indices;

 private:
  Sparse2D(Sparse2D &that) = delete;
  Sparse2D(const Sparse2D &that) = delete;
  Sparse2D(const Sparse2D &&that) = delete;
  Sparse2D &operator=(Sparse2D &that) = delete;
  Sparse2D &operator=(Sparse2D &&that) = delete;
  Sparse2D &operator=(const Sparse2D &that) = delete;
  Sparse2D &operator=(const Sparse2D &&that) = delete;
};

template <class T>
class RawSparse2D {
 public:
  using value_type = T;
  using real_type  = Sparse2D<T>;
  using raw_type   = RawSparse2D<T>;
  using raw1d_type = Sparse<T>;
  static constexpr bool is_sparse =  1;

  RawSparse2D(T * const _data, const size_t *_info, const INDICE_TYPE *_indices,
              const INDICE_TYPE *_row_indices)
      : v_data(_data), m_cols(&_info[0]), m_rows(&_info[1]), m_size(&_info[2]),
        v_indices(_indices), v_row_indices(_row_indices) {}
  RawSparse2D(RawSparse2D &&that)
      : v_data(that.v_data), m_cols(that.m_cols), m_rows(that.m_rows), m_size(that.m_size),
        v_indices(that.v_indices), v_row_indices(that.v_row_indices) {}
  T &operator[](int i) { return v_data[i]; }
  const T *data() const { return v_data; }
  T *mutable_data() { return v_data; }
  const T *row_raw(size_t i) const { return v_data + v_row_indices[i]; }
  INDICE_TYPE row_size(size_t i) const { return v_row_indices[i + 1] - v_row_indices[i]; }
  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices(size_t i) const { return v_indices + v_row_indices[i]; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }

  Sparse<T> row(size_t i) const {
    return Sparse<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                     v_indices + v_row_indices[i]);
  }

  statick::Array2D<T> toArray2D() const;

  const size_t &cols() const { return *m_cols; }
  const size_t &rows() const { return *m_rows; }
  const size_t &size() const { return *m_size; }

  T *v_data;
  const size_t *m_cols, *m_rows, *m_size;
  const INDICE_TYPE *v_indices, *v_row_indices;

 private:
  RawSparse2D() = delete;
  RawSparse2D(RawSparse2D &that) = delete;
  RawSparse2D(const RawSparse2D &that) = delete;
  RawSparse2D(const RawSparse2D &&that) = delete;
  RawSparse2D &operator=(RawSparse2D &that) = delete;
  RawSparse2D &operator=(RawSparse2D &&that) = delete;
  RawSparse2D &operator=(const RawSparse2D &that) = delete;
  RawSparse2D &operator=(const RawSparse2D &&that) = delete;
};

template <class Archive, class T>
bool load_sparse2dlist_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info,
                                     std::vector<INDICE_TYPE> &indices,
                                     std::vector<INDICE_TYPE> &row_indices) {
  size_t rows, cols, size_sparse, size;
  ar(size_sparse, rows, cols, size);

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

  s_info[0] = cols; s_info[1] = rows; s_info[2] = size_sparse;
  s_info[3] = data.size() - size_sparse;
  s_info[4] = row_indices.size() - (rows + 1);
  return true;
}

template <class T>
class Sparse2DList {
 private:
  static constexpr size_t INFO_SIZE = 5;
  static bool FROM_FILE(Sparse2DList &list, const std::string &&file) {
    std::ifstream bin_data(file, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    return statick::load_sparse2dlist_with_raw_data(
      iarchive, list.m_data, list.m_info, list.m_indices, list.m_row_indices);
  }

 public:
  Sparse2DList() {}
  RawSparse2D<T> operator[](size_t i) const {
    return RawSparse2D<T>(m_data.data() + (m_info[(i * INFO_SIZE) + 3]), &m_info[i * INFO_SIZE],
                          m_indices.data() + (m_info[(i * INFO_SIZE) + 3]),
                          m_row_indices.data() + (m_info[(i * INFO_SIZE) + 4]));
  }

  const INDICE_TYPE *indices() const { return m_indices.data(); }
  const INDICE_TYPE *row_indices() const { return m_row_indices.data(); }

  bool add_cereal(const std::string &file) { return FROM_FILE(*this, file); }

  static std::shared_ptr<Sparse2DList<T>> FROM_CEREALS(const std::vector<std::string> &&files) {
    auto array = std::make_shared<Sparse2DList<T>>();
    for (const auto &file : files)
      if (!FROM_FILE(*array.get(), file)) return nullptr;
    return std::move(array);
  }

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;
  std::vector<INDICE_TYPE> m_indices, m_row_indices;

  Sparse2DList(Sparse2DList &that) = delete;
  Sparse2DList(const Sparse2DList &that) = delete;
  Sparse2DList(Sparse2DList &&that) = delete;
  Sparse2DList(const Sparse2DList &&that) = delete;
  Sparse2DList &operator=(Sparse2DList &that) = delete;
  Sparse2DList &operator=(Sparse2DList &&that) = delete;
  Sparse2DList &operator=(const Sparse2DList &that) = delete;
  Sparse2DList &operator=(const Sparse2DList &&that) = delete;
};

template <class T>
class RawSparse2DList {
 private:
  static constexpr size_t INFO_SIZE = 5;

 public:
  RawSparse2DList(const T *data, const size_t *info, const INDICE_TYPE *_indices,
                  const INDICE_TYPE *_rows_indices)
      : v_data(data), v_info(info), v_indices(_indices), v_row_indices(_rows_indices) {}
  RawSparse2DList(std::vector<T> &data, std::vector<size_t> &info,
                  std::vector<INDICE_TYPE> &_indices, std::vector<INDICE_TYPE> &_rows_indices)
      : v_data(data.data()),
        v_info(info.data()),
        v_indices(_indices.data()),
        v_row_indices(_rows_indices.data()) {}

  RawSparse2D<T> operator[](size_t i) const {
    return RawSparse2D<T>(v_data + (v_info[(i * INFO_SIZE) + 3]), &v_info[i * INFO_SIZE],
                          v_indices + (v_info[(i * INFO_SIZE) + 3]),
                          v_row_indices + (v_info[(i * INFO_SIZE) + 4]));
  }

  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }

 private:
  const T *v_data;
  const size_t *v_info;
  const INDICE_TYPE *v_indices, *v_row_indices;

  RawSparse2DList() = delete;
  RawSparse2DList(RawSparse2DList &that) = delete;
  RawSparse2DList(const RawSparse2DList &that) = delete;
  RawSparse2DList(RawSparse2DList &&that) = delete;
  RawSparse2DList(const RawSparse2DList &&that) = delete;
  RawSparse2DList &operator=(RawSparse2DList &that) = delete;
  RawSparse2DList &operator=(RawSparse2DList &&that) = delete;
  RawSparse2DList &operator=(const RawSparse2DList &that) = delete;
  RawSparse2DList &operator=(const RawSparse2DList &&that) = delete;
};

template <typename T, typename A2D/* != is_sparse*/>
Sparse2D<T> to_sparse2d(const A2D &a2d){
  constexpr T zero {0};
  size_t _n_rows = a2d.rows(), _n_cols = a2d.cols(), nnz = 0;

  Sparse2D<T> sparse;
  auto &data = sparse.m_data;
  auto &indices = sparse.m_indices;
  auto &row_indices = sparse.m_row_indices;
  row_indices.resize(_n_rows + 1);
  row_indices[0] = 0;
  for (size_t r = 0; r < _n_rows; r++) {
    size_t nnz_row = 0;
    for (size_t c = 0; c < _n_cols; c++) {
      T val {0};
      if ((val = a2d.data()[(r * _n_cols) + c]) != zero) {
        nnz++;
        nnz_row++;
        data.push_back(val);
        indices.push_back(c);
      }
    }
    row_indices[r + 1] = row_indices[r] + nnz_row;
  }
  sparse.m_info.resize(3);
  sparse.m_info[0] = _n_cols;
  sparse.m_info[1] = _n_rows;
  sparse.m_info[2] = data.size();
  return sparse;
}

template <typename T>
statick::Sparse2D<T> statick::Array2D<T>::toSparse2D()    const { return statick::to_sparse2d<T>(*this); }
template <typename T>
statick::Sparse2D<T> statick::RawArray2D<T>::toSparse2D() const { return statick::to_sparse2d<T>(*this); }

template <typename T, typename S2D/* == is_sparse*/>
statick::Array2D<T> to_array2d(const S2D &s2d){
  size_t rows = s2d.rows(), cols = s2d.cols();
  auto &data = s2d.m_data;
  auto &indices = s2d.m_indices;
  auto &row_indices = s2d.m_row_indices;

  statick::Array2D<T> c(rows, cols);
  c.m_data.fill(0);
  auto &c_data = c.m_data.data();
  for (size_t i = 0; i < rows; i++)
    for (size_t j = row_indices[i]; j < row_indices[i + 1]; j++)
      c_data[i * cols + indices[j]] = data[j];
  return c;
}

template <typename T>
statick::Array2D<T> statick::Sparse2D<T>::toArray2D()    const { return statick::to_array2d<T>(*this); }
template <typename T>
statick::Array2D<T> statick::RawSparse2D<T>::toArray2D() const { return statick::to_array2d<T>(*this); }

}

#endif  //  TICK_ARRAY_SPARSE_ARRAY2D_HPP_
