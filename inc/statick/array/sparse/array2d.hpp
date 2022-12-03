#ifndef STATICK_ARRAY_SPARSE_ARRAY2D_HPP_
#define STATICK_ARRAY_SPARSE_ARRAY2D_HPP_

namespace statick {
namespace sparse_2d {

class Exception : public mkn::kul::Exception {
 public:
  Exception(const char *f, const size_t &l, const std::string &s) : mkn::kul::Exception(f, l, s) {}
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
  ar(s2d.size(), s2d.rows(), s2d.cols(), s2d.cols() * s2d.rows());
  inner_save<typename S2D::value_type>(ar, s2d);
}
template <class S2D>
void save(const S2D &s2d, const std::string &_file) {
  std::ofstream ss(_file, std::ios::out | std::ios::binary);
  cereal::PortableBinaryOutputArchive ar(ss);
  save_to(ar, s2d);
}
}  // namespace sparse_2d

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

template <class T>
class Sparse2DView;

template <class T>
class Sparse2D;

template <typename T, typename A2D /* != is_sparse*/>
Sparse2D<T> to_sparse2d(const A2D &a2d);

template <class T>
class Sparse2D {
 public:
  using value_type = T;
  using real_type = Sparse2D<T>;
  using view_type = Sparse2DView<T>;
  using view1d_type = SparseView<T>;
  static constexpr bool is_sparse = 1;

  template <typename T1, typename A2D /* != is_sparse*/>
  friend Sparse2D<T1> to_sparse2d(const A2D &a2d);

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

  auto const &data_vector() const { return m_data; }
  auto const &indices_vector() const { return m_indices; }
  auto const &row_indices_vector() const { return m_row_indices; }

  SparseView<T> row(size_t i) const {
    return SparseView<T>(m_data.data() + m_row_indices[i], m_row_indices[i + 1] - m_row_indices[i],
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

  template <class Archive>
  void load(Archive &ar) {
    load_sparse2d_with_raw_data(ar, m_data, m_info, m_indices, m_row_indices);
  }
  template <class Archive>
  void save(Archive &ar) const {
    sparse_2d::save<T>(ar, *this);
  }

  std::shared_ptr<view_type> as_view();

  statick::Array2D<T> toArray2D() const;
  
  // Credit for this function goes to: https://github.com/andro2157 
  static std::shared_ptr<Sparse2D<T>> RANDOM(size_t rows, size_t cols, T density, T seed = -1) {
    if (density < 0 || density > 1)
      KEXCEPT(sparse_2d::Exception, "Invalid sparse density, must be between 0 and 1");

    std::mt19937_64 generator;
    if (seed > 0)
      generator = std::mt19937_64(seed);
    else {
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
      std::uniform_int_distribution<size_t> dist_int(1, 100);  // to do 50 50
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
      for (size_t j = 0; j < i; j++) arr->m_indices[index++] = indice_comb[j];
    }

    if (index != arr->m_indices.size() - 1) std::runtime_error("Uh something is wrong");

    arr->m_row_indices.resize(rows + 1);
    arr->m_row_indices[0] = 0;
    for (size_t i = 1; i < rows + 1; i++)
      arr->m_row_indices[i] = arr->m_row_indices[i - 1] + nnz_row[i - 1];

    return arr;
  }

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;
  std::vector<INDICE_TYPE> m_indices, m_row_indices;
};

template <class T>
class Sparse2DView {
 public:
  using value_type = T;
  using real_type = Sparse2D<T>;
  using view_type = Sparse2DView<T>;
  using view1d_type = SparseView<T>;
  static constexpr bool is_sparse = 1;

  Sparse2DView(T *_data, size_t *_info, INDICE_TYPE *_indices, INDICE_TYPE *_row_indices)
      : v_data(_data), v_indices(_indices), v_row_indices(_row_indices) {
    set_info(_info);
  }
  Sparse2DView(Sparse2DView &&that)
      : v_data(that.v_data), v_indices(that.v_indices), v_row_indices(that.v_row_indices) {
    set_info(that.m_info);
  }
  T &operator[](int i) { return v_data[i]; }

  const T *data() const { return v_data; }

  T *mutable_data() { return v_data; }
  const T *row_raw(size_t i) const { return v_data + v_row_indices[i]; }
  INDICE_TYPE row_size(size_t i) const { return v_row_indices[i + 1] - v_row_indices[i]; }
  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices(size_t i) const { return v_indices + v_row_indices[i]; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }

  SparseView<T> row(size_t i) const {
    return SparseView<T>(v_data + v_row_indices[i], v_row_indices[i + 1] - v_row_indices[i],
                         v_indices + v_row_indices[i]);
  }

  void set_info(const size_t *_info) {
    m_info[0] = _info[0];
    m_info[1] = _info[1];
    m_info[2] = _info[2];
  }

  statick::Array2D<T> toArray2D() const;

  const size_t &cols() const { return m_info[0]; }
  const size_t &rows() const { return m_info[1]; }
  const size_t &size() const { return m_info[2]; }

 private:
  T *v_data = nullptr;
  size_t m_info[3] = {0, 0, 0};
  INDICE_TYPE *v_indices = nullptr, *v_row_indices = nullptr;
};

template <class T>
std::shared_ptr<Sparse2DView<T>> Sparse2D<T>::as_view() {
  std::vector<size_t> info = {cols(), rows(), size()};
  return std::make_shared<Sparse2DView<T>>(&m_data[0], info.data(), &m_indices[0],
                                           &m_row_indices[0]);
}

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

  s_info[0] = cols;
  s_info[1] = rows;
  s_info[2] = size_sparse;
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
    return statick::load_sparse2dlist_with_raw_data(iarchive, list.m_data, list.m_info,
                                                    list.m_indices, list.m_row_indices);
  }

 public:
  Sparse2DList() {}
  Sparse2DView<T> operator[](size_t i) const {
    return Sparse2DView<T>(m_data.data() + (m_info[(i * INFO_SIZE) + 3]), &m_info[i * INFO_SIZE],
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
};

template <class T>
class Sparse2DViewList {
 private:
  static constexpr size_t INFO_SIZE = 5;

 public:
  Sparse2DViewList(const T *data, const size_t *info, const INDICE_TYPE *_indices,
                   const INDICE_TYPE *_rows_indices)
      : v_data(data), v_info(info), v_indices(_indices), v_row_indices(_rows_indices) {}
  Sparse2DViewList(std::vector<T> &data, std::vector<size_t> &info,
                   std::vector<INDICE_TYPE> &_indices, std::vector<INDICE_TYPE> &_rows_indices)
      : v_data(data.data()),
        v_info(info.data()),
        v_indices(_indices.data()),
        v_row_indices(_rows_indices.data()) {}

  Sparse2DView<T> operator[](size_t i) const {
    return Sparse2DView<T>(v_data + (v_info[(i * INFO_SIZE) + 3]), &v_info[i * INFO_SIZE],
                           v_indices + (v_info[(i * INFO_SIZE) + 3]),
                           v_row_indices + (v_info[(i * INFO_SIZE) + 4]));
  }

  const INDICE_TYPE *indices() const { return v_indices; }
  const INDICE_TYPE *row_indices() const { return v_row_indices; }

 private:
  const T *v_data;
  const size_t *v_info;
  const INDICE_TYPE *v_indices, *v_row_indices;
};

template <typename T, typename A2D /* != is_sparse*/>
Sparse2D<T> to_sparse2d(const A2D &a2d) {
  constexpr T zero{0};
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
      T val{0};
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
statick::Sparse2D<T> statick::Array2D<T>::toSparse2D() const {
  return statick::to_sparse2d<T>(*this);
}
template <typename T>
statick::Sparse2D<T> statick::Array2DView<T>::toSparse2D() const {
  return statick::to_sparse2d<T>(*this);
}

template <typename T, typename S2D /* == is_sparse*/>
statick::Array2D<T> to_array2d(const S2D &s2d) {
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
statick::Array2D<T> statick::Sparse2D<T>::toArray2D() const {
  return statick::to_array2d<T>(*this);
}
template <typename T>
statick::Array2D<T> statick::Sparse2DView<T>::toArray2D() const {
  return statick::to_array2d<T>(*this);
}

}  // namespace statick

#endif  //  STATICK_ARRAY_SPARSE_ARRAY2D_HPP_
