#ifndef TICK_ARRAY_ARRAY_HPP_
#define TICK_ARRAY_ARRAY_HPP_

namespace tick {
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

template <typename T>
class Array {
 public:
  Array() {}
  Array(size_t size) : m_data(size) {}
  Array(Array &&that) : m_data(that.m_data) {}

  const T *data() const { return m_data.data(); }
  const size_t &size() const { return m_data.size(); }
  const T &operator[](int i) const { return m_data[i]; }

  T dot(const T *const that) const { return dot(this->m_data.data(), that.m_data.data(), this->m_data.size()); }
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

  static std::shared_ptr<Array<T>> RANDOM(size_t size, T seed = -1) {
    auto arr = std::make_shared<Array<T>>(size);
    std::random_device r;
    std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
    std::mt19937_64 generator(seed_seq);
    if constexpr (std::is_floating_point<T>::value) {
      std::uniform_real_distribution<T> uniform_dist;
      for (size_t i = 0; i < size; i++) arr->m_data[i] = uniform_dist(generator);
    } else {
      std::uniform_int_distribution<T> uniform_dist;
      for (size_t i = 0; i < size; i++) arr->m_data[i] = uniform_dist(generator);
    }
    return arr;
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

template <typename T>
class RawArray {
 public:
  RawArray(const T *_data, const size_t _size) : _size(_size), v_data(_data) {}
  RawArray(RawArray &&that) : _size(that._size), v_data(that.v_data) {}

  const T *data() const { return v_data; }
  const size_t &size() const { return _size; }
  const T &operator[](int i) const { return v_data[i]; }

  T dot(const T *const that) const { return tick::dot(this->v_data, that, _size); }
  T dot(const RawArray<T> &that) const { return dot(that.v_data); }

 private:
  const size_t _size;
  const T *v_data;
  RawArray() = delete;
  RawArray(RawArray &that) = delete;
  RawArray(const RawArray &that) = delete;
  RawArray(const RawArray &&that) = delete;
  RawArray &operator=(RawArray &that) = delete;
  RawArray &operator=(RawArray &&that) = delete;
  RawArray &operator=(const RawArray &that) = delete;
  RawArray &operator=(const RawArray &&that) = delete;
};

template <class Archive, class T>
bool load_arraylist_with_raw_data(Archive &ar, std::vector<T> &data, std::vector<size_t> &info) {
  bool is_sparse = false;
  ar(CEREAL_NVP(is_sparse));
  if (is_sparse) return false;
  size_t vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));
  if (data.size() < vectorSize) data.resize(vectorSize);
  ar(cereal::binary_data(data.data(), static_cast<std::size_t>(vectorSize) * sizeof(T)));
  info.resize(info.size() + 2);
  size_t *s_info = &info[info.size()] - 2;
  s_info[0] = vectorSize;
  s_info[1] = data.size() - vectorSize;
  return true;
}

template <typename T>
class ArrayList {
 private:
  static constexpr size_t INFO_SIZE = 2;
  static bool FROM_FILE(ArrayList<T> &list, const std::string &&file) {
    std::ifstream bin_data(file, std::ios::in | std::ios::binary);
    cereal::PortableBinaryInputArchive iarchive(bin_data);
    return tick::load_arraylist_with_raw_data(iarchive, list.m_data, list.m_info);
  }

 public:
  ArrayList() {}
  RawArray<T> operator[](size_t i) const {
    return RawArray<T>(m_data.data() + (m_info[(i * INFO_SIZE) + 1]), m_info[i * INFO_SIZE]);
  }

  const size_t *info() const { return m_info.data(); }
  const T *data() const { return m_data.data(); }
  size_t size() const { return m_data.size(); }

  bool add_cereal(const std::string &file) { return FROM_FILE(*this, file); }

  static std::shared_ptr<ArrayList<T>> FROM_CEREALS(const std::vector<std::string> &&files) {
    auto array = std::make_shared<ArrayList<T>>();
    for (const auto &file : files)
      if (!FROM_FILE(*array.get(), file)) return nullptr;
    return std::move(array);
  }

 private:
  std::vector<T> m_data;
  std::vector<size_t> m_info;

  ArrayList(ArrayList &that) = delete;
  ArrayList(const ArrayList &that) = delete;
  ArrayList(ArrayList &&that) = delete;
  ArrayList(const ArrayList &&that) = delete;
  ArrayList &operator=(ArrayList &that) = delete;
  ArrayList &operator=(ArrayList &&that) = delete;
  ArrayList &operator=(const ArrayList &that) = delete;
  ArrayList &operator=(const ArrayList &&that) = delete;
};

template <typename T, typename ARRAY = Array<T>>
class SharedArrayList {
 public:
  SharedArrayList() {}
  SharedArrayList(size_t size) : m_data(size) {}

  size_t size() const { return m_data.size(); }

  void push_back(std::shared_ptr<ARRAY> arr) { m_data.push_back(arr); }
  void add_at(std::shared_ptr<ARRAY> arr, size_t i) { m_data[i] = arr; }

  auto &operator[](size_t index) { return m_data[index]; }

 private:
  std::vector<std::shared_ptr<ARRAY>> m_data;

  SharedArrayList(SharedArrayList &that) = delete;
  SharedArrayList(const SharedArrayList &that) = delete;
  SharedArrayList(SharedArrayList &&that) = delete;
  SharedArrayList(const SharedArrayList &&that) = delete;
  SharedArrayList &operator=(SharedArrayList &that) = delete;
  SharedArrayList &operator=(SharedArrayList &&that) = delete;
  SharedArrayList &operator=(const SharedArrayList &that) = delete;
  SharedArrayList &operator=(const SharedArrayList &&that) = delete;
};
template <typename T>
using SharedRawArrayList = SharedArrayList<T, RawArray<T>>;

}  // namespace tick

#endif  //  TICK_ARRAY_ARRAY_HPP_
