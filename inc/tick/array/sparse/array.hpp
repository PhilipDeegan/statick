#ifndef TICK_ARRAY_ARRAY_HPP_
#define TICK_ARRAY_ARRAY_HPP_

namespace tick {

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

}  // namespace tick

#endif  //  TICK_ARRAY_ARRAY_HPP_
