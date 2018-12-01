
namespace statick {
namespace hcc {

template <class Archive, class T>
void load_array_with_raw_data(Archive& ar, T* data) {
  bool is_sparse = false;
  ar(CEREAL_NVP(is_sparse));
  ulong vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));
  ar(cereal::binary_data(data, static_cast<std::size_t>(vectorSize) * sizeof(T)));
  // if (is_sparse) STATICK_ERROR("Deserializing sparse arrays is not supported yet.");
}

template <class Archive>
void load_array_info_only(Archive& ar, size_t* s_info) {
  bool is_sparse = false;
  ar(CEREAL_NVP(is_sparse));
  ulong vectorSize = 0;
  ar(cereal::make_size_tag(vectorSize));
  s_info[0] = vectorSize;
}

template <class Archive, class T>
void load_sparse2d_with_raw_data(Archive& ar, std::vector<T>& data, std::vector<size_t>& info,
                                 std::vector<INDICE_TYPE>& indices,
                                 std::vector<INDICE_TYPE>& row_indices) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse);
  ar(rows);
  ar(cols);
  ar(size);

  data.resize(data.size() + size_sparse);
  info.resize(info.size() + 5);
  indices.resize(indices.size() + size_sparse);
  row_indices.resize(row_indices.size() + rows + 1);

  T* s_data = &data[data.size()] - size_sparse;
  size_t* s_info = &info[info.size()] - 5;
  INDICE_TYPE* s_indices = &indices[indices.size()] - size_sparse;
  INDICE_TYPE* s_row_indices = &row_indices[row_indices.size()] - (rows + 1);

  ar(cereal::binary_data(s_data, sizeof(T) * size_sparse));
  ar(cereal::binary_data(s_indices, sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(s_row_indices, sizeof(INDICE_TYPE) * (rows + 1)));

  s_info[0] = cols;
  s_info[1] = rows;
  s_info[2] = size_sparse;
  s_info[3] = data.size() - size_sparse;
  s_info[4] = row_indices.size() - (rows + 1);
}

template <class Archive, class T>
void load_raw_sparse2d_with_raw_data(Archive& ar, T* data, size_t* info, INDICE_TYPE* indices,
                                     INDICE_TYPE* row_indices) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse);
  ar(rows);
  ar(cols);
  ar(size);

  ar(cereal::binary_data(data, sizeof(T) * size_sparse));
  ar(cereal::binary_data(indices, sizeof(INDICE_TYPE) * size_sparse));
  ar(cereal::binary_data(row_indices, sizeof(INDICE_TYPE) * (rows + 1)));

  info[0] = cols;
  info[1] = rows;
  info[2] = size_sparse;
}

template <class Archive>
void load_sparse2d_info_only(Archive& ar, size_t* s_info) {
  size_t rows = 0, cols = 0, size_sparse, size = 0;
  ar(size_sparse);
  ar(rows);
  ar(cols);
  ar(size);

  s_info[0] = cols;
  s_info[1] = rows;
  s_info[2] = size_sparse;
}

}  // namespace hcc
}  // namespace statick
