

#ifndef STATICK_PYBIND_CSR_HPP_
#define STATICK_PYBIND_CSR_HPP_

namespace statick_py {

template <typename T>
struct Sparse2dState {
  T *p_data = nullptr;
  INDICE_TYPE *p_indices = nullptr, *p_row_indices = nullptr;
  size_t info[4];
};

template <class Archive, class T>
bool load_sparse2d_with_new_data(Archive &ar, Sparse2dState<T> &state) {
  auto &info = state.info;
  ar(info[2], info[1], info[0], info[3]);
  state.p_data = reinterpret_cast<T *>(PyMem_RawMalloc(info[2] * sizeof(T)));
  state.p_indices = reinterpret_cast<INDICE_TYPE *>(PyMem_RawMalloc(info[2] * sizeof(INDICE_TYPE)));
  state.p_row_indices =
      reinterpret_cast<INDICE_TYPE *>(PyMem_RawMalloc((info[1] + 1) * sizeof(INDICE_TYPE)));
  ar(cereal::binary_data(state.p_data, sizeof(T) * info[2]));
  ar(cereal::binary_data(state.p_indices, sizeof(INDICE_TYPE) * info[2]));
  ar(cereal::binary_data(state.p_row_indices, sizeof(INDICE_TYPE) * (info[1] + 1)));
  return true;
}

template <typename T>
PyObject *sparse2d_to_csr(Sparse2dState<T> &state) {
  KLOG(TRC);

  auto data_type = pybind11::detail::npy_api::NPY_UINT64_;
  if constexpr (std::is_same<T, double>::value)
    data_type = pybind11::detail::npy_api::NPY_DOUBLE_;
  else if constexpr (std::is_same<T, float>::value)
    data_type = pybind11::detail::npy_api::NPY_FLOAT_;
  else
    KEXCEPT(mkn::kul::Exception, "Unhandeled data type for csr_matrix");

#ifdef TICK_SPARSE_INDICES_INT64
  auto indice_type = pybind11::detail::npy_api::NPY_UINT64_;
#else
  auto indice_type = pybind11::detail::npy_api::NPY_UINT32_;
#endif

  auto info = state.info;
  size_t cols = info[0], rows = info[1], size_sparse = info[2];
  Py_intptr_t dims[1];
  dims[0] = size_sparse;
  Py_intptr_t rowDim[1];
  rowDim[0] = rows + 1;

  py_array_t<T> py_array{pybind11::buffer_info(state.p_data, sizeof(T), pybind11::format_descriptor<T>::value,
                                   1, dims, {sizeof(T)})};
  PyObject *array = py_array.release().ptr();
  if (!array) throw std::runtime_error("Array failed");

  // PyArrayObject *indices =
  //     (PyArrayObject *)PyArray_SimpleNewFromData(1, dims, indice_type, state.p_indices);

  py_array_t<INDICE_TYPE> py_indices{pybind11::buffer_info(state.p_indices, sizeof(INDICE_TYPE), pybind11::format_descriptor<INDICE_TYPE>::value,
                                   1, dims, {sizeof(INDICE_TYPE)})};
  PyObject *indices = py_indices.release().ptr();
  if (!indices) throw std::runtime_error("indices failed");
  // if (!PyArray_Check(indices)) throw std::runtime_error("indices check failed");

  // PyArrayObject *row_indices =
  //     (PyArrayObject *)PyArray_SimpleNewFromData(1, rowDim, indice_type, state.p_row_indices);
  py_array_t<INDICE_TYPE> py_row_indices{pybind11::buffer_info(state.p_row_indices, sizeof(INDICE_TYPE), pybind11::format_descriptor<INDICE_TYPE>::value,
                                   1, rowDim, {sizeof(INDICE_TYPE)})};
  PyObject *row_indices = py_row_indices.release().ptr();
  if (!row_indices) throw std::runtime_error("row_indices failed");
  // if (!PyArray_Check(row_indices)) throw std::runtime_error("row_indices check failed");


  PyObject *tuple = PyTuple_New(3);
  if (!tuple) throw std::runtime_error("tuple new failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 1 failed");

  if (PyTuple_SetItem(tuple, 0, (PyObject *)array))
    throw std::runtime_error("tuple PyTuple_SetItem 0 failed");
  if (PyTuple_SetItem(tuple, 1, (PyObject *)indices))
    throw std::runtime_error("tuple PyTuple_SetItem 1 failed");
  if (PyTuple_SetItem(tuple, 2, (PyObject *)row_indices))
    throw std::runtime_error("tuple PyTuple_SetItem 2 failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 2 failed");

  PyObject *Otuple = PyTuple_New(1);
  if (!Otuple) throw std::runtime_error("Otuple new failed");
  if (PyTuple_SetItem(Otuple, 0, (PyObject *)tuple))
    throw std::runtime_error("Otuple PyTuple_SetItem 0 failed");
  if (!PyTuple_Check(tuple)) throw std::runtime_error("Otuple check failed");

  PyObject *shape = Py_BuildValue("ii", rows, cols);
  if (!shape) throw std::runtime_error("Shape tuple new failed");
  if (!PyTuple_Check(shape)) throw std::runtime_error("shape tuple check failed");

  PyObject *dic = PyDict_New();
  if (!dic) throw std::runtime_error("dict new failed");
  if (PyDict_SetItemString(dic, "shape", shape) == -1)
    throw std::runtime_error("shape set failed on dic");
  if (!PyDict_Check(dic)) throw std::runtime_error("dic is no dic");

  PyObject *scipy_sparse_csr = PyImport_ImportModule("scipy.sparse.csr");
  if (!scipy_sparse_csr) throw std::runtime_error("scipy_sparse_csr failed");
  PyObject *csr_matrix = PyObject_GetAttrString(scipy_sparse_csr, "csr_matrix");
  if (!csr_matrix) throw std::runtime_error("csr_matrix failed");
  if (!PyCallable_Check(csr_matrix)) throw std::runtime_error("csr_matrix check failed");
  PyObject *matrix = PyObject_Call(csr_matrix, Otuple, dic);
  if (!matrix) throw std::runtime_error("matrix failed to call object");

  // csr_matrix destruction does not remove these unless set like this ¯\_(ツ)_/¯
  if (PyObject_SetAttrString(matrix, "_indices", (PyObject *)indices))
    throw std::runtime_error("set indices failed");
  if (PyObject_SetAttrString(matrix, "_row_indices", (PyObject *)row_indices))
    throw std::runtime_error("set row_indices failed");

#if (NPY_API_VERSION >= 7)
  // PyArray_ENABLEFLAGS(array, pybind11::detail::npy_api::NPY_ARRAY_OWNDATA_);
  // PyArray_ENABLEFLAGS(indices, pybind11::detail::npy_api::NPY_ARRAY_OWNDATA_);
  // PyArray_ENABLEFLAGS(row_indices, pybind11::detail::npy_api::NPY_ARRAY_OWNDATA_);
#else
  // PyArray_FLAGS(array) |= NPY_OWNDATA;
  // PyArray_FLAGS(indices) |= NPY_OWNDATA;
  // PyArray_FLAGS(row_indices) |= NPY_OWNDATA;
#endif

  Py_DECREF(array);
  Py_DECREF(indices);
  Py_DECREF(row_indices);

  return matrix;
}
}  // namespace statick_py

#endif  //  STATICK_PYBIND_CSR_HPP_