

#ifndef STATICK_PYBIND_CSR_HPP_
#define STATICK_PYBIND_CSR_HPP_

namespace statick_py {

template <typename T>
class _PC{
 public:
  _PC(): info(4){}
  T *p_data = nullptr;
  INDICE_TYPE *p_indices = nullptr, *p_row_indices = nullptr;
  std::vector<size_t> info;
  PyArrayObject * array = nullptr, * indices = nullptr, * row_indices = nullptr;
};

template <class Archive, class T>
bool load_sparse2d_with_new_data(Archive &ar, _PC<T> &pc) {
  ar(pc.info[2], pc.info[1], pc.info[0], pc.info[3]);
  auto &info = pc.info;
  auto data_size = info[2] * sizeof(T);
  pc.p_data = reinterpret_cast<T *>(PyMem_RawMalloc(data_size));
  pc.p_indices = reinterpret_cast<INDICE_TYPE *>(PyMem_RawMalloc(info[2] * sizeof(INDICE_TYPE)));
  pc.p_row_indices = reinterpret_cast<INDICE_TYPE *>(PyMem_RawMalloc((info[1] + 1) * sizeof(INDICE_TYPE)));

  ar(cereal::binary_data(pc.p_data, sizeof(T) * info[2]));
  ar(cereal::binary_data(pc.p_indices, sizeof(INDICE_TYPE) * info[2]));
  ar(cereal::binary_data(pc.p_row_indices, sizeof(INDICE_TYPE) * (info[1] + 1)));
  return true;
}

template <typename T>
PyObject * sparse2d_to_csr(_PC<T> &pc){

  auto data_type = NPY_UINT64;
  if constexpr (std::is_same<T, double>::value)
      data_type = NPY_DOUBLE;
  else
      data_type = NPY_FLOAT;

#ifdef TICK_SPARSE_INDICES_INT64
  auto indice_type = NPY_UINT64;
#else
  auto indice_type = NPY_UINT32;
#endif

  auto info = pc.info;
  auto p_data = pc.p_data;
  auto p_indices = pc.p_indices;
  auto p_row_indices = pc.p_row_indices;

  auto *&array = pc.array;
  auto *&indices = pc.indices;
  auto *&row_indices = pc.row_indices;

  PyObject *scipy_sparse_csr = nullptr, *csr_matrix = nullptr, *matrix = nullptr;
  // auto *&scipy_sparse_csr = pc.scipy_sparse_csr;
  // auto *&csr_matrix = pc.csr_matrix;
  // auto *&matrix = pc.matrix;

  size_t cols = info[0], rows = info[1], size_sparse = info[2];
  npy_intp dims[1];  dims[0] = size_sparse;
  npy_intp rowDim[1]; rowDim[0] = rows + 1;

  array = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, data_type, p_data);
  if(!PyArray_Check(array)) throw std::runtime_error("Array check failed");

  indices = (PyArrayObject *) PyArray_SimpleNewFromData(1, dims, indice_type, p_indices);
  if(!PyArray_Check(indices)) throw std::runtime_error("indices check failed");

  row_indices = (PyArrayObject *) PyArray_SimpleNewFromData(1, rowDim, indice_type, p_row_indices);
  if(!PyArray_Check(row_indices)) throw std::runtime_error("row_indices check failed");

  if(!array) throw std::runtime_error("Array failed");
  if(!indices) throw std::runtime_error("indices failed");
  if(!row_indices) throw std::runtime_error("row_indices failed");

  PyObject* tuple = PyTuple_New(3);
  if(!tuple) throw std::runtime_error("tuple new failed");
  if(!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 1 failed");

  if(PyTuple_SetItem(tuple, 0, (PyObject *) array)) throw std::runtime_error("tuple PyTuple_SetItem 0 failed");
  if(PyTuple_SetItem(tuple, 1, (PyObject *) indices)) throw std::runtime_error("tuple PyTuple_SetItem 1 failed");
  if(PyTuple_SetItem(tuple, 2, (PyObject *) row_indices)) throw std::runtime_error("tuple PyTuple_SetItem 2 failed");
  if(!PyTuple_Check(tuple)) throw std::runtime_error("tuple type 2 failed");

  PyObject* Otuple = PyTuple_New(1);
  if(!Otuple) throw std::runtime_error("Otuple new failed");
  if(PyTuple_SetItem(Otuple, 0, (PyObject *) tuple)) throw std::runtime_error("Otuple PyTuple_SetItem 0 failed");
  if(!PyTuple_Check(tuple)) throw std::runtime_error("Otuple check failed");

  PyObject* shape = Py_BuildValue("ii", rows, cols);
  if(!shape) throw std::runtime_error("Shape tuple new failed");
  if(!PyTuple_Check(shape)) throw std::runtime_error("shape tuple check failed");

  PyObject *dic = PyDict_New();
  if(!dic) throw std::runtime_error("dict new failed");
  if(PyDict_SetItemString(dic, "shape", shape) == -1)
     throw std::runtime_error("shape set failed on dic");
  if(!PyDict_Check(dic)) throw std::runtime_error("dic is no dic");

  scipy_sparse_csr = PyImport_ImportModule("scipy.sparse.csr");
  if(!scipy_sparse_csr) throw std::runtime_error("scipy_sparse_csr failed");
  csr_matrix = PyObject_GetAttrString(scipy_sparse_csr, "csr_matrix");
  if(!csr_matrix) throw std::runtime_error("csr_matrix failed");
  if(!PyCallable_Check(csr_matrix)) throw std::runtime_error("csr_matrix check failed");
  if(!(matrix = PyObject_Call(csr_matrix, Otuple, dic))) throw std::runtime_error("matrix failed to call object");

  // csr_matrix destruction does not remove these unless set like this ¯\_(ツ)_/¯
  if(PyObject_SetAttrString(matrix, "_indices", (PyObject *)indices)) throw std::runtime_error("set indices failed");
  if(PyObject_SetAttrString(matrix, "_row_indices", (PyObject *)row_indices)) throw std::runtime_error("set row_indices failed");

  #if (NPY_API_VERSION >= 7)
  PyArray_ENABLEFLAGS(array,NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(indices,NPY_ARRAY_OWNDATA);
  PyArray_ENABLEFLAGS(row_indices,NPY_ARRAY_OWNDATA);
  #else
  PyArray_FLAGS(array) |= NPY_OWNDATA ;
  PyArray_FLAGS(indices) |= NPY_OWNDATA ;
  PyArray_FLAGS(row_indices) |= NPY_OWNDATA ;
  #endif

  Py_DECREF(array);
  Py_DECREF(indices);
  Py_DECREF(row_indices);

  return (PyObject *) matrix;
}
}

#endif  //  STATICK_PYBIND_CSR_HPP_