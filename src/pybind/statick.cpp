#include <random>
#include "kul/log.hpp"
#include <Python.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "statick/pybind/numpy.hpp"

namespace py = pybind11;
using py_array_double = py::array_t<double, py::array::c_style | py::array::forcecast>;

py_array_double make_array(const py::ssize_t size) {
  return py_array_double(size);
}

statick::ArrayView<double> py_array_double_as_raw_array(py_array_double &pad){
  py::buffer_info pad_info = pad.request();
  return statick::ArrayView<double>((double*)pad_info.ptr, pad_info.shape[0]);
}

py_array_double add_arrays(py_array_double i, py_array_double j)
{
  py::buffer_info info_i = i.request(), info_j = j.request();
  if ((info_i.ndim != 1) || (info_j.ndim != 1))
    throw std::runtime_error("Number of dimensions must be one");
  if (info_i.shape[0] != info_j.shape[0])
    throw std::runtime_error("Input shapes must be equal");
  double *d_info_i_ptr = (double*)info_i.ptr, *d_info_j_ptr = (double*)info_j.ptr;
  std::vector<double> ret(info_i.shape[0]);
  for (unsigned int idx = 0; idx < info_i.shape[0]; idx++)
    ret[idx] = d_info_i_ptr[idx] + d_info_j_ptr[idx];
  return py::array(py::buffer_info(ret.data(), sizeof(double),
                   py::format_descriptor<double>::value,
                   1, info_i.shape, {sizeof(double)}));
}

using py_csr_double = py::csr_t<double, py::array::c_style | py::array::forcecast>;
void take_sparse2d(py_csr_double &v) {
  v.m_data_ptr->mutable_data()[0] = v.m_data_ptr->row(0)[0] + v.m_data_ptr->row(1)[0];
}
void save_double_sparse2d(py_csr_double &v, std::string &file) {
  statick::sparse_2d::save(*v.raw(), file);
}
void save_double_array(py_array_double &v, std::string &file) {
  statick::dense::save<double>(py_array_double_as_raw_array(v), file);
}

std::vector<int> make_vector(){
  return std::vector<int>{1, 2, 3};
}

std::tuple<std::vector<int>, std::vector<int>> make_tuple_vector(){
  return std::make_tuple(std::vector<int>{1, 2, 3}, std::vector<int>{3, 2, 1});
}

std::tuple<std::vector<int>&, std::vector<int>&> take_tuple_vector(std::tuple<std::vector<int>&, std::vector<int>&> vt){
  std::get<0>(vt)[0] = std::get<1>(vt)[0];
  return std::move(vt); // need to return or vt doesn't change
}

void print_sparse(py_csr_double &v){
  auto &s_sparse = *v.raw();
  for(size_t i = 0; i < s_sparse.size(); i++)
    std::cout << s_sparse.data()[i] << std::endl;
}

bool compare(py_array_double &dense, py_csr_double &sparse){
  using T = double;
  py::buffer_info dbinfo = dense.request();
  std::vector<size_t> dinfo(3);
  dinfo[0] = dbinfo.shape[1];
  dinfo[1] = dbinfo.shape[0];
  dinfo[2] = dbinfo.shape[0]*dbinfo.shape[1];
  statick::Array2DView<T> arr((T*)dbinfo.ptr, dinfo.data());
  auto d_sparse = arr.toSparse2D();
  auto &s_sparse = *sparse.raw();

  return ( s_sparse.size() == d_sparse.m_data.size()) &&
         (std::vector<T>(s_sparse.v_data, s_sparse.v_data + s_sparse.size()) == d_sparse.m_data) &&
         (std::vector<INDICE_TYPE>(s_sparse.v_indices, s_sparse.v_indices + s_sparse.size()) == d_sparse.m_indices) &&
         (std::vector<INDICE_TYPE>(s_sparse.v_row_indices, s_sparse.v_row_indices + s_sparse.rows() + 1) == d_sparse.m_row_indices);
}

namespace statick{
PYBIND11_MODULE(statick, m) {
  m.def("make_array", &make_array, py::return_value_policy::move);
  m.def("add_arrays", &add_arrays, "Adding two numpy arrays");
  m.def("take_sparse2d", &take_sparse2d, "take_sparse2d");
  m.def("make_vector", &make_vector, "make_vector");
  m.def("make_tuple_vector", &make_tuple_vector, "make_tuple_vector");
  m.def("take_tuple_vector", &take_tuple_vector, "take_tuple_vector");
  m.def("save_double_sparse2d", &save_double_sparse2d, "save_double_sparse2d");
  m.def("save_double_array", &save_double_array, "save_double_array");
  m.def("compare", &compare, "compare");
  m.def("print_sparse", &print_sparse, "print_sparse");
}
}
