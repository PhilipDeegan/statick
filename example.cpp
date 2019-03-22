#include <random>
#include "kul/log.hpp"
#include "kul/signal.hpp"
#include <Python.h>
#include <pybind11/stl.h>
#include "pybind11/numpy.h"
#include "numpy/arrayobject.h"
#include "numpy/npy_math.h"
#include "cereal/archives/portable_binary.hpp"
#include "statick/array.hpp"
#include "numpy.h"

namespace py = pybind11;

template <typename T>
using py_array_t = py::array_t<T, py::array::c_style | py::array::forcecast>;

using py_array_double = py::array_t<double, py::array::c_style | py::array::forcecast>;
using py_array_int32 = py::array_t<int32_t, py::array::c_style | py::array::forcecast>;

py_array_double make_array(const py::ssize_t size) {
  return py_array_double(size);
}

statick::RawArray<double> py_array_double_as_raw_array(py_array_double &pad){
  py::buffer_info pad_info = pad.request();
  return statick::RawArray<double>((double*)pad_info.ptr, pad_info.shape[0]);
}

template <typename T, typename I>
std::shared_ptr<statick::RawArray2D<T, I>> py_array2d_double_as_raw_array2d(py_array_t<T> &pad){
  std::vector<I> info{pad.shape(0), pad.shape(1), pad.shape(0) *pad.shape(1)};
  py::buffer_info pad_info = pad.request();
  return std::move(std::make_shared<statick::RawArray2D<T, I>>((T*)pad_info.ptr, info.data()));
}
template <typename T>
std::shared_ptr<statick::RawArray<T>> py_array_double_as_raw_array_ptr(py_array_t<T> &pad){
  py::buffer_info pad_info = pad.request();
  return std::move(std::make_shared<statick::RawArray<T>>((T*)pad_info.ptr, pad_info.shape[0]));
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
  statick::sparse_2d::save_to(*v.raw(), file);
}
void save_double_array(py_array_double &v, std::string &file) {
  statick::dense::save_to<double>(py_array_double_as_raw_array(v), file);
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


#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <iostream>
#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/vector.hpp"
#include "statick/array.hpp"
#include "statick/base_model/dao/model_lipschitz.hpp"
#include "statick/survival/model_sccs.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/solver/svrg.hpp"


constexpr size_t N_FEATURES = 10000, N_ITER = 100, N_THREADS = 12;
void solve_svrg_sccs(
  std::vector<py_array_double> &_features, std::vector<py_array_int32> &_labels){

  using T        = double;
  using I        = ssize_t;
  using FEATURES = statick::RawArray2D<T, I>;
  using LABELS   = statick::RawArray<int32_t>;
  using MODAO    = statick::sccs::DAO<T, FEATURES, LABELS>;
  using MODEL    = statick::TModelSCCS<MODAO>;
  using DAO      = statick::svrg::dense::DAO<MODAO>;

  auto N_SAMPLES = _features.size();

  std::vector<std::shared_ptr<FEATURES>> features(_features.size());
  std::vector<std::shared_ptr<LABELS>>  labels(_labels.size());
  for (size_t i = 0; i < N_SAMPLES; ++i)
    features[i] = py_array2d_double_as_raw_array2d<double, I>(_features[i]);

  for (size_t i = 0; i < N_SAMPLES; ++i)
    labels[i] = py_array_double_as_raw_array_ptr(_labels[i]);

  MODAO madao(features, labels);
  const size_t n_samples = madao.n_samples();
#include "test/random_seq.ipp"
  const double STRENGTH = (1. / N_SAMPLES) + 1e-10;
  auto call = [&](const T *coeffs, T step, T *out, size_t size) {
    statick::prox_l2sq::call(coeffs, step, out, size, STRENGTH);
  };
  DAO dao(madao, N_ITER, N_SAMPLES, N_THREADS);
  statick::svrg::dense::solve<MODEL>(dao, madao, call, next_i);
}

PYBIND11_MODULE(example, m) {
  m.def("make_array", &make_array, py::return_value_policy::move);
  m.def("add_arrays", &add_arrays, "Adding two numpy arrays");
  m.def("take_sparse2d", &take_sparse2d, "take_sparse2d");
  m.def("make_vector", &make_vector, "make_vector");
  m.def("make_tuple_vector", &make_tuple_vector, "make_tuple_vector");
  m.def("take_tuple_vector", &take_tuple_vector, "take_tuple_vector");
  m.def("save_double_sparse2d", &save_double_sparse2d, "save_double_sparse2d");
  m.def("save_double_array", &save_double_array, "save_double_array");
  m.def("solve_svrg_sccs", &solve_svrg_sccs, "solve_svrg_sccs");
}
