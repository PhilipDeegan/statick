
#ifndef STATICK_PYBIND_LINEAR_MODEL_HPP_
#define STATICK_PYBIND_LINEAR_MODEL_HPP_

#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
#include "statick/linear_model/model_logreg.hpp"

namespace statick_py {
namespace logreg {
template <typename T, typename DAO>
void fit_d(DAO &dao, py_array_t<T> & a, py_array_t<T> & b){
  pybind11::buffer_info a_info = a.request(), b_info = b.request();
  std::vector<size_t> ainfo(3);
  ainfo[0] = a_info.shape[1];
  ainfo[1] = a_info.shape[0];
  ainfo[2] = a_info.shape[0]*a_info.shape[1];
  dao.m_features = std::make_shared<statick::Array2DView<T>>((T*)a_info.ptr, ainfo.data());
  dao.m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
}

template <typename T, typename DAO>
void fit_s(DAO &dao, py_csr_t<T> & a, py_array_t<T> & b){
  dao.X = a;
  dao.y = b;
  pybind11::buffer_info b_info = dao.y.request();
  dao.m_features = dao.X.m_data_ptr;
  KLOG(INF) << statick::sum(dao.X.m_data_ptr->data(), dao.X.m_data_ptr->size());
  dao.m_labels = std::make_shared<statick::ArrayView<T>>((T*)b_info.ptr, b_info.shape[0]);
}

template <typename _F, typename _L>
class DAO : public statick::logreg::DAO<_F, _L> {
 public:
  using SUPER = statick::logreg::DAO<_F, _L>;
  using FEATURES = typename SUPER::FEATURES;
  using FEATURE = typename SUPER::FEATURE;
  using LABELS = typename SUPER::LABELS;
  using LABEL = typename SUPER::LABEL;
  using T = typename SUPER::T;

  using X_TYPE = typename std::conditional<FEATURE::is_sparse, py_csr_t<T>, py_array_t<T>>::type;

  template<typename F = FEATURE, typename = typename std::enable_if<!F::is_sparse>::type >
  DAO(py_array_t<T> & a, py_array_t<T> & b){ fit_d(*this, a, b); }
  template<typename F = FEATURE, typename = typename std::enable_if< F::is_sparse>::type >
  DAO(py_csr_t<T> & a, py_array_t<T> & b)  { fit_s(*this, a, b); }

  X_TYPE X;
  py_array_t<T> y;
};
}  /* namespace logreg*/

template <typename _F, typename _L>
class ModelLogReg : public statick::ModelLogReg<_F, _L, logreg::DAO<_F, _L>> {
 public:
  using SUPER = statick::ModelLogReg<_F, _L, logreg::DAO<_F, _L>>;
  using DAO = logreg::DAO<_F, _L>;
  using SUPER::NAME;
};
}  /* namespace statick_py*/
#endif  /* STATICK_PYBIND_LINEAR_MODEL_HPP_ */