#ifndef STATICK_ARRAY_HPP_
#define STATICK_ARRAY_HPP_

#include <memory>
#include <vector>
#include <random>

#ifndef INDICE_TYPE
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#endif

namespace statick {
template<class T>
struct is_shared_ptr : std::false_type {};
template<class T>
struct is_shared_ptr<std::shared_ptr<T>> : std::true_type {};
}

#include "cereal/types/vector.hpp"
#include "cereal/archives/portable_binary.hpp"

#include "statick/array/math.hpp"
#include "statick/array/array.hpp"
#include "statick/array/array2d.hpp"
#include "statick/array/sparse/array.hpp"
#include "statick/array/sparse/array2d.hpp"

namespace statick {
#define AR_P_T(C, n)                                         \
  using arrayv_##n          = statick::ArrayView<C>;         \
  using arrayv_##n##_ptr    = std::shared_ptr<arrayv_##n>;   \
  using array2dv_##n        = statick::Array2DView<C>;       \
  using array2dv_##n##_ptr  = std::shared_ptr<array2dv_##n>; \
  using sparse2dv_##n       = statick::Sparse2DView<C>;      \
  using sparse2dv_##n##_ptr = std::shared_ptr<sparse2dv_##n>;
AR_P_T(double, d)
AR_P_T(float, s)
#undef AR_P_T
}

#endif  //  STATICK_ARRAY_HPP_
