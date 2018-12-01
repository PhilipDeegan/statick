#ifndef STATICK_ARRAY_HPP_
#define STATICK_ARRAY_HPP_

#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif

#include "statick/array/math.hpp"
#include "statick/array/array.hpp"
#include "statick/array/array2d.hpp"
#include "statick/array/sparse/array.hpp"
#include "statick/array/sparse/array2d.hpp"

#endif  //  TICK_ARRAY_HPP_
