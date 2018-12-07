#ifndef TICK_ARRAY_HPP_
#define TICK_ARRAY_HPP_

#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif

#include "tick/array/math.hpp"
#include "tick/array/array.hpp"
#include "tick/array/array2d.hpp"
#include "tick/array/sparse/array.hpp"
#include "tick/array/sparse/array2d.hpp"

#endif  //  TICK_ARRAY_HPP_
