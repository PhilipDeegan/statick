#include "statick/array.hpp"
#include "statick/pybind/def.hpp"
namespace py = pybind11;

namespace statick {
PYBIND11_MODULE(statick_array, m) {
  m.attr("__name__") = "statick.array";
}
}  //  namespace statick
