#include <atomic>
#include <chrono>
#include <random>
#include <vector>
#include <fstream>
#include <algorithm>
#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/vector.hpp"
#include "kul/cpu.hpp"
#include "statick/array.hpp"
#include "statick/linear_model/model_logreg.hpp"
#include "statick/prox/prox_l2sq.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
