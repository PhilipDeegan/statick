#include <chrono>
#include <random>
#include <fstream>
#include <algorithm>
#include "mkn/kul/cpu.hpp"
#include "statick/array.hpp"
#include "statick/prox/prox_l2sq.hpp"
#include "statick/linear_model/model_logreg.hpp"
#define NOW                                                \
  std::chrono::duration_cast<std::chrono::milliseconds>(   \
      std::chrono::system_clock::now().time_since_epoch()) \
      .count()
