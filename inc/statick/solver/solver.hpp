#ifndef STATICK_SOLVER_SOLVER_HPP_
#define STATICK_SOLVER_SOLVER_HPP_

#include "statick/random.hpp"

namespace statick {

class Solver {
 public:
  template <typename MODEL, bool INTERCEPT = false>
  using DAO = typename std::conditional<MODEL::DAO::FEATURE::is_sparse, statick::saga::sparse::DAO<typename MODEL::DAO, INTERCEPT>, statick::saga::dense::DAO<typename MODEL::DAO, INTERCEPT>>::type;


};
}  // namespace statick
#endif  // STATICK_SOLVER_SOLVER_HPP_
