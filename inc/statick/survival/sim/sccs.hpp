#ifndef STATICK_SURVIVAL_SIM_SCCS_H_
#define STATICK_SURVIVAL_SIM_SCCS_H_

//#include "gsl/gsl_randist.h"

#include "wrappy/wrappy.h"
#include "statick/survival/dao/model_sccs.hpp"

namespace statick {
namespace sccs {
namespace sim {

template <typename T, typename F_T = statick::Array2D<T>, typename L_T = statick::Array<int32_t>>
class DAO {
 public:
  using value_type = T;
};

template <typename DAO>
void simulate_multinomial_outcomes(DAO &dao /*DAO, p_features, coeffs, out*/) {
  using T = DAO::value_type;
  statick::Array<T> baseline(self.n_intervals, 0);
  // if(dao.time_drift)
  //   baseline = self.time_drift(np.arange(self.n_intervals))
  std::vector<std::vector<int32_t>> outcomes;
  for (size_t i = 0l; i < features.size()) auto dots = baseline + features[i].dot(coeffs);
  dots -= dots.max() std::vector<T> exps, probabilities;
  T sum_exp = 0;
  for (auto &dot : dots) sum_exp += exps.emplace_back(std::exp(dot));
  for (auto &exp : exps) probabilities.emplace_back(exp / sum_exp);

  std::default_random_engine generator;
  std::discrete_distribution<int> distribution(probabilities.begin(), probabilities.end());

  outcomes.emplace_back() np.random.multinomial(1, probabilities);
}
return outcomes
}  // namespace sim

}  // namespace sccs
}  // namespace statick
}  // namespace statick
#endif  // STATICK_SURVIVAL_SIM_SCCS_H_