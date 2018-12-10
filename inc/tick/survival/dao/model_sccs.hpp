#ifndef TICK_SURVIVAL_DAO_MODEL_SCCS_H_
#define TICK_SURVIVAL_DAO_MODEL_SCCS_H_

#define TICK_ERROR(x) std::cout << x << std::endl;

namespace tick {
namespace sccs {

template <typename T = double>
class DAO {
 public:
  DAO() {}
  DAO(size_t _samples, size_t _features)
      : features(_samples), labels(_samples), censoring(_samples), n_lags(_features, 0) {}

  DAO &load() {
    vars[0] = features[0]->n_rows();
    vars[1] = features->size();
    vars[2] = vars[0] * vars[1];
    vars[3] = sum(n_lags.data(), n_lags.size()) + n_lags.size();
    vars[4] = n_lags.size();

    auto &m_n_intervals = vars[0];
    auto &m_n_samples = vars[1];
    auto &m_n_observations = vars[2];
    auto &m_n_lagged_features = vars[3];
    auto &m_n_features = vars[4];

    if (n_lags[0] >= m_n_intervals) {
      TICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
    }
    col_offset = std::vector<size_t>(n_lags.size(), 0);
    for (size_t i(1); i < n_lags.size(); i++) {
      if (n_lags[i] >= m_n_intervals) {
        TICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
      }
      col_offset[i] = col_offset[i - 1] + n_lags[i - 1] + 1;
    }

    if (m_n_samples != labels.size() || m_n_samples != censoring.size())
      TICK_ERROR("features, labels and censoring should have equal length.");

    for (size_t i(0); i < m_n_samples; i++) {
      if (features[i]->n_rows() != m_n_intervals)
        TICK_ERROR("All feature matrices should have " << m_n_intervals << " rows");

      if (features[i]->n_cols() != m_n_lagged_features)
        TICK_ERROR("All feature matrices should have " << m_n_lagged_features << " cols");

      if (labels[i]->size() != m_n_intervals)
        TICK_ERROR("All labels should have " << m_n_intervals << " rows");
    }
    return *this;
  }

  size_t vars[5];
  SharedArray2DList<T> features;
  SharedArrayList<size_t> labels;
  std::vector<size_t> censoring, col_offset, n_lags;
  // m_n_intervals // m_n_samples // m_n_observations;
  // m_n_lagged_features        // m_n_features;

  inline const size_t &n_intervals() const { return vars[0]; }
  inline const size_t &n_samples() const { return vars[1]; }
  inline const size_t &n_observations() const { return vars[2]; }
  inline const size_t &n_lagged_features() const { return vars[3]; }
  inline const size_t &n_features() const { return vars[4]; }
};
}  // namespace sccs
}  // namespace tick
#endif  // TICK_SURVIVAL_DAO_MODEL_SCCS_H_
