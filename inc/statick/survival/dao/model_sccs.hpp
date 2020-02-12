#ifndef STATICK_SURVIVAL_DAO_MODEL_SCCS_H_
#define STATICK_SURVIVAL_DAO_MODEL_SCCS_H_

#define STATICK_ERROR(x) std::cerr << x << std::endl;

namespace statick {
namespace sccs {
class Exception : public kul::Exception {
 public:
  Exception(const char *f, const size_t &l, const std::string &s) : kul::Exception(f, l, s) {}
};

template <typename T, typename _F = statick::Array2D<T>, typename _L = statick::Array<int32_t>>
class DAO {
 public:
  using This = DAO<T, _F, _L>;
  using FEATURES = std::vector<std::shared_ptr<_F>>;
  using LABELS = std::vector<std::shared_ptr<_L>>;
  using FEATURE = _F;
  using LABEL = _L;
  using value_type = T;
  DAO() {}
  DAO(FEATURES &_features, LABELS &_labels) : features(_features), labels(_labels) {
    censoring.m_data = std::vector<size_t>(_features.size(), 1);
    load();
  }
  DAO &load() {
    if (features.empty() || labels.empty()) STATICK_ERROR("features or labels is empty!");
    n_lags.m_data = std::vector<size_t>(features[0]->cols(), 0);
    vars[0] = features[0]->rows(), vars[1] = features.size(), vars[2] = vars[0] * vars[1];
    vars[3] = sum(n_lags.data(), n_lags.size()) + n_lags.size(), vars[4] = n_lags.size();
    auto &m_n_intervals = vars[0], &m_n_samples = vars[1], &m_n_lagged_features = vars[3];
    if (n_lags[0] >= m_n_intervals)
      STATICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
    col_offset.m_data = std::vector<size_t>(n_lags.size(), 0);
    for (size_t i(1); i < n_lags.size(); i++) {
      if (n_lags[i] >= m_n_intervals)
        STATICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
      col_offset[i] = col_offset[i - 1] + n_lags[i - 1] + 1;
    }
    if (m_n_samples != labels.size() || m_n_samples != censoring.size())
      STATICK_ERROR("features, labels and censoring should have equal length.");
    for (size_t i(0); i < m_n_samples; i++) {
      if (features[i]->rows() != m_n_intervals)
        STATICK_ERROR("All feature matrices should have " << m_n_intervals << " rows");
      if (features[i]->cols() != m_n_lagged_features)
        STATICK_ERROR("All feature matrices should have " << m_n_lagged_features << " cols");
      if (labels[i]->size() != m_n_intervals)
        STATICK_ERROR("All labels should have " << m_n_intervals << " rows");
    }
    return *this;
  }

  inline const auto &n_intervals() const { return vars[0]; }
  inline const auto &n_samples() const { return vars[1]; }
  inline const auto &n_observations() const { return vars[2]; }
  inline const auto &n_lagged_features() const { return vars[3]; }
  inline const auto &n_features() const { return vars[4]; }

  template <class Archive>
  This &load(Archive &ar) {
    this->lip_dao.serialize(ar);
    bool sparse_features = 1;
    ar(sparse_features);
    if constexpr (_F::is_sparse) {
      if (!sparse_features) KEXCEPTION("Found dense features when expected sparse");
    } else  //   !_F::is_sparse)
        if (sparse_features)
      KEXCEPTION("Found sparse features when expected dense");

    ar(vars[1], vars[4], vars[2], vars[3], vars[0]);
    ar(n_lags, col_offset);
    for (size_t i = 0; i < vars[1]; i++) ar(*labels.emplace_back(std::make_shared<_L>()));
    for (size_t i = 0; i < vars[1]; i++) ar(*features.emplace_back(std::make_shared<_F>()));
    ar(cereal::make_nvp("censoring", this->censoring));
    return *this;
  }

  template <class Archive>
  This &save(Archive &ar) const {
    this->lip_dao.serialize(ar);
    bool sparse_features = 0;
    if constexpr (_F::is_sparse) sparse_features = true;
    ar(sparse_features);
    ar(vars[1], vars[4], vars[2], vars[3], vars[0], n_lags, col_offset);
    for (auto &l : labels) ar(*l);
    for (auto &f : features) ar(*f);
    ar(cereal::make_nvp("censoring", this->censoring));
    return *this;
  }

  FEATURES features;
  LABELS labels;
  size_t vars[5]{0};
  statick::Array<size_t> col_offset, n_lags, censoring;
  model::lipschitz::DAO<T> lip_dao;
};

template <class MODAO>
std::shared_ptr<MODAO> load_from(std::string file) {
  std::ifstream bin_data(file, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive ar(bin_data);
  auto dao = std::make_shared<MODAO>();
  ar(*dao.get());
  return dao;
}

}  // namespace sccs
}  // namespace statick
#endif  // STATICK_SURVIVAL_DAO_MODEL_SCCS_H_
