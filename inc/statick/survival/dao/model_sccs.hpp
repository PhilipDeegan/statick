#ifndef TICK_SURVIVAL_DAO_MODEL_SCCS_H_
#define TICK_SURVIVAL_DAO_MODEL_SCCS_H_

#define TICK_ERROR(x) std::cerr << x << std::endl;

namespace statick {
namespace sccs {

template <typename T, typename _F_ = statick::Array2D<T>, typename _L_ = statick::Array<int32_t>>
class DAO {
 public:
  using value_type = T;
  using FEATURES = std::vector<std::shared_ptr<_F_>>;
  using LABELS = std::vector<std::shared_ptr<_L_>>;
  DAO(){}
  DAO(FEATURES &_features, LABELS &_labels) : features(_features), labels(_labels){
    censoring.m_data = std::vector<size_t>(_features.size(), 1);
    load();
  }
 DAO &load(){
    if(features.empty() || labels.empty()) TICK_ERROR("features or labels is empty!");
    n_lags.m_data = std::vector<size_t>(features[0]->cols(), 0);
    vars[0] = features[0]->rows(), vars[1] = features.size(), vars[2] = vars[0] * vars[1];
    vars[3] = sum(n_lags.data(), n_lags.size()) + n_lags.size(), vars[4] = n_lags.size();
    auto &m_n_intervals = vars[0], &m_n_samples = vars[1], &m_n_observations = vars[2],
         &m_n_lagged_features = vars[3], &m_n_features = vars[4];
    if (n_lags[0] >= m_n_intervals)
      TICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
    col_offset.m_data = std::vector<size_t>(n_lags.size(), 0);
    for (size_t i(1); i < n_lags.size(); i++) {
      if (n_lags[i] >= m_n_intervals)
        TICK_ERROR("n_lags elements must be between 0 and (m_n_intervals - 1).");
      col_offset[i] = col_offset[i - 1] + n_lags[i - 1] + 1;
    }
    if (m_n_samples != labels.size() || m_n_samples != censoring.size())
      TICK_ERROR("features, labels and censoring should have equal length.");
    for (size_t i(0); i < m_n_samples; i++) {
      if (features[i]->rows() != m_n_intervals)
        TICK_ERROR("All feature matrices should have " << m_n_intervals << " rows");
      if (features[i]->cols() != m_n_lagged_features)
        TICK_ERROR("All feature matrices should have " << m_n_lagged_features << " cols");
      if (labels[i]->size() != m_n_intervals)
        TICK_ERROR("All labels should have " << m_n_intervals << " rows");
    }
    return *this;
  }

  size_t vars[5]{0};
  statick::Array<size_t> col_offset, n_lags, censoring;
  FEATURES features;
  LABELS labels;
  std::shared_ptr<model::lipschitz::DAO<T>> dao_lip;

  inline const auto &n_intervals() const { return vars[0]; }
  inline const auto &n_samples() const { return vars[1]; }
  inline const auto &n_observations() const { return vars[2]; }
  inline const auto &n_lagged_features() const { return vars[3]; }
  inline const auto &n_features() const { return vars[4]; }

  template <class Archive>
  void load(Archive &ar) {
    if(!this->dao_lip) this->dao_lip = std::make_shared<model::lipschitz::DAO<T>>();
    ar(cereal::make_nvp("ModelSCCS", *this->dao_lip.get()));
    ar(vars[1], vars[4], vars[2], vars[3], vars[0]);
    ar(n_lags, col_offset);
    // this->labels = std::make_shared<std::vector<Array<int32_t>>>();
    for(size_t i = 0; i < vars[1]; i++)
      ar(*labels.emplace_back(std::make_shared<Array<int32_t>>()));
    for(size_t i = 0; i < vars[1]; i++)
      ar(*features.emplace_back(std::make_shared<Array2D<T>>()));
    ar(cereal::make_nvp("censoring", this->censoring));
  }

  template <class Archive>
  void save(Archive &ar) const {
    if(!this->dao_lip) this->dao_lip = std::make_shared<model::lipschitz::DAO<T>>();
    ar(cereal::make_nvp("ModelSCCS", *this->dao_lip.get()));
    ar(vars[1], vars[4], vars[2], vars[3], vars[0], n_lags, col_offset);
    for(auto &l : labels) ar(*l);
    for(auto &f : features) ar(*f);
    ar(cereal::make_nvp("censoring", this->censoring));
  }
};

template <class T>
std::shared_ptr<DAO<T>> load_from(std::string file){
  std::ifstream bin_data(file, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive ar(bin_data);
  auto dao = std::make_shared<DAO<T>>();
  ar(*dao.get());
  return dao;
}

}  // namespace sccs
}  // namespace statick
#endif  // TICK_SURVIVAL_DAO_MODEL_SCCS_H_
