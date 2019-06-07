#ifndef STATICK_BASE_MODEL_DAO_MODEL_LIPSCHITZ_H_
#define STATICK_BASE_MODEL_DAO_MODEL_LIPSCHITZ_H_

#define STATICK_ERROR(x) std::cerr << x << std::endl;

namespace statick {
namespace model {
namespace lipschitz {

template <typename T = double>
class DAO {
 public:
  inline bool ready_lip_consts() const { return b_vars[0]; }
  inline bool ready_lip_max() const { return b_vars[1]; }
  inline bool ready_lip_mean() const { return b_vars[2]; }

  inline T lip_mean() const { return t_vars[0]; }
  inline T lip_max() const { return t_vars[1]; }

  template <class Archive>
  void serialize(Archive &ar) {
    bool &ready_lip_consts = b_vars[0], &ready_lip_max = b_vars[1], &ready_lip_mean = b_vars[2];
    T &lip_mean = t_vars[0],  &lip_max = t_vars[1];
    ar(CEREAL_NVP(ready_lip_consts), CEREAL_NVP(ready_lip_max),
       CEREAL_NVP(ready_lip_mean), CEREAL_NVP(lip_consts), CEREAL_NVP(lip_mean),
       CEREAL_NVP(lip_max));
  }

  bool b_vars[3]{false};
  T t_vars[2]{0};
  statick::Array<T> lip_consts;

  size_t size_of() const {
    return (sizeof(T) * 2) + 24 + lip_consts.size();
  }
};

template <class ARCHIVE, class T>
std::shared_ptr<DAO<T>> load(ARCHIVE &ar, std::shared_ptr<DAO<T>> &dao = nullptr){
  if(!dao) dao = std::make_shared<DAO<T>>();
  ar(*dao.get());
  return dao;
}

template <class T>
std::shared_ptr<DAO<T>> load(std::string file){
  std::ifstream bin_data(file, std::ios::in | std::ios::binary);
  cereal::PortableBinaryInputArchive ar(bin_data);
  return load<cereal::PortableBinaryInputArchive, T>(ar);
}

}  // namespace lipschitz
}  // namespace model
}  // namespace statick
#endif  // STATICK_BASE_MODEL_DAO_MODEL_LIPSCHITZ_H_
