#ifndef STATICK_SOLVER_HISTORY_HPP_
#define STATICK_SOLVER_HISTORY_HPP_

namespace statick {
namespace solver {

class NoTolerance{};

template <typename T>
class Tolerance{
 public:
   T val{0}, prev_obj{0};
};

template <typename T>
class NoObjective {
 public:
   static NoObjective &I() { static NoObjective i; return i; }
   std::function<T(T*, size_t)> noop = [](T* iterate, size_t size){ return iterate ? 0 : size - size; };
};
class INC{
 public:
  INC(size_t &_i) : i(_i){}
  ~INC(){ i++; }
  size_t &i;
};


template <typename T, typename TOL>
class History {
 public:
  using TOLERANCE = TOL;
  bool save_history(double time, size_t epoch, T *iterate, size_t size) {
    INC inc(i); // increments i on stack unwinding
    time_history[i] = last_record_time + time;
    epoch_history[i] = last_record_epoch + epoch;
    std::copy(iterate, iterate+size, iterate_history[i].data());
    objectives[i] = f_objective(iterate, size);
    if constexpr (std::is_same<TOLERANCE, Tolerance<T>>::value) {
      auto &prev_obj = tol.prev_obj;
      auto &obj = objectives[i];
      auto rel_obj = prev_obj != 0 ? std::abs(obj - prev_obj) / std::abs(prev_obj)
                   : std::abs(obj);
      tol.prev_obj = obj;
      return rel_obj < tol.val;
    }
    return false;
  }

  size_t record_every = 10, last_record_epoch = 0, i = 0;
  double last_record_time = 0;
  std::vector<double> time_history;
  std::vector<size_t> epoch_history;
  std::vector<std::vector<T>> iterate_history;
  std::vector<T> objectives;
  TOLERANCE tol;
  std::function<T(T*, size_t)> &f_objective = NoObjective<T>::I().noop;

  History(){};
  ~History(){};
  void init(size_t v_size, size_t i_size){
    time_history.resize(v_size);
    epoch_history.resize(v_size);
    objectives.resize(v_size);
    iterate_history.resize(0);
    for(size_t vi = 0; vi < v_size; vi++) iterate_history.emplace_back(i_size);
  }
  void set_f_objective(std::function<T(T*, size_t)> &fob){ f_objective = fob; }

  History(History &that) = delete;
  History(const History &that) = delete;
  History(History &&that) = delete;
  History(const History &&that) = delete;
  History &operator=(History &that) = delete;
  History &operator=(History &&that) = delete;
  History &operator=(const History &that) = delete;
  History &operator=(const History &&that) = delete;

};

class NoHistory {
 public:
  using TOLERANCE = NoTolerance;
  NoHistory &save_history() { return *this; }
  void init(size_t, size_t){}
  template <typename T>
  void set_f_objective(std::function<T(T*, size_t)> &){}
};
}  // namespace solver
}  // namespace statick

#endif  // STATICK_SOLVER_HISTORY_HPP_
