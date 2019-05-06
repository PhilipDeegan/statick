#ifndef STATICK_SOLVER_HISTORY_HPP_
#define STATICK_SOLVER_HISTORY_HPP_

namespace statick {
namespace solver {

template <typename T>
class History {
 public:
  History &save_history(double time, size_t epoch, T *iterate, size_t size) {
    time_history[i] = last_record_time + time;
    epoch_history[i] = last_record_epoch + epoch;
    std::copy(iterate, iterate+size, iterate_history[i].data());
    this->i++;
    return *this;
  }
  History &operator+=(size_t iterations) {
    (void) iterations;  // todo?
    return *this;
  }
  History &add_time(double time) { return *this; }

  size_t record_every = 10, last_record_epoch = 0, i = 0;
  double last_record_time = 0;
  std::vector<double> time_history;
  std::vector<size_t> epoch_history;
  std::vector<std::vector<T>> iterate_history;

  History(){};
  ~History(){};
  void init(size_t v_size, size_t i_size){
    time_history.resize(v_size);
    epoch_history.resize(v_size);
    for(size_t vi = 0; vi < v_size; vi++) iterate_history.emplace_back(i_size);
  }

  History(History &that) = delete;
  History(const History &that) = delete;
  History(History &&that) = delete;
  History(const History &&that) = delete;
  History &operator=(History &that) = delete;
  History &operator=(History &&that) = delete;
  History &operator=(const History &that) = delete;
  History &operator=(const History &&that) = delete;
};

template <typename T>
class NoHistory {
 public:
  NoHistory &save_history() { return *this; }
  NoHistory operator+=(size_t iterations) { return *this; }
  NoHistory add_time(double time) { return *this; }
  void init(size_t, size_t){}
};
}  // namespace solver
}  // namespace statick

#endif  // STATICK_SOLVER_HISTORY_HPP_