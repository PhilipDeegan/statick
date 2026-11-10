#ifndef STATICK_RANDOM_HPP_
#define STATICK_RANDOM_HPP_

#ifndef INDICE_TYPE
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#endif

#include <random>
#include <vector>

namespace statick {

template <typename T>
class Random {
 public:
  Random(INDICE_TYPE seed = 0) { reseed(seed); }
  void reseed(INDICE_TYPE seed) {
    if (seed > 0)
      generator = std::mt19937_64(seed);
    else {
      std::random_device r;
      std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
      generator = std::mt19937_64(seed_seq);
    }
  }

  std::mt19937_64 generator;
  std::uniform_int_distribution<INDICE_TYPE> uniform_dist;
};
template <typename T>
class RandomMinMax : public Random<T> {
 public:
  using Random<T>::reseed;
  using Random<T>::generator;
  using Random<T>::uniform_dist;
  RandomMinMax(INDICE_TYPE min, INDICE_TYPE max) : Random<T>(0), p(min, max) {}
  RandomMinMax(INDICE_TYPE seed, INDICE_TYPE min, INDICE_TYPE max) : Random<T>(seed), p(min, max) {}
  INDICE_TYPE next() { return uniform_dist(generator, p); }

  std::uniform_int_distribution<INDICE_TYPE>::param_type p;
};

// One independently seeded generator per thread - std::mt19937_64 is not thread safe
//  and sharing one across threads can index past its state array
template <typename T>
std::vector<RandomMinMax<T>> make_thread_randoms(size_t n_threads, INDICE_TYPE min,
                                                 INDICE_TYPE max) {
  std::vector<RandomMinMax<T>> rands;
  rands.reserve(n_threads);
  for (size_t i = 0; i < n_threads; ++i) rands.emplace_back(min, max);
  return rands;
}

}  // namespace statick

#endif  //  STATICK_RANDOM_HPP_
