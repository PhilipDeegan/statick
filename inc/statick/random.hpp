#ifndef STATICK_RANDOM_HPP_
#define STATICK_RANDOM_HPP_

#ifndef INDICE_TYPE
#ifdef TICK_SPARSE_INDICES_INT64
#define INDICE_TYPE size_t
#else
#define INDICE_TYPE std::uint32_t
#endif
#endif

namespace statick {

template <typename T>
class Random {
 public:
  Random(INDICE_TYPE seed = 0) { reseed(seed); }
  void reseed(INDICE_TYPE seed){
    if(seed > 0) generator = std::mt19937_64(seed);
    else{
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

}

#endif  //  STATICK_RANDOM_HPP_
