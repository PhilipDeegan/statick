  std::mt19937_64 generator;
  std::random_device r;
  std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
  generator = std::mt19937_64(seed_seq);
  std::uniform_int_distribution<size_t> uniform_dist;
  std::uniform_int_distribution<size_t>::param_type p(0, n_samples - 1);
  auto next_i = [&]() { return uniform_dist(generator, p); };
