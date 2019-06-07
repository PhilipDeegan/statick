
#include <random>
#include <thread>

#include <hcc/hc.hpp>

#include "kul/os.hpp"
#include "kul/log.hpp"
#include "kul/proc.hpp"
#include "kul/signal.hpp"

#include "cereal/archives/portable_binary.hpp"
#include "cereal/types/vector.hpp"

#include "array.hpp"
#include "log_reg.hpp"
#include "saga.hpp"
#include "serial.hpp"

constexpr size_t SEED = 1933, N_ITER = 30, TIMES = 1;

#define RAM_HERE KLOG(INF) << (kul::this_proc::physicalMemory() / 1000) << " MB";

void sparse(int argc, char *argv[]);
void dense(int argc, char *argv[]);

int main(int argc, char *argv[]) {
  RAM_HERE;
  if (kul::env::EXISTS("SOLVE_TYPE") && (kul::env::GET("SOLVE_TYPE") == "dense"))
    dense(argc, argv);
  else
    sparse(argc, argv);
  return 0;
}

void sparse(int argc, char *argv[]) {
  using INDEX_TYPE = INDICE_TYPE;
  kul::Signal sig;
  KOUT(NON) << "sparse";
  bool cpu = 0, gpu = 1;
  std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all(), accelerators;
  for (auto a : all_accelerators)
    if (a.is_hsa_accelerator()) accelerators.push_back(a);
  size_t devices = accelerators.size(), total_devices = all_accelerators.size();
  KOUT(NON) << "Devices " << devices;
  if (devices == 0) exit(2);

  std::string features_s("sparse.features.data"), labels_s("sparse.labels.data");
  auto dev_lambda = [&](size_t dev_id) {
    try {
      std::vector<hc::accelerator_view> acc_views;
      acc_views.push_back(accelerators[dev_id].get_default_view());
      auto synchronize_to_device = [&](auto &view) {
        if (dev_id) {
          view.synchronize_to(acc_views.back());
          view.synchronize();
        }
      };

      std::vector<double> data;
      std::vector<size_t> info;
      std::vector<INDICE_TYPE> indices, row_indices;
      for (size_t i = 0; i < TIMES; i++) {
        std::ifstream bin_data(features_s, std::ios::in | std::ios::binary);
        cereal::PortableBinaryInputArchive iarchive(bin_data);
        statick::hcc::load_sparse2d_with_raw_data(iarchive, data, info, indices, row_indices);
      }
      RAM_HERE;

      statick::hcc::Sparse2DList<double> l_features(data, info, indices, row_indices);
      RAM_HERE;

      size_t N_SAMPLES = l_features[0].rows(), N_FEATURES = l_features[0].cols();
      std::vector<double> gradients_average(N_FEATURES * TIMES),
          gradients_memory(N_SAMPLES * TIMES), iterate(N_FEATURES * TIMES);
      RAM_HERE;

      std::vector<double> steps_corrections(
          statick::saga::sparse::compute_step_corrections(l_features[0]));
      RAM_HERE;

      std::vector<size_t> next_i(N_SAMPLES * TIMES);
      std::random_device r;
      std::seed_seq seed_seq{r(), r(), r(), r(), r(), r(), r(), r()};
      std::mt19937_64 generator(seed_seq);
      std::uniform_int_distribution<size_t> uniform_dist;
      std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
      for (size_t i = 0; i < next_i.size(); i++) next_i[i] = uniform_dist(generator, p);
      hc::array_view<size_t, 1> ar_next_i(next_i.size(), next_i.data());
      synchronize_to_device(ar_next_i);
      RAM_HERE;

      size_t ti[TIMES], vals[2] = {N_ITER, devices};
      for (size_t i = 0; i < TIMES; i++) ti[i] = i;
      hc::array_view<size_t, 1> tim(TIMES, ti), ar_vals(2, vals);
      synchronize_to_device(tim);
      RAM_HERE;

      std::vector<double> vlabels(N_SAMPLES * TIMES);
      for (size_t i = 0; i < TIMES; i++) {
        std::ifstream data(labels_s, std::ios::in | std::ios::binary);
        cereal::PortableBinaryInputArchive iarchive(data);
        statick::hcc::load_array_with_raw_data(iarchive, vlabels.data() + (N_SAMPLES * i));
      }
      RAM_HERE;

      hc::array_view<double, 1> ar_labels(vlabels.size(), vlabels.data()),
          ar_gradients_average(gradients_average.size(), gradients_average.data()),
          ar_gradients_memory(gradients_memory.size(), gradients_memory.data()),
          ar_iterate(iterate.size(), iterate.data()),
          ar_steps_corrections(steps_corrections.size(), steps_corrections.data());
      synchronize_to_device(ar_labels);
      synchronize_to_device(ar_gradients_average);
      synchronize_to_device(ar_gradients_memory);
      synchronize_to_device(ar_iterate);
      synchronize_to_device(ar_steps_corrections);

      RAM_HERE;

      if (cpu)
        statick::saga::sparse::solve(l_features[0], ar_labels.data(), ar_gradients_average.data(),
                                  ar_gradients_memory.data(), ar_iterate.data(), ar_next_i.data(),
                                  ar_steps_corrections.data());

      RAM_HERE;
      std::vector<hc::completion_future> futures;
      futures.emplace_back(hc::parallel_for_each(acc_views.back(), hc::extent<1>(1), [=
      ](hc::index<1> i)[[hc]] { ar_gradients_average[0] = 10 + dev_id; }));
      for (auto f : futures) f.wait();
      std::cout << __LINE__ << " : " << dev_id << " : " << ar_gradients_average[0] << std::endl;
      futures.clear();
      futures.emplace_back(
          hc::parallel_for_each(acc_views.back(), hc::extent<1>(TIMES), [=](hc::index<1> i)[[hc]] {
            for (size_t iter = 0; iter < N_ITER; iter++)
              statick::saga::sparse::solve(l_features[tim[i]], ar_labels.data() + (N_SAMPLES * tim[i]),
                                        ar_gradients_average.data() + (N_FEATURES * tim[i]),
                                        ar_gradients_memory.data() + (N_SAMPLES * tim[i]),
                                        ar_iterate.data() + (N_FEATURES * tim[i]),
                                        ar_next_i.data() + (N_SAMPLES * tim[i]),
                                        ar_steps_corrections.data());
          }));
      for (auto f : futures) f.wait();
    } catch (const std::exception &e) {
      std::cerr << e.what() << std::endl;
    } catch (...) {
      std::cerr << "UNKOWN EXCEPTION CAUGHT" << std::endl;
    }
  };
  std::vector<std::thread> threads;
  for (size_t di = 0; di < devices; di++)
    threads.emplace_back([&](size_t dev_id) { dev_lambda(dev_id); }, di);
  for (auto &thread : threads) thread.join();
  RAM_HERE;
}

void dense(int argc, char *argv[]) {
  KOUT(NON) << "dense";
  // using INDEX_TYPE = INDICE_TYPE;
  // kul::Signal sig;
  // bool cpu = 0, gpu = 1;
  // std::vector<hc::accelerator> all_accelerators = hc::accelerator::get_all(), accelerators;
  // for (auto a : all_accelerators) if (a.is_hsa_accelerator()) accelerators.push_back(a);
  // size_t devices = accelerators.size();
  // KOUT(NON) << "Devices " << devices;
  // if(devices == 0) exit(2);
  try {
    // size_t N_FEATURES = 200;
    // size_t N_SAMPLES = 75000;

    // std::vector<double> vfeatures(N_FEATURES * N_SAMPLES * TIMES), vlabels(N_SAMPLES * TIMES),
    //     gradients_average(N_FEATURES * TIMES), gradients_memory(N_SAMPLES * TIMES),
    //     iterate(N_FEATURES * TIMES);
    // std::fill(gradients_average.begin(), gradients_average.end(), 0);
    // std::fill(gradients_memory.begin(), gradients_memory.end(), 0);
    // std::fill(iterate.begin(), iterate.end(), 0);

    // std::vector<INDEX_TYPE> next_i(N_SAMPLES * TIMES, 0);
    // const auto random_time_start = kul::Now::MILLIS();
    // std::mt19937_64 generator;
    // std::uniform_int_distribution<size_t> uniform_dist;
    // std::uniform_int_distribution<size_t>::param_type p(0, N_SAMPLES - 1);
    // for (size_t i = 0; i < TIMES; i++)
    //   for (size_t j = 0; j < N_SAMPLES; j++) next_i[j + (i * TIMES)] = uniform_dist(generator,
    //   p);
    // KOUT(NON) << "Randoms done in  " << (kul::Now::MILLIS() - random_time_start) << " ms";

    // const auto features_time_start = kul::Now::MILLIS();
    // for (size_t t = 0; t < TIMES; t++) {
    //   auto sample = test_uniform(N_SAMPLES * N_FEATURES, SEED);
    //   ArrayDouble2d sample2d(N_SAMPLES, N_FEATURES, sample->data());
    //   auto features = SArrayDouble2d::new_ptr(sample2d);
    //   auto int_sample = test_uniform_int(0, 2, N_SAMPLES, SEED);
    //   auto labels = SArrayDouble::new_ptr(N_SAMPLES);
    //   for (int i = 0; i < N_SAMPLES; ++i) (*labels)[i] = (*int_sample)[i] - 1;
    //   for (size_t i = 0; i < (N_FEATURES * N_SAMPLES); i++)
    //     vfeatures[i + (t * TIMES)] = features->data()[i];
    //   for (size_t i = 0; i < (N_SAMPLES); i++) vlabels[i + (t * TIMES)] = (*labels)[i];
    // }
    // KOUT(NON) << "Features done in  " << (kul::Now::MILLIS() - features_time_start) << " ms";
    // const auto s = kul::Now::MILLIS();
    // {
    //   size_t ti[TIMES], vals[1];
    //   vals[0] = N_ITER;
    //   for (size_t i = 0; i < TIMES; i++) ti[i] = i;
    //   hc::array_view<size_t, 1> tim(TIMES, ti);
    //   hc::array_view<size_t, 1> ar_vals(1, vals);
    //   hc::array_view<double, 1> ar_features(vfeatures.size(), vfeatures.data()),
    //       ar_labels(vlabels.size(), vlabels.data()),
    //       ar_gradients_average(gradients_average.size(), gradients_average.data()),
    //       ar_gradients_memory(gradients_memory.size(), gradients_memory.data()),
    //       ar_iterate(iterate.size(), iterate.data());
    //   hc::array_view<INDEX_TYPE, 1> ar_next_i(next_i.size(), next_i.data());

    //   if (cpu)
    //     statick::saga::dense::solve(ar_features.data(), ar_labels.data(),
    //     ar_gradients_average.data(),
    //                              ar_gradients_memory.data(), ar_iterate.data(),
    //                              ar_next_i.data());

    //   if (gpu)
    //     hc::parallel_for_each(hc::extent<1>TIMES, [=](hc::index<1> i) [[hc]] {
    //       for (int j = 0; j < ar_vals[0]; ++j)
    //         statick::saga::dense::solve(ar_features.data() + (N_FEATURES * N_SAMPLES * tim[i]),
    //                                  ar_labels.data() + (N_SAMPLES * tim[i]),
    //                                  ar_gradients_average.data() + (N_FEATURES * tim[i]),
    //                                  ar_gradients_memory.data() + (N_SAMPLES * tim[i]),
    //                                  ar_iterate.data() + (N_FEATURES * tim[i]),
    //                                  ar_next_i.data() + (N_SAMPLES * tim[i]));
    //     });
    // }
    // KOUT(NON) << "SOLVE for " << TIMES << " : " << (kul::Now::MILLIS() - s) << " ms";
  } catch (const std::exception &e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    std::cerr << "UNKOWN EXCEPTION CAUGHT" << std::endl;
  }
}
