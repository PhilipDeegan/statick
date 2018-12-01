

#include <rocrand.h>
#include <rocrand_kernel.h>
#include <rocrand_mtgp32_11213.h>

/*
rocrand_state_mtgp32 states[N_SAMPLES];
hc::array_view<rocrand_device::mtgp32_fast_params, 1> ar_states(N_SAMPLES, states);
rocrand_device::mtgp32_fast_params params[N_SAMPLES];
std::copy(std::begin(mtgp32dc_params_fast_11213), std::end(mtgp32dc_params_fast_11213),
std::begin(params));
hc::array_view<rocrand_device::mtgp32_fast_params, 1> ar_params(N_SAMPLES, params);*/

unsigned int temper(rocrand_state_mtgp32 &state, unsigned int V, unsigned int T) __HC__ {
  unsigned int MAT;
  T ^= T >> 16;
  T ^= T >> 8;
  MAT = state.temper_tbl[T & 0x0f];
  return V ^ MAT;
}

unsigned int para_rec(rocrand_state_mtgp32 &state, unsigned int X1, unsigned int X2,
                      unsigned int Y) __HC__ {
  unsigned int X = (X1 & state.mask) ^ X2;
  unsigned int MAT;

  X ^= X << state.sh1_tbl;
  Y = X ^ (Y >> state.sh2_tbl);
  MAT = state.param_tbl[Y & 0x0f];
  return Y ^ MAT;
}

void p_gpu(mtgp32_fast_params *params, size_t n, size_t *ref) __HC__ {
  rocrand_state_mtgp32 h_state[n];
  unsigned int seed = 1337;
  seed = seed ^ (seed >> 32);
  size_t i = 0, j = 1;
  for (i = 0; i < n; i++) {
    auto array = &(h_state[i].m_state.status[0]);
    const mtgp32_fast_params *para = &params[i];
    size_t size = para->mexp / 32 + 1;
    unsigned int hidden_seed;
    unsigned int tmp;
    hidden_seed = para->tbl[4] ^ (para->tbl[8] << 16);
    tmp = hidden_seed;
    tmp += tmp >> 16;
    tmp += tmp >> 8;
    memset(array, tmp & 0xff, sizeof(unsigned int) * size);
    array[0] = (unsigned int)seed + i + 1;
    array[1] = hidden_seed;
    for (j = 1; j < size; j++) array[j] ^= (1812433253) * (array[j - 1] ^ (array[j - 1] >> 30)) + j;

    h_state[i].m_state.offset = 0;
    h_state[i].m_state.id = i;
    h_state[i].pos_tbl = params[i].pos;
    h_state[i].sh1_tbl = params[i].sh1;
    h_state[i].sh2_tbl = params[i].sh2;
    h_state[i].mask = params[0].mask;
    for (int j = 0; j < MTGP_TS; j++) {
      h_state[i].param_tbl[j] = params[i].tbl[j];
      h_state[i].temper_tbl[j] = params[i].tmp_tbl[j];
      h_state[i].single_temper_tbl[j] = params[i].flt_tmp_tbl[j];
    }
  }

  for (i = 0; i < n; i++) {
    unsigned int t = hipThreadIdx_x;
    unsigned int d = hipBlockDim_x;
    int pos = h_state[i].pos_tbl;
    unsigned int r;
    unsigned int o;

    r = para_rec(h_state[i], h_state[i].m_state.status[(t + h_state[i].m_state.offset) & MTGP_MASK],
                 h_state[i].m_state.status[(t + h_state[i].m_state.offset + 1) & MTGP_MASK],
                 h_state[i].m_state.status[(t + h_state[i].m_state.offset + pos) & MTGP_MASK]);
    h_state[i].m_state.status[(t + h_state[i].m_state.offset + MTGP_N) & MTGP_MASK] = r;

    o = temper(h_state[i], r,
               h_state[i].m_state.status[(t + h_state[i].m_state.offset + pos - 1) & MTGP_MASK]);
    __syncthreads();
    if (t == 0) h_state[i].m_state.offset = (h_state[i].m_state.offset + d) & MTGP_MASK;
    __syncthreads();
    ref[i] = o;
  }
}
