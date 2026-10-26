#include <cmath>
#include <cstddef>

constexpr int threads_per_block = 1024;
constexpr int warp_size = 32;
constexpr int num_warps = threads_per_block / warp_size;
#define CUDART_INF_F INFINITY

__inline__ __device__ float warpReduceSum(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val += __shfl_xor_sync(0xffffffff, val, mask, 32);
  }

  return val;
}

__inline__ __device__ float blockReduceSum(float val) {
  static __shared__ float shared[32];

  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : 0.0f;
  val = warpReduceSum(val);

  return val;
}

__inline__ __device__ float warpReduceMax(float val) {
#pragma unroll
  for (int mask = 16; mask > 0; mask >>= 1) {
    val = fmaxf(val, __shfl_xor_sync(0xffffffff, val, mask, 32));
  }

  return val;
}

__inline__ __device__ float blockReduceMax(float val) {
  static __shared__ float shared[32];

  int lane = threadIdx.x & 0x1f;
  int wid = threadIdx.x >> 5;

  val = warpReduceSum(val);
  if (lane == 0) {
    shared[wid] = val;
  }
  __syncthreads();

  val = (threadIdx.x < (blockDim.x / 32.f)) ? shared[lane] : 0.0f;
  val = warpReduceSum(val);

  return val;
}

__device__ float dotp(const float *x, const float *y, int dim) {
  float res = 0.0;
  for (int i = 0; i < dim; ++i) {
    res += x[i] * y[i];
  }

  return res;
}

__device__ void
max_sim_kernel(const float *const query,         // [query_tokens, token_dim]
               const float *const document,      // [document_tokens, token_dim]
               const int *const document_starts, // [n_docs]
               const int *const document_lengths, // [n_docs]
               float *res,                        // [n_docs]
               const int query_tokens, const int token_dim) {
  int q_token_idx = blockIdx.y;
  int d_idx = blockIdx.x;
  int d_start = document_starts[d_idx];
  int d_len = document_lengths[d_idx];

  extern __shared__ float maxes[];
  int tid = threadIdx.x;

  // initialize per token maximum
  for (int i = tid; i < query_tokens; i += threads_per_block) {
    maxes[i] = -CUDART_INF_F;
  }
  __syncthreads();

  int lane = tid & (warp_size - 1);
  int warp_id = tid >> 5;

  //                      num_warps
  //                   +--------------+
  //                   |              |
  //                   w1             w1
  //   doc \ query:  | t0 | t1 | t2 | t3 | ... | tn |
  //   t0            | xx             xx            | l1 -+
  //   t1            |                              |     | warp_size
  //   t2            |                              |     |
  //   t3            | xx             xx            | l1 -+
  //   t4            |                              |
  //   t5            |                              |
  //   t6            | xx             xx            |
  //   ...           |                              |
  //   tn            |                              |

  for (int q_tok = warp_id; q_tok < query_tokens; q_tok += num_warps) {
    int q_idx = (q_token_idx + q_tok) * token_dim;
    const float *const query_start = &query[q_idx];

    float my_max = -CUDART_INF_F;
    for (int d_tok = lane; d_tok < d_len; d_tok += warp_size) {
      const float *const document_start =
          &document[(d_start + d_tok) * token_dim];
      my_max = fmaxf(my_max, dotp(query_start, document_start, token_dim));
    }

    for (int off = warp_size / 2; off > 0; off >>= 1) {
      my_max = fmaxf(my_max, __shfl_xor_sync(0xffffffff, my_max, off));
    }

    if (lane == 0)
      maxes[q_tok] = my_max;
  }

  __syncthreads();

  if (warp_id == 0) {
    float s = 0.0f;
    for (int i = lane; i < query_tokens; i += warp_size) {
      s += maxes[i];
    }

    for (int off = warp_size / 2; off > 0; off >>= 1) {
      s += __shfl_xor_sync(0xffffffff, s, off);
    }

    if (lane == 0) {
      res[d_idx] = s;
    }
  }
}

extern "C" {
__global__ void
max_sim(const float *const query,          // [query_tokens, token_dim]
        const float *const documents,      // [document_tokens, token_dim]
        const int *const document_starts,  // [n_docs]
        const int *const document_lengths, // [n_docs]
        float *res,                        // [n_docs]
        const int query_tokens, const int token_dim) {
  max_sim_kernel(query, documents, document_starts, document_lengths, res,
                 query_tokens, token_dim);
}
}
