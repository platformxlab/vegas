"""
Variable-length mean reduce — single-file, JIT-compiled.

Usage:
    from varlen_reduce import varlen_reduce

    output = varlen_reduce(x, valid_lens, reduce_entry, output,
                           lse, cu_seqlens_q, softmax_scale, use_weight)

Two ranking modes (selected by ``use_weight``):

* logit (use_weight=False): average the raw (pre-softmax) QK scores in ``x``.
* weight (use_weight=True):  rematerialize the softmax attention weight for
  each key before reducing,

      weight[h, k] = exp(softmax_scale * score[h, k] - lse[h, tok])

  where ``lse`` is the per-(head, query-token) log-sum-exp produced by the
  attention kernel (natural log of sum_k exp(softmax_scale * score)) and
  ``tok`` is the first/last query token of the sequence (the two query rows
  the attention kernel stored into ``x``). Normalizing makes heads comparable.

Input (bf16):  x   (batch, dim1, 2, max_seqlen)   raw QK scores
Input (fp32):  lse (dim1, total_q)                attention log-sum-exp
Output (bf16): output (batch, max_seqlen)
Accumulation in fp32.
"""

import os
import torch
from torch.utils.cpp_extension import load_inline

# ----------------------------------------------------------------
# CUDA kernel source
# ----------------------------------------------------------------

_CUDA_SRC = r"""
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <torch/extension.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAStream.h>

static constexpr int kBlockSize = 128;
static constexpr int kVecWidth  = 8;   // 8 bf16 = uint4 = 16 bytes
static constexpr float kLog2e   = 1.4426950408889634f;  // log2(e)

// Native single-instruction base-2 exponential. This is exactly the SFU op
// __expf compiles to after it folds in log2(e); same accuracy as __expf.
__device__ __forceinline__ float fast_exp2(float x) {
    float r;
    asm("ex2.approx.f32 %0, %1;" : "=f"(r) : "f"(x));
    return r;
}

// Rematerialize one attention weight from a raw (pre-softmax) score.
//   exp(scale*score - lse) == exp2(log2e*scale*score - log2e*lse)
// The log2(e) factor is folded into the caller's ``scale2`` (= scale*log2e)
// and ``lse2`` (= lse*log2e), which is computed once per head rather than per
// key, so this reduces to a single FMA + native ex2 -- one fewer multiply per
// element than __expf (which folds log2e internally on every call).
// Masked keys are stored as -inf -> scale2 * -inf - lse2 = -inf -> 0.
__device__ __forceinline__ float weight_from_score(
    float score, float scale2, float lse2) {
    return fast_exp2(scale2 * score - lse2);
}

// ---- scalar kernel (1 element per thread) ----

template <bool USE_WEIGHT>
__global__ void VarlenMeanReduceKernel(
    const __nv_bfloat16* __restrict__ input,   // (batch, dim1, 2, max_seqlen)
    const float* __restrict__ lse,             // (dim1, total_q) or unused
    const int32_t* __restrict__ cu_seqlens_q,  // (batch + 1,) or unused
    __nv_bfloat16* __restrict__ output,        // (batch, max_seqlen)
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ reduce_entry,
    float softmax_scale,
    int32_t dim1,
    int32_t total_q,
    int32_t max_seqlen) {

    const int32_t bid = blockIdx.y;
    const int32_t l   = blockIdx.x * blockDim.x + threadIdx.x;

    if (l >= max_seqlen) return;

    const int32_t vlen = valid_lens[bid];
    if (vlen == 0 || l >= vlen) return;

    const int32_t entry = reduce_entry[bid];
    const int32_t hstride = 2 * max_seqlen;
    const int64_t boff =
        static_cast<int64_t>(bid) * dim1 * hstride;

    int32_t first_tok = 0, last_tok = 0;
    if constexpr (USE_WEIGHT) {
        first_tok = cu_seqlens_q[bid];
        last_tok  = cu_seqlens_q[bid + 1] - 1;
    }

    const float scale2 = softmax_scale * kLog2e;

    // Transform one raw score into the per-key contribution.
    auto val = [&](float score, int64_t ho, int32_t tok) -> float {
        if constexpr (USE_WEIGHT) {
            return weight_from_score(score, scale2, lse[ho + tok] * kLog2e);
        } else {
            return score;
        }
    };

    float sum = 0.f;

    if (entry == 0) {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const float s0 = __bfloat162float(
                input[boff + h * hstride + l]);
            const float s1 = __bfloat162float(
                input[boff + h * hstride + max_seqlen + l]);
            sum += val(s0, ho, first_tok);
            sum += val(s1, ho, last_tok);
        }
        sum *= (1.f / static_cast<float>(dim1 * 2));
    } else if (entry == 1) {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const float s0 = __bfloat162float(
                input[boff + h * hstride + l]);
            sum += val(s0, ho, first_tok);
        }
        sum *= (1.f / static_cast<float>(dim1));
    } else {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const float s1 = __bfloat162float(
                input[boff + h * hstride + max_seqlen + l]);
            sum += val(s1, ho, last_tok);
        }
        sum *= (1.f / static_cast<float>(dim1));
    }

    output[bid * max_seqlen + l] = __float2bfloat16(sum);
}

// ---- vectorized kernel (8 elements per thread, uint4 loads) ----

template <bool USE_WEIGHT>
__global__ void VarlenMeanReduceKernelVec(
    const __nv_bfloat16* __restrict__ input,
    const float* __restrict__ lse,
    const int32_t* __restrict__ cu_seqlens_q,
    __nv_bfloat16* __restrict__ output,
    const int32_t* __restrict__ valid_lens,
    const int32_t* __restrict__ reduce_entry,
    float softmax_scale,
    int32_t dim1,
    int32_t total_q,
    int32_t max_seqlen) {

    const int32_t bid    = blockIdx.y;
    const int32_t base_l =
        (blockIdx.x * blockDim.x + threadIdx.x) * kVecWidth;

    if (base_l >= max_seqlen) return;

    const int32_t vlen = valid_lens[bid];
    if (vlen == 0 || base_l >= vlen) return;

    const int64_t out_off =
        static_cast<int64_t>(bid) * max_seqlen + base_l;

    const int32_t entry   = reduce_entry[bid];
    const int32_t hstride = 2 * max_seqlen;
    const int64_t boff    =
        static_cast<int64_t>(bid) * dim1 * hstride;

    int32_t first_tok = 0, last_tok = 0;
    if constexpr (USE_WEIGHT) {
        first_tok = cu_seqlens_q[bid];
        last_tok  = cu_seqlens_q[bid + 1] - 1;
    }

    const float scale2 = softmax_scale * kLog2e;

    float s[kVecWidth];
    #pragma unroll
    for (int i = 0; i < kVecWidth; ++i) s[i] = 0.f;

    // Accumulate one slot (uint4-loaded 8 scores) into s[]. lse2 is the
    // log2-domain log-sum-exp (lse * log2e); only used in weight mode.
    auto accum_slot = [&](int64_t off, float lse2) {
        const uint4 v = __ldg(
            reinterpret_cast<const uint4*>(&input[off]));
        const __nv_bfloat16* p =
            reinterpret_cast<const __nv_bfloat16*>(&v);
        #pragma unroll
        for (int i = 0; i < kVecWidth; ++i) {
            const float sc = __bfloat162float(p[i]);
            if constexpr (USE_WEIGHT) {
                s[i] += weight_from_score(sc, scale2, lse2);
            } else {
                s[i] += sc;
            }
        }
    };

    if (entry == 0) {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const int64_t off0 =
                boff + static_cast<int64_t>(h) * hstride + base_l;
            const float lse0 = USE_WEIGHT ? lse[ho + first_tok] * kLog2e : 0.f;
            const float lse1 = USE_WEIGHT ? lse[ho + last_tok] * kLog2e : 0.f;
            accum_slot(off0, lse0);
            accum_slot(off0 + max_seqlen, lse1);
        }
        const float inv = 1.f / static_cast<float>(dim1 * 2);
        #pragma unroll
        for (int i = 0; i < kVecWidth; ++i) s[i] *= inv;
    } else if (entry == 1) {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const int64_t off =
                boff + static_cast<int64_t>(h) * hstride + base_l;
            const float lse0 = USE_WEIGHT ? lse[ho + first_tok] * kLog2e : 0.f;
            accum_slot(off, lse0);
        }
        const float inv = 1.f / static_cast<float>(dim1);
        #pragma unroll
        for (int i = 0; i < kVecWidth; ++i) s[i] *= inv;
    } else {
        #pragma unroll 8
        for (int32_t h = 0; h < dim1; ++h) {
            const int64_t ho = static_cast<int64_t>(h) * total_q;
            const int64_t off =
                boff + static_cast<int64_t>(h) * hstride + max_seqlen + base_l;
            const float lse1 = USE_WEIGHT ? lse[ho + last_tok] * kLog2e : 0.f;
            accum_slot(off, lse1);
        }
        const float inv = 1.f / static_cast<float>(dim1);
        #pragma unroll
        for (int i = 0; i < kVecWidth; ++i) s[i] *= inv;
    }

    union { __nv_bfloat16 bf[kVecWidth]; uint4 u4; } res;
    #pragma unroll
    for (int i = 0; i < kVecWidth; ++i) res.bf[i] = __float2bfloat16(s[i]);

    *reinterpret_cast<uint4*>(&output[out_off]) = res.u4;
}

// ---- launcher with runtime dispatch ----

void launch_varlen_reduce(
    at::Tensor input,
    at::Tensor lse,
    at::Tensor cu_seqlens_q,
    at::Tensor output,
    at::Tensor valid_lens,
    at::Tensor reduce_entry,
    double softmax_scale,
    bool use_weight) {

    TORCH_CHECK(input.dim() == 4);
    TORCH_CHECK(output.dim() == 2);
    TORCH_CHECK(input.is_cuda() && output.is_cuda());
    TORCH_CHECK(input.is_contiguous());
    TORCH_CHECK(output.is_contiguous());
    TORCH_CHECK(input.size(2) == 2);
    TORCH_CHECK(
        input.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(
        output.scalar_type() == at::ScalarType::BFloat16);
    TORCH_CHECK(
        reduce_entry.scalar_type() == at::ScalarType::Int);

    const int32_t batch_size = input.size(0);
    const int32_t dim1       = input.size(1);
    const int32_t max_seqlen = input.size(3);

    TORCH_CHECK(valid_lens.size(0) == batch_size);
    TORCH_CHECK(reduce_entry.size(0) == batch_size);
    TORCH_CHECK(output.size(0) == batch_size);
    TORCH_CHECK(output.size(1) == max_seqlen);
    TORCH_CHECK(max_seqlen % kVecWidth == 0,
        "max_seqlen must be a multiple of ", kVecWidth);
    TORCH_CHECK(
        valid_lens.scalar_type() == at::ScalarType::Int);
    TORCH_CHECK(valid_lens.is_contiguous());
    TORCH_CHECK(reduce_entry.is_contiguous());

    int32_t total_q = 0;
    if (use_weight) {
        // Attention log-sum-exp, shape (dim1, total_q), fp32, contiguous.
        TORCH_CHECK(lse.is_cuda() && lse.is_contiguous());
        TORCH_CHECK(lse.scalar_type() == at::ScalarType::Float);
        TORCH_CHECK(lse.dim() == 2 && lse.size(0) == dim1,
            "lse must be (num_heads, total_q), num_heads == input.size(1)");
        total_q = lse.size(1);
        // cu_seqlens_q, shape (batch + 1,), int32.
        TORCH_CHECK(cu_seqlens_q.is_cuda() && cu_seqlens_q.is_contiguous());
        TORCH_CHECK(cu_seqlens_q.scalar_type() == at::ScalarType::Int);
        TORCH_CHECK(cu_seqlens_q.size(0) == batch_size + 1);
    }

    const float softmax_scale_f = static_cast<float>(softmax_scale);
    const float* lse_ptr = static_cast<const float*>(lse.data_ptr());
    const int32_t* cu_ptr =
        static_cast<const int32_t*>(cu_seqlens_q.data_ptr());
    const __nv_bfloat16* in_ptr =
        reinterpret_cast<const __nv_bfloat16*>(input.data_ptr());
    __nv_bfloat16* out_ptr =
        reinterpret_cast<__nv_bfloat16*>(output.data_ptr());
    const int32_t* vl_ptr =
        static_cast<const int32_t*>(valid_lens.data_ptr());
    const int32_t* re_ptr =
        static_cast<const int32_t*>(reduce_entry.data_ptr());

    const c10::cuda::OptionalCUDAGuard guard(input.device());
    const cudaStream_t stream = c10::cuda::getCurrentCUDAStream();

    // Decide: vectorized (8 elems/thread) vs scalar
    const int elems_per_block_vec = kBlockSize * kVecWidth;   // 1024
    const int grid_x_vec =
        (max_seqlen + elems_per_block_vec - 1) / elems_per_block_vec;
    const int total_blocks_vec = grid_x_vec * batch_size;
    const bool use_vec = (total_blocks_vec >= 256);

    dim3 block(kBlockSize);
    dim3 grid_vec(grid_x_vec, batch_size);
    dim3 grid_sc((max_seqlen + kBlockSize - 1) / kBlockSize, batch_size);

#define LAUNCH_REDUCE(USE_W)                                                \
    do {                                                                    \
        if (use_vec) {                                                      \
            VarlenMeanReduceKernelVec<USE_W><<<grid_vec, block, 0,          \
                stream>>>(in_ptr, lse_ptr, cu_ptr, out_ptr, vl_ptr,        \
                re_ptr, softmax_scale_f, dim1, total_q, max_seqlen);        \
        } else {                                                            \
            VarlenMeanReduceKernel<USE_W><<<grid_sc, block, 0,             \
                stream>>>(in_ptr, lse_ptr, cu_ptr, out_ptr, vl_ptr,        \
                re_ptr, softmax_scale_f, dim1, total_q, max_seqlen);        \
        }                                                                   \
    } while (0)

    if (use_weight) {
        LAUNCH_REDUCE(true);
    } else {
        LAUNCH_REDUCE(false);
    }
#undef LAUNCH_REDUCE
}
"""

_CPP_SRC = """
void launch_varlen_reduce(
    at::Tensor input,
    at::Tensor lse,
    at::Tensor cu_seqlens_q,
    at::Tensor output,
    at::Tensor valid_lens,
    at::Tensor reduce_entry,
    double softmax_scale,
    bool use_weight);
"""

# ----------------------------------------------------------------
# JIT compile (cached after first call)
# ----------------------------------------------------------------

_module = None


def _get_module():
    global _module
    if _module is not None:
        return _module

    os.environ.setdefault(
        "TORCH_CUDA_ARCH_LIST",
        "8.0;8.9;9.0",
    )

    _module = load_inline(
        name="varlen_reduce_jit",
        cpp_sources=_CPP_SRC,
        cuda_sources=_CUDA_SRC,
        functions=["launch_varlen_reduce"],
        extra_cuda_cflags=[
            "-O3",
            "-std=c++17",
            "--expt-relaxed-constexpr",
            "--expt-extended-lambda",
        ],
        verbose=False,
    )
    return _module


# ----------------------------------------------------------------
# Public API
# ----------------------------------------------------------------

def varlen_reduce(
    x: torch.Tensor,
    valid_lens: torch.Tensor,
    reduce_entry: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    softmax_scale: float,
    use_weight: bool,
) -> torch.Tensor:
    """Variable-length mean reduction over dim1/dim2.

    If ``use_weight`` is False, averages the raw QK scores in ``x`` (logit
    ranking). If True, rematerializes the softmax attention weight
    ``exp(softmax_scale * score - lse)`` per key (weight ranking) before the
    head/slot mean. Reduction accumulates in fp32; output is bf16.

    CUDA-graph compatible: all per-batch tensors have shape[0] == max_num_seq.
    Inactive sequences are indicated by valid_lens[i] == 0.

    Args:
        x:      (batch_size, dim1, 2, max_seqlen)  bf16
                 raw (pre-softmax) QK scores; max_seqlen a multiple of 8.
                 dim2[0] = first query token, dim2[1] = last query token.
        valid_lens:    (batch_size,)  int32. 0 means inactive (output skipped).
        reduce_entry:  (batch_size,)  int32
            0 -> mean over dim1 & both first/last query slots.
            1 -> mean over dim1, first-query slot only.
            2 -> mean over dim1, last-query slot only.
        output: (batch_size, max_seqlen) bf16 buffer.
        lse:    (dim1, total_q) fp32 attention log-sum-exp (natural log). Used
            only when use_weight=True; pass any CUDA tensor otherwise.
        cu_seqlens_q: (batch_size + 1,) int32 query cumulative seqlens; used
            only when use_weight=True to locate each sequence's first/last
            query token in ``lse``.
        softmax_scale: float attention scale; used only when use_weight=True.
        use_weight: select weight (True) vs logit (False) ranking.

    Returns:
        (batch_size, max_seqlen) bf16 tensor.
    """
    assert x.is_cuda, "x must be on CUDA"
    assert x.dim() == 4
    assert x.size(2) == 2
    assert x.size(3) % 8 == 0, "max_seqlen must be a multiple of 8"
    assert x.dtype == torch.bfloat16
    assert x.is_contiguous()

    bs = x.size(0)
    dim1 = x.size(1)
    sl = x.size(3)

    assert output.shape == (bs, sl)
    assert output.dtype == torch.bfloat16
    assert output.is_contiguous()

    assert valid_lens.shape == (bs,)
    assert valid_lens.dtype == torch.int32
    assert valid_lens.is_contiguous()
    assert reduce_entry.shape == (bs,)
    assert reduce_entry.dtype == torch.int32
    assert reduce_entry.is_contiguous()

    if use_weight:
        assert lse.is_cuda and lse.is_contiguous()
        assert lse.dtype == torch.float32
        assert lse.dim() == 2 and lse.size(0) == dim1
        assert cu_seqlens_q.is_cuda and cu_seqlens_q.is_contiguous()
        assert cu_seqlens_q.dtype == torch.int32
        assert cu_seqlens_q.shape == (bs + 1,)

    _get_module().launch_varlen_reduce(
        x, lse, cu_seqlens_q, output, valid_lens, reduce_entry,
        float(softmax_scale), bool(use_weight))
    return output
