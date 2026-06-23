# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# StreamingLLM: https://arxiv.org/abs/2309.17453
import torch
import triton
import triton.language as tl

from vllm.config import VllmConfig
from vllm.v1.spec_decode.sparse_attn.attn_overrider import BaseAttnOverrider


@triton.jit
def copy_blocks_kernel(
    src_ptr,
    dst_ptr,
    start_indices_ptr,
    lengths_ptr,
    src_stride,
    dst_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)

    # Offsets for this sequence
    src_row_start = src_ptr + pid * src_stride
    dst_row_start = dst_ptr + pid * dst_stride

    # 1. Copy the sink block (index 0)
    # block_table is int32 usually
    sink_block = tl.load(src_row_start)
    tl.store(dst_row_start, sink_block)

    # 2. Copy the tail blocks
    # src index: start_indices[pid]
    # dst index: 1
    start_idx = tl.load(start_indices_ptr + pid)
    length = tl.load(lengths_ptr + pid)

    src_tail_start = src_row_start + start_idx
    dst_tail_start = dst_row_start + 1

    # Loop over the length of the tail
    # We can perform the copy in a loop with BLOCK_SIZE
    for off in range(0, length, BLOCK_SIZE):
        offsets = off + tl.arange(0, BLOCK_SIZE)
        mask = offsets < length

        # Load from src
        vals = tl.load(src_tail_start + offsets, mask=mask)
        # Store to dst
        tl.store(dst_tail_start + offsets, vals, mask=mask)


class StreamingLLMAttnOverrider(BaseAttnOverrider):
    """An attention overrider that implements StreamingLLM attention."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        super().__init__(vllm_config, device)

        self.max_sliding_blocks = self.max_blocks - 1   # excluding sink
        assert self.max_sliding_blocks > 0

        # Pre-allocate a block table for sliding window attention.
        self._block_table_tensor = torch.empty(
            (self.max_batch_size, self.max_blocks),
            dtype=torch.int32, device=device
        )

    def _may_update_attn_metadata(self, *args, **kwargs):
        if self.curr_layer > 0:
            return

        seqlens: torch.Tensor = kwargs['seqused_k']
        block_table: torch.Tensor = kwargs['block_table']
        batch_size = seqlens.shape[0]

        # 1. Calculate Per-Request Budget
        budget_tokens = seqlens * self.sparse_ratio
        budget_tokens = budget_tokens.clamp_min_(self.min_tokens)
        budget_blocks = torch.ceil(budget_tokens / self.block_size)
        # Clamp budget_blocks to not exceed max_sliding_blocks,
        # ensuring we don't go out of bounds of preallocated block table.
        budget_blocks.clamp_max_(self.max_sliding_blocks)

        # 2. Identify blocks to copy for each request
        # Index of last sliding block to copy (inclusive)
        last_block_indices = (seqlens - 1) // self.block_size
        # Index of first sliding block to copy (inclusive)
        # We want 'budget_blocks' ending at 'last_block_indices'.
        # window range: [last - budget + 1, ..., last].
        # But we clamp start to 1 (since 0 is sink).
        start_block_indices = \
            (last_block_indices - budget_blocks + 1).clamp_min_(1)
        # Number of sliding blocks to copy for each request.
        num_sliding_blocks = \
            (last_block_indices - start_block_indices + 1).clamp_min_(0)

        # 3. Construct gather indices for block table
        # Total columns = 1 (sink) + max_sliding_blocks
        # We use self.max_cols which is precomputed.
        # Ensure indices are contigous int32
        # Use small block size for the copy loop inside kernel
        BLOCK_SIZE = 128
        grid = (batch_size,)
        copy_blocks_kernel[grid](
            block_table,
            self._block_table_tensor,
            start_block_indices.to(torch.int32),
            num_sliding_blocks.to(torch.int32),
            block_table.stride(0),
            self._block_table_tensor.stride(0),
            BLOCK_SIZE=BLOCK_SIZE
        )
        # We take a slice of block_table_tensor
        self.block_table = self._block_table_tensor[:batch_size]

        # 4. Update seqlens (seqused_k)
        # Reduce length by the number of skipped blocks (1 to start - 1)
        skipped_blocks = start_block_indices - 1
        seqlens.copy_(seqlens - (skipped_blocks * self.block_size))

    def _draft_attention(self, *args, **kwargs):
        # Modify attn_metadata if necessary.
        self._may_update_attn_metadata(*args, **kwargs)

        # Override attention metadata
        kwargs['block_table'] = self.block_table

        # Call the original attention function with modified attn_metadata.
        rtv = BaseAttnOverrider._original_attn_func(*args, **kwargs)

        return rtv

    def _verify_attention(self, *args, **kwargs):
        return BaseAttnOverrider._original_attn_func(*args, **kwargs)
