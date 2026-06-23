# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from typing import TYPE_CHECKING

import numpy as np
import torch
import torch.nn as nn

from vllm.config import (
    CUDAGraphMode,
    VllmConfig,
    get_layers_from_vllm_config,
)
from vllm.forward_context import set_forward_context
from vllm.logger import init_logger
from vllm.model_executor.layers.attention_layer_base import AttentionLayerBase
from vllm.triton_utils import triton
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.attention.backend import (
    AttentionMetadataBuilder,
    CommonAttentionMetadata,
)
from vllm.v1.attention.backends.flash_attn import FlashAttentionMetadata
from vllm.v1.cudagraph_dispatcher import CudagraphDispatcher
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.outputs import SamplerOutput
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.rejection_sampler import compute_probs
from vllm.v1.spec_decode.metadata import SpecDecodeMetadata
from vllm.v1.spec_decode.sparse_attn.attn_overrider import (
    build_attention_overrider,
)
from vllm.v1.spec_decode.utils import (
    PADDING_SLOT_ID,
    eagle_prepare_inputs_padded_kernel,
    eagle_prepare_next_token_padded_kernel,
    gather_draft_hidden_states,
)
from vllm.v1.utils import CpuGpuBuffer
from vllm.v1.worker.dp_utils import coordinate_batch_across_dp
from vllm.v1.worker.gpu_input_batch import CachedRequestState, InputBatch

if TYPE_CHECKING:
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

logger = init_logger(__name__)


def _method_wrapper(enter_fn, exit_fn):
    """Decorator that calls enter/exit hooks around a method.

    Useful for setting and resetting states or configurations
    before and after a method call, without modifying the
    method's internal logic.

    Args:
        enter_fn: Called with ``self`` before the wrapped
            method. Use it to set up required state.
        exit_fn: Called with ``self`` after the wrapped
            method (in a ``finally`` block). Use it to
            clean up or reset state.

    Returns:
        A decorator that can be applied to a method.
    """

    def decorator(method):
        def wrapper(self, *args, **kwargs):
            enter_fn(self)
            try:
                return method(self, *args, **kwargs)
            finally:
                exit_fn(self)
        return wrapper
    return decorator


class SparseAttnProposer:
    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
        runner: "GPUModelRunner",
    ):
        # TODO: Support multimodal inputs and more RoPE techniques.
        self.vllm_config = vllm_config
        assert vllm_config.speculative_config is not None
        self.speculative_config = vllm_config.speculative_config

        self.runner = runner
        self.device = device
        self.dtype = vllm_config.model_config.dtype
        self.max_model_len = vllm_config.model_config.max_model_len
        self.dp_rank = vllm_config.parallel_config.data_parallel_rank
        self.num_speculative_tokens = \
            self.speculative_config.num_speculative_tokens
        self.block_size = vllm_config.cache_config.block_size
        self.hidden_size = vllm_config.model_config.get_hidden_size()

        max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.token_arange_np = np.arange(max_batch_size)

        self.attn_metadata_builder: AttentionMetadataBuilder | None = None
        self.attn_layer_names: list[str] = []

        # Yikang: Given SparseAttnProposer is based on self-speculation,
        # it seems natural to reuse the cudagraph_dispatcher from runner.
        # However, as that cudagraph_dispatcher assumes uniform decode with
        # (num_speculative_tokens + 1) tokens for each request, it misbehaves
        # when we use it for padding the draft tokens during propose().
        self.cudagraph_dispatcher = CudagraphDispatcher(self.vllm_config)
        self.cudagraph_dispatcher.uniform_decode_query_len = 1

        # We need +1 here because the arange is used to set query_start_loc,
        # which has one more element than batch_size.
        max_num_slots_for_arange = max(max_batch_size + 1, max_batch_size)
        self.arange = torch.arange(
            max_num_slots_for_arange, device=device, dtype=torch.int32)

        self.backup_next_token_ids = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # Determine allowed attention backends once during initialization.
        # TODO: Support more attention backends.
        self.allowed_attn_types = (FlashAttentionMetadata,)

        # Sampled draft token ids buffer.
        self._sampled_token_ids = torch.empty(
            (max_batch_size, self.num_speculative_tokens),
            dtype=torch.int32, device=device
        )

        # Last hidden states for draft tokens
        self._draft_hidden_states = torch.empty(
            (max_batch_size, self.num_speculative_tokens, self.hidden_size),
            dtype=self.dtype, device=device
        )
        # Previous req_id_to_index for potentially reordering the batch.
        self._prev_req_id_to_index: dict[str, int] | None = None
        # Indicates whether a batch reorder is needed
        self._need_batch_reorder = False
        # Reorder mapping created by update_req_order
        # Usage: new_draft_hidden_states = prev_draft_hidden_states[mapping]
        self._batch_reorder_mapping = CpuGpuBuffer(
            max_batch_size,
            dtype=torch.int32,
            pin_memory=is_pin_memory_available(),
            device=device,
            with_numpy=True,
        )

        # Initialize attention overrider.
        self.attn_overrider = build_attention_overrider(
            vllm_config=self.vllm_config,
            device=self.device,
        )

    @property
    def sampled_token_ids(self) -> torch.Tensor:
        return self._sampled_token_ids[:self.batch_size]

    def initialize_cudagraph_keys(self, cudagraph_mode: CUDAGraphMode) -> None:
        # This should be called BEFORE adjust_cudagraph_sizes_for_spec_decode.
        self.cudagraph_dispatcher.initialize_cudagraph_keys(cudagraph_mode)

    def get_draft_probs(
        self,
        spec_decode_metadata: SpecDecodeMetadata,
        sampling_metadata: SamplingMetadata,
    ) -> torch.Tensor:
        cu_num_draft_tokens = spec_decode_metadata.cu_num_draft_tokens
        batch_size = cu_num_draft_tokens.shape[0]
        num_draft_tokens = spec_decode_metadata.num_draft_tokens
        total_tokens = sum(num_draft_tokens)

        if not self._need_batch_reorder and \
                batch_size * self.num_speculative_tokens == total_tokens:
            # Fast path: no reorder and every request drafted exactly gamma
            # tokens — just reshape the contiguous buffer (no copy).
            hidden_states = (self._draft_hidden_states[:batch_size]
                             .reshape(-1, self.hidden_size))
        else:
            # Need the gather kernel: either the batch was reordered
            # or the number of drafted tokens per request is not uniform
            if self._need_batch_reorder:
                reorder_mapping = self._batch_reorder_mapping.gpu[:batch_size]
            else:
                # Token counts are misaligned but batch order is the
                # same — use identity mapping via self.arange.
                reorder_mapping = self.arange[:batch_size]
            hidden_states = gather_draft_hidden_states(
                src=self._draft_hidden_states,
                cu_num_draft_tokens=cu_num_draft_tokens,
                total_tokens=total_tokens,
                batch_size=batch_size,
                reorder_mapping=reorder_mapping,
            )

        # Compute the draft probabilities.
        draft_logits: torch.Tensor = self.model.compute_logits(hidden_states)
        return compute_probs(
            draft_logits, cu_num_draft_tokens, sampling_metadata)

    def _save_hidden_states_and_sample(
        self,
        draft_depth: int,
        hidden_states: torch.Tensor,
        sampling_metadata: SamplingMetadata,
    ):
        # Save draft hidden states for later use in verification.
        hidden_states = hidden_states[:self.batch_size]
        self._draft_hidden_states[:self.batch_size, draft_depth].copy_(
            hidden_states)

        # Compute the draft logits
        logits = self.model.compute_logits(hidden_states)

        # Sample using the full sampler (applies penalties, temperature,
        # top-k/top-p, and sampling in-place on logits).
        output: SamplerOutput = self.runner.sampler(logits, sampling_metadata)
        token_ids = output.sampled_token_ids.flatten()
        self._sampled_token_ids[:self.batch_size, draft_depth].copy_(token_ids)

    def update_batch_order(self, req_id_to_index: dict[str, int]):
        # Initialization: no previous hidden states to remap.
        if self._prev_req_id_to_index is None:
            self._prev_req_id_to_index = req_id_to_index.copy()
            self._need_batch_reorder = False
            return

        # Check if the order has changed.
        if req_id_to_index == self._prev_req_id_to_index:
            self._need_batch_reorder = False
            self._prev_req_id_to_index = req_id_to_index.copy()
            return

        # Build reorder mapping: mapping[new_index] = prev_index.
        # Only requests present in both batches need remapping.
        # New requests (not in prev) have no previous hidden states;
        # their mapping slot is left as-is (the hidden states at that
        # position will be overwritten during propose() anyway).
        self._need_batch_reorder = True
        for req_id, new_index in req_id_to_index.items():
            prev_index = self._prev_req_id_to_index.get(req_id)
            if prev_index is not None:
                self._batch_reorder_mapping.np[new_index] = prev_index
            else:
                # New request — map to itself so index_select doesn't
                # read stale data from an unrelated slot.
                self._batch_reorder_mapping.np[new_index] = new_index
        self._batch_reorder_mapping.copy_to_gpu()
        self._prev_req_id_to_index = req_id_to_index.copy()

    def _enter_propose(self):
        self.attn_overrider.enter_propose()

    def _exit_propose(self):
        self.attn_overrider.exit_propose()

    @_method_wrapper(enter_fn=_enter_propose, exit_fn=_exit_propose)
    def propose(
        self,
        # [batch_size]
        next_token_ids: torch.Tensor,
        token_indices_to_sample: torch.Tensor | None,
        common_attn_metadata: CommonAttentionMetadata,
        sampling_metadata: SamplingMetadata,
        mm_embed_inputs: tuple[list[torch.Tensor], torch.Tensor] | None = None,
        num_rejected_tokens_gpu: torch.Tensor | None = None,
        slot_mappings: dict[str, torch.Tensor]
        | list[dict[str, torch.Tensor]]
        | None = None,
    ) -> torch.Tensor | None:
        # Keep track of the batch size for later use in properties.
        self.batch_size = num_tokens = common_attn_metadata.batch_size()
        # Set the input ids for the first drafting step.
        self.input_ids[:num_tokens] = next_token_ids

        # Rollback seqlens based on num_rejected_tokens_gpu.
        # Use an out-of-place subtract: under async scheduling this tensor may
        # still be read by the main model's next-step prepare_input, so an
        # in-place op here could corrupt it.
        if num_rejected_tokens_gpu is not None:
            common_attn_metadata.seq_lens = \
                common_attn_metadata.seq_lens - num_rejected_tokens_gpu

        # Mask out the positions that exceed the max model length.
        # Otherwise, we may get out-of-range error in RoPE.
        positions = common_attn_metadata.seq_lens
        exceeds_max_model_len = positions >= self.max_model_len
        clamped_positions = torch.where(exceeds_max_model_len, 0, positions)
        self.positions[:num_tokens] = clamped_positions

        # Update common attention metadata for the first drafting step.
        common_attn_metadata.num_actual_tokens = num_tokens
        common_attn_metadata.max_query_len = 1
        common_attn_metadata.query_start_loc = self.arange[: num_tokens + 1]
        common_attn_metadata.query_start_loc_cpu = \
            torch.from_numpy(self.token_arange_np[: num_tokens + 1]).clone()
        common_attn_metadata.max_seq_len = \
            min(common_attn_metadata.max_seq_len + 1, self.max_model_len)
        # For the requests that exceed the max model length, we set
        # their sequence lengths to 1 to minimize their overheads in attention.
        common_attn_metadata.seq_lens = clamped_positions + 1

        # Compute the slot mapping.
        block_numbers = clamped_positions // self.block_size
        block_ids = common_attn_metadata.block_table_tensor.gather(
            dim=1, index=block_numbers.view(-1, 1))
        block_ids = block_ids.view(-1)
        common_attn_metadata.slot_mapping[:num_tokens].copy_((
            block_ids * self.block_size + clamped_positions % self.block_size))
        # Mask out the slot mappings that exceed the max model length.
        # Otherwise, the KV cache will be updated with the padding tokens.
        common_attn_metadata.slot_mapping[:num_tokens].masked_fill_(
            exceeds_max_model_len, PADDING_SLOT_ID)
        slot_mapping = common_attn_metadata.slot_mapping[:num_tokens]

        if self.attn_metadata_builder is None:
            attn_metadata_builder = self._get_attention_metadata_builder()
        else:
            attn_metadata_builder = self.attn_metadata_builder

        # Create and update attention metadata for the first drafting step.
        attn_metadata = attn_metadata_builder.build_for_drafting(
            common_attn_metadata=common_attn_metadata, draft_index=0)
        assert isinstance(attn_metadata, self.allowed_attn_types), (
            f"Attention metadata type {type(attn_metadata)} not supported. "
            f"Supported types: {self.allowed_attn_types}"
        )

        # At this moment, we assume all attn layers belong to the same KV
        # cache group, thus using the same attention metadata.
        per_layer_attn_metadata = {}
        per_layer_slot_mapping = {}
        for layer_name in self.attn_layer_names:
            per_layer_attn_metadata[layer_name] = attn_metadata
            per_layer_slot_mapping[layer_name] = slot_mapping

        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = \
            self._determine_batch_execution_and_padding(num_tokens)

        model_kwargs = {
            "input_ids": self.input_ids[:num_input_tokens],
            "positions": self.positions[:num_input_tokens],
            "inputs_embeds": None,  # MM input support can be added here
        }

        with set_forward_context(
            per_layer_attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=per_layer_slot_mapping,
        ):
            hidden_states = self.model(**model_kwargs)
            self._save_hidden_states_and_sample(
                draft_depth=0,
                hidden_states=hidden_states,
                sampling_metadata=sampling_metadata,
            )

        # Early exit if there is only one draft token to be generated.
        if self.num_speculative_tokens == 1:
            return self.sampled_token_ids

        # Speculatively sample multiple tokens.
        for step in range(1, self.num_speculative_tokens):
            self.input_ids[:num_tokens] = \
                self._sampled_token_ids[:num_tokens, step - 1]
            self.positions[:num_tokens] += 1

            # Mask out the positions that exceed the max model length.
            # Otherwise, we may get out-of-range error in RoPE.
            positions = self.positions[:num_tokens]
            exceeds_max_model_len = positions >= self.max_model_len
            clamped_positions = \
                torch.where(exceeds_max_model_len, 0, positions)
            self.positions[:num_tokens] = clamped_positions

            # Update the attention metadata. Accumulate on attn_metadata's own
            # field (reused across steps): common_attn_metadata.max_seq_len is
            # fixed before the loop, so reading it here would pin max_seq_len
            # at +1 even though seq_lens grows by one every step.
            attn_metadata.max_seq_len = \
                min(attn_metadata.max_seq_len + 1, self.max_model_len)
            attn_metadata.seq_lens[:num_tokens].copy_(clamped_positions + 1)

            # Compute the slot mapping.
            block_numbers = clamped_positions // self.block_size
            block_ids = attn_metadata.block_table.gather(
                dim=1, index=block_numbers.view(-1, 1))
            block_ids = block_ids.view(-1)
            slot_mapping.copy_(
                block_ids * self.block_size +
                clamped_positions % self.block_size
            )
            # Mask out the slot mappings that exceed the max model length.
            # Otherwise, the KV cache will be inadvertently updated with
            # the padding tokens.
            slot_mapping.masked_fill_(exceeds_max_model_len, PADDING_SLOT_ID)

            model_kwargs = {
                "input_ids": self.input_ids[:num_input_tokens],
                "positions": self.positions[:num_input_tokens],
                "inputs_embeds": None,  # MM input support can be added here
            }

            with set_forward_context(
                per_layer_attn_metadata,
                self.vllm_config,
                num_tokens=num_input_tokens,
                num_tokens_across_dp=num_tokens_across_dp,
                cudagraph_runtime_mode=cudagraph_runtime_mode,
                slot_mapping=per_layer_slot_mapping,
            ):
                hidden_states = self.model(**model_kwargs)
                # Get the logits and sample the next token.
                self._save_hidden_states_and_sample(
                    draft_depth=step,
                    hidden_states=hidden_states,
                    sampling_metadata=sampling_metadata,
                )

        # [batch_size, num_speculative_tokens]
        return self.sampled_token_ids

    def prepare_next_token_ids_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        sampled_token_ids: torch.Tensor,
        requests: dict[str, CachedRequestState],
        gpu_input_batch: InputBatch,
        discard_request_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding.
        It calculates the next token ids and the number of valid sampled tokens
        for each request, considering the "discarded" requests whose next token
        is not sampled and comes from `request.get_token_id()` instead.
        This is denoted the "backup" token id.
        It also counts rejected tokens via `sampled_token_ids`.
        """
        # Precompute get_token_id for when there is no valid next token
        num_reqs = gpu_input_batch.num_reqs
        self.backup_next_token_ids.np[:num_reqs] = np.array(
            [
                requests[gpu_input_batch.req_ids[i]].get_token_id(
                    common_attn_metadata.seq_lens_cpu[i].item()
                )
                for i in range(num_reqs)
            ],
            dtype=np.int32,
        )
        self.backup_next_token_ids.copy_to_gpu(num_reqs)
        backup_tokens_gpu = self.backup_next_token_ids.gpu

        batch_size, num_tokens = sampled_token_ids.shape
        device = sampled_token_ids.device

        assert discard_request_mask.dtype == torch.bool
        assert backup_tokens_gpu.dtype == torch.int32

        next_token_ids = \
            torch.empty(batch_size, dtype=torch.int32, device=device)
        valid_sampled_tokens_count = next_token_ids.new_empty(batch_size)

        # Kernel grid: one program per request (row)
        grid = (batch_size,)

        # Find the next power of 2 for block sizes
        BLOCK_SIZE_TOKENS = triton.next_power_of_2(num_tokens)
        eagle_prepare_next_token_padded_kernel[grid](
            sampled_token_ids,
            discard_request_mask,
            backup_tokens_gpu,
            next_token_ids,
            valid_sampled_tokens_count,
            gpu_input_batch.vocab_size,
            num_tokens,
            batch_size,
            sampled_token_ids.stride(0),
            BLOCK_SIZE_TOKENS=BLOCK_SIZE_TOKENS,
        )

        return next_token_ids, valid_sampled_tokens_count

    def prepare_inputs_padded(
        self,
        common_attn_metadata: CommonAttentionMetadata,
        spec_decode_metadata: SpecDecodeMetadata,
        valid_sampled_tokens_count: torch.Tensor,
    ) -> tuple[CommonAttentionMetadata, torch.Tensor, torch.Tensor]:
        """
        This function is used to prepare the inputs for speculative decoding
        It updates the common_attn_metadata for speculative decoding,
        but does not consider the rejected tokens. Instead, all tokens
        are included as inputs to the speculator, with the rejected tokens
        used as padding and filtered out later by `token_indices_to_sample`.
        No blocking CPU operations should be introduced in this function.
        """
        num_reqs = common_attn_metadata.num_reqs
        device = valid_sampled_tokens_count.device

        token_indices_to_sample = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )
        num_rejected_tokens_gpu = torch.empty(
            (num_reqs,), dtype=torch.int32, device=device
        )

        grid = (num_reqs,)
        eagle_prepare_inputs_padded_kernel[grid](
            spec_decode_metadata.cu_num_draft_tokens,
            valid_sampled_tokens_count,
            common_attn_metadata.query_start_loc,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
            num_reqs,
        )

        query_start_loc_cpu = common_attn_metadata.query_start_loc_cpu
        new_query_len_per_req = \
            query_start_loc_cpu[1:] - query_start_loc_cpu[:-1]

        total_num_tokens = query_start_loc_cpu[-1].item()

        spec_common_attn_metadata = CommonAttentionMetadata(
            query_start_loc=common_attn_metadata.query_start_loc,
            seq_lens=common_attn_metadata.seq_lens,
            query_start_loc_cpu=query_start_loc_cpu,
            _seq_lens_cpu=common_attn_metadata._seq_lens_cpu,
            _num_computed_tokens_cpu=(
                common_attn_metadata._num_computed_tokens_cpu),
            num_reqs=common_attn_metadata.num_reqs,
            num_actual_tokens=total_num_tokens,
            max_query_len=new_query_len_per_req.max().item(),
            max_seq_len=common_attn_metadata.seq_lens_cpu.max().item(),
            block_table_tensor=common_attn_metadata.block_table_tensor,
            slot_mapping=common_attn_metadata.slot_mapping[:total_num_tokens],
            causal=True,
            dcp_local_seq_lens=common_attn_metadata.dcp_local_seq_lens,
        )

        return (
            spec_common_attn_metadata,
            token_indices_to_sample,
            num_rejected_tokens_gpu,
        )

    def load_model(self, target_model: nn.Module) -> None:
        # Self-speculative decoding
        self.model = target_model

        # Register attention layers and their metadata builders.
        self.attn_layer_names = list(get_layers_from_vllm_config(
            self.vllm_config, AttentionLayerBase).keys())

        # Reuse runner's buffers for inputs and positions.
        self.input_ids = self.runner.input_ids.gpu
        self.positions = self.runner.positions.gpu

    @torch.inference_mode()
    @_method_wrapper(enter_fn=_enter_propose, exit_fn=_exit_propose)
    def dummy_run(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
        is_graph_capturing: bool = False,
        attn_metadata: dict[str, FlashAttentionMetadata] | None = None,
        slot_mappings: dict[str, torch.Tensor] | None = None,
    ) -> None:
        cudagraph_runtime_mode, num_input_tokens, num_tokens_across_dp = \
            self._determine_batch_execution_and_padding(
                num_tokens, use_cudagraphs=use_cudagraphs
            )

        kwargs = dict(
            input_ids=self.input_ids[:num_input_tokens],
            positions=self.positions[:num_input_tokens],
            inputs_embeds=None,
        )

        with set_forward_context(
            attn_metadata,
            self.vllm_config,
            num_tokens=num_input_tokens,
            num_tokens_across_dp=num_tokens_across_dp,
            cudagraph_runtime_mode=cudagraph_runtime_mode,
            slot_mapping=slot_mappings,
        ):
            self.model(**kwargs)

    def _get_attention_metadata_builder(self) -> AttentionMetadataBuilder:
        """Find and return the attention metadata builders for EAGLE layers.

        Returns:
            The metadata builders for target model layers.

        Raises:
            AssertionError: If no metadata builders are found.
        """
        builder = None
        chosen_layer = self.attn_layer_names[0]

        for kv_cache_group in self.runner.attn_groups:
            for attn_group in kv_cache_group:
                if chosen_layer in attn_group.layer_names:
                    builder = attn_group.get_metadata_builder()
                    break
            if builder is not None:
                break

        assert builder is not None, (
            "Failed to find attention metadata builder."
        )
        return builder

    def validate_same_kv_cache_group(self, kv_cache_config: KVCacheConfig):
        """
        Validate that all target model layers belong to the same KVCacheGroup.
        Need this assumption to ensure all target model layers can use the
        same AttentionMetadata.
        May extend to multiple AttentionMetadata in the future.
        """
        kv_cache_groups: dict[str, int] = {}
        for id, kv_cache_group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in kv_cache_group.layer_names:
                kv_cache_groups[layer_name] = id
        assert (
            len(
                set(
                    [
                        kv_cache_groups[layer_name]
                        for layer_name in self.attn_layer_names
                    ]
                )
            )
            == 1
        ), "All target model layers should belong to the same kv cache group"

    def _determine_batch_execution_and_padding(
        self,
        num_tokens: int,
        use_cudagraphs: bool = True,
    ) -> tuple[CUDAGraphMode, int, torch.Tensor | None]:
        cudagraph_mode, batch_desc = self.cudagraph_dispatcher.dispatch(
            num_tokens=num_tokens, uniform_decode=True)
        num_tokens_padded = batch_desc.num_tokens

        # Extra coordination when running data-parallel since we need to
        # coordinate across ranks
        # TODO(Flechman): support DBO ubatching
        should_ubatch, num_tokens_across_dp = False, None
        if self.vllm_config.parallel_config.data_parallel_size > 1:
            should_ubatch, num_tokens_across_dp, synced_cudagraph_mode = (
                coordinate_batch_across_dp(
                    num_tokens_unpadded=num_tokens,
                    parallel_config=self.vllm_config.parallel_config,
                    allow_microbatching=False,
                    num_tokens_padded=num_tokens_padded,
                    cudagraph_mode=cudagraph_mode.value,
                )
            )
            assert not should_ubatch, "DBO ubatching not implemented."

            # Extract DP-synced values
            if num_tokens_across_dp is not None:
                dp_rank = self.dp_rank
                num_tokens_padded = int(num_tokens_across_dp[dp_rank].item())
                # Re-dispatch with DP padding so we have the correct
                # batch_descriptor
                cudagraph_mode, batch_desc = \
                    self.cudagraph_dispatcher.dispatch(
                        num_tokens_padded, uniform_decode=True
                    )
                # Assert to make sure the agreed upon token count is correct
                # otherwise num_tokens_across_dp will no-longer be valid
                assert batch_desc.num_tokens == num_tokens_padded
                num_tokens_across_dp[dp_rank] = num_tokens_padded

        return cudagraph_mode, num_tokens_padded, num_tokens_across_dp
