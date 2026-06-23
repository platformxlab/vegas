# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
from abc import ABC, abstractmethod
from math import ceil

import torch

from vllm.config import VllmConfig
from vllm.logger import init_logger

logger = init_logger(__name__)


class BaseAttnOverrider(ABC):
    _GLOBAL_OVERRIDER_COUNT = 0

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        self.vllm_config = vllm_config
        self.device = device

        speculative_config = vllm_config.speculative_config
        self.num_spec_tokens = speculative_config.num_speculative_tokens
        self.sparse_ratio = speculative_config.sparse_attn_ratio
        self.min_tokens = speculative_config.sparse_attn_min_tokens
        self.block_size = vllm_config.cache_config.block_size
        self.max_batch_size = vllm_config.scheduler_config.max_num_seqs
        self.max_model_len = vllm_config.model_config.max_model_len

        max_tokens = ceil(self.max_model_len * self.sparse_ratio)
        max_tokens = max(max_tokens, self.min_tokens)
        max_blocks = (max_tokens + self.block_size - 1) // self.block_size
        self.max_tokens = max_tokens
        self.max_blocks = max_blocks

        # TODO: Develop a more reliable way to iterate through attn layers.
        self.num_layers = self.vllm_config.model_config.get_num_layers(
                self.vllm_config.parallel_config)
        self.curr_layer = 0

        # TODO: Support more attention backends.
        if BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT == 0:
            import vllm.v1.attention.backends.flash_attn as flash_attn
            BaseAttnOverrider._original_attn_func = \
                flash_attn.flash_attn_varlen_func
            flash_attn.flash_attn_varlen_func = \
                lambda *args, **kwargs: self._attention(*args, **kwargs)

        # By default, we are not inside the propose method
        self.in_propose = False

        # Increment the global overrider count
        BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT += 1

    def __del__(self):
        # Decrement the global overrider count
        BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT -= 1
        if BaseAttnOverrider._GLOBAL_OVERRIDER_COUNT == 0:
            import vllm.v1.attention.backends.flash_attn as flash_attn
            flash_attn.flash_attn_varlen_func = \
                BaseAttnOverrider._original_attn_func

    def enter_propose(self):
        self.in_propose = True
        assert self.curr_layer == 0

    def exit_propose(self):
        self.in_propose = False
        assert self.curr_layer == 0

    def _attention(self, *args, **kwargs):
        if self.in_propose:
            rtv = self._draft_attention(*args, **kwargs)
        else:
            rtv = self._verify_attention(*args, **kwargs)

        self.curr_layer = (self.curr_layer + 1) % self.num_layers
        return rtv

    @abstractmethod
    def _draft_attention(self, *args, **kwargs):
        pass

    @abstractmethod
    def _verify_attention(self, *args, **kwargs):
        pass


def build_attention_overrider(
    vllm_config: VllmConfig,
    device: torch.device,
):
    assert vllm_config.speculative_config is not None

    method = vllm_config.speculative_config.sparse_attn_algorithm
    if method == "streamingllm":
        from .streamingllm import StreamingLLMAttnOverrider
        cls = StreamingLLMAttnOverrider
    elif method == "vegas":
        from .vegas import VegasAttnOverrider
        cls = VegasAttnOverrider
    else:
        raise ValueError(f"Unknown sparse_attn_algorithm: {method}")

    cls_name = cls.__name__.strip('\'')
    logger.info(f"Resolved attention overrider: {cls_name}")

    # Instantiate the attention overrider.
    return cls(
        vllm_config=vllm_config,
        device=device,
    )
