<!-- markdownlint-disable MD001 MD041 -->

# Vegas: Verification-Guided Sparse Attention for Self-Speculative Decoding

[![Paper](https://img.shields.io/badge/arXiv-2602.07223-b31b1b.svg)](https://arxiv.org/abs/2602.07223)

<!-- TODO: add teaser/overview figure -->
<p align="center">
  <img alt="Vegas overview" src="assets/overview.png" width=80%>
</p>

## Abstract

Long-context large language model (LLM) inference has become the norm for today's AI applications. However, it is severely bottlenecked by the increasing memory demands of its KV cache. Previous works have shown that self-speculative decoding with sparse attention, where tokens are drafted using a subset of the KV cache and verified in parallel against the full KV cache, speeds up inference in a lossless manner. However, they rely on a standalone KV selection algorithm to select the KV entries used for drafting and overlook the fact that the criticality of each KV entry is inherently computed during verification.

To this end, we propose Vegas, a self-speculative decoding method with verification-guided sparse attention. Vegas identifies critical KV cache entries as a byproduct of verification and computes attention only over these entries when drafting subsequent tokens. This not only improves the draft token acceptance rate but also incurs low KV selection overhead, thereby improving decoding throughput. Vegas achieves a 1.25×-2.81× speedup in decoding throughput over default vLLM and a 1.15×-1.29× speedup over state-of-the-art sparse attention-based self-speculative decoding methods.

## Features

- **Verification-guided KV selection.** During each verification pass Vegas
  collects per-token attention importance (raw pre-softmax logits, or
  rematerialized softmax weights) and ranks the KV cache. On the next draft
  step, instead of attending to the full cache, the model attends to only the
  top-k highest-ranked entries plus the most recent tokens. In other words, the
  draft attends to the entries verification deemed important.
- **Self-speculation, no extra model.** The same weights draft and verify, so
  there is nothing extra to download, train, or keep in memory.
- **Sparse-attention drafting.** The draft pass runs flash-attention over only
  the selected slots via a custom per-step page table, cutting draft attention
  cost on long contexts.
- **CUDA-graph compatible.** Both the verify and draft passes run under CUDA
  graphs; the sparse page table and per-request KV budgets are rebuilt each
  propose so replayed graphs stay correct.

## Installation

Vegas is implemented as a fork of vLLM and builds against a companion
[flash-attention fork](https://github.com/npz7yyk/vllm-flash-attn) (CUDA 12.x,
FlashAttention-3). Build from source:

```bash
git clone https://github.com/npz7yyk/vegas.git
cd vegas
pip install -v -e .   # compiles the CUDA/FA3 kernels; takes a while
pip install datasets  # for benchmarks
```

## Usage

Enable Vegas by passing a `speculative_config` with `method="sparse_attn"`:

```python
from vllm import LLM, SamplingParams

speculative_config = {
    "method": "sparse_attn",
    "num_speculative_tokens": 6,
    "sparse_attn_algorithm": "vegas",   # Also supported: "streamingllm"
    "sparse_attn_ratio": 0.07,          # fraction of KV kept for drafting
    # "sparse_attn_min_tokens": 256,    # floor on the per-request KV budget
}

llm = LLM(
    model="Qwen/Qwen3-8B",
    speculative_config=speculative_config,
)
print(llm.generate(..., SamplingParams(...)))
```

Key knobs (`speculative_config`):

| Field | Meaning | Default |
| --- | --- | --- |
| `sparse_attn_algorithm` | `"vegas"` or `"streamingllm"` | `"streamingllm"` |
| `sparse_attn_ratio` | Fraction of KV kept for drafting | `0.05` |
| `sparse_attn_min_tokens` | Floor on the per-request KV budget | `256` |
| `num_speculative_tokens` | Draft length per step | / |

The top-k ranking metric (`"logit"` raw scores vs `"weight"` rematerialized
softmax weights) is a class-level `SCORE_MODE` toggle on `VegasAttnOverrider`.

## Example

A complete, runnable end-to-end example (AIME'25, Qwen3-8B) lives in
[`benchmarks/benchmark_vegas.py`](benchmarks/benchmark_vegas.py):

```bash
python benchmarks/benchmark_vegas.py
```

## Code Layout

Vegas lives in a self-contained module under `vllm/v1/spec_decode/sparse_attn/`,
plus a handful of edits to wire it into vLLM's speculative-decoding path.

```text
vllm/v1/spec_decode/sparse_attn/
├── proposer.py                     # SparseAttnProposer: drives the self-speculative draft loop
├── attn_overrider/
│   ├── __init__.py                 # BaseAttnOverrider + build_attention_overrider() dispatch
│   ├── vegas.py                    # VegasAttnOverrider: verification-guided KV selection
│   ├── streamingllm.py             # StreamingLLMAttnOverrider: sink + sliding-window baseline
│   └── utils/
│       ├── varlen_reduce.py        # CUDA kernel: reduce per-query scores/weights -> per-token metric
│       └── varlen_topk.py          # CUDA kernel: variable-length top-k KV selection
```

Integration points (edited vLLM files):

| File | What it does for Vegas |
| --- | --- |
| `vllm/config/speculative.py` | Adds the `sparse_attn_*` config fields (`method="sparse_attn"`) |
| `vllm/v1/worker/gpu_model_runner.py` | Constructs and drives `SparseAttnProposer`; CUDA-graph wiring |
| `vllm/v1/spec_decode/utils.py` | Shared spec-decode helpers used by the proposer |
| `vllm/v1/sample/rejection_sampler.py` | Accept/reject of drafted tokens |
| `vllm/v1/core/sched/scheduler.py` | Reserves lookahead slots so KV pages are allocated correctly for the draft tokens |
| `benchmarks/benchmark_vegas.py` | End-to-end example / benchmark |

The verification-guided selection relies on a modified attention kernel that
collects the per-token attention logits (raw pre-softmax QK scores, and
optionally the log-sum-exp for weight rematerialization) as a byproduct of the
verify pass. This is exposed through the `scores` parameter of
[`flash_attn_varlen_func`](vllm/vllm_flash_attn/flash_attn_interface.py) and
implemented in our companion
[flash-attention fork](https://github.com/npz7yyk/vllm-flash-attn).

## Citation

If you find this project helpful to your research, please consider citing our paper:

```bibtex
@misc{yue2026vegasselfspeculativedecodingverificationguided,
      title={Vegas: Self-Speculative Decoding with Verification-Guided Sparse Attention}, 
      author={Yikang Yue and Yuqi Xue and Jian Huang},
      year={2026},
      eprint={2602.07223},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2602.07223}, 
}
```

---

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-dark.png">
    <img alt="vLLM" src="https://raw.githubusercontent.com/vllm-project/vllm/main/docs/assets/logos/vllm-logo-text-light.png" width=55%>
  </picture>
</p>

<h3 align="center">
Easy, fast, and cheap LLM serving for everyone
</h3>

<p align="center">
| <a href="https://docs.vllm.ai"><b>Documentation</b></a> | <a href="https://blog.vllm.ai/"><b>Blog</b></a> | <a href="https://arxiv.org/abs/2309.06180"><b>Paper</b></a> | <a href="https://x.com/vllm_project"><b>Twitter/X</b></a> | <a href="https://discuss.vllm.ai"><b>User Forum</b></a> | <a href="https://slack.vllm.ai"><b>Developer Slack</b></a> |
</p>

🔥 We have built a vllm website to help you get started with vllm. Please visit [vllm.ai](https://vllm.ai) to learn more.
For events, please visit [vllm.ai/events](https://vllm.ai/events) to join us.

---

## About

vLLM is a fast and easy-to-use library for LLM inference and serving.

Originally developed in the [Sky Computing Lab](https://sky.cs.berkeley.edu) at UC Berkeley, vLLM has evolved into a community-driven project with contributions from both academia and industry.

vLLM is fast with:

- State-of-the-art serving throughput
- Efficient management of attention key and value memory with [**PagedAttention**](https://blog.vllm.ai/2023/06/20/vllm.html)
- Continuous batching of incoming requests
- Fast model execution with CUDA/HIP graph
- Quantizations: [GPTQ](https://arxiv.org/abs/2210.17323), [AWQ](https://arxiv.org/abs/2306.00978), [AutoRound](https://arxiv.org/abs/2309.05516), INT4, INT8, and FP8
- Optimized CUDA kernels, including integration with FlashAttention and FlashInfer
- Speculative decoding
- Chunked prefill

vLLM is flexible and easy to use with:

- Seamless integration with popular Hugging Face models
- High-throughput serving with various decoding algorithms, including *parallel sampling*, *beam search*, and more
- Tensor, pipeline, data and expert parallelism support for distributed inference
- Streaming outputs
- OpenAI-compatible API server
- Support for NVIDIA GPUs, AMD CPUs and GPUs, Intel CPUs and GPUs, PowerPC CPUs, Arm CPUs, and TPU. Additionally, support for diverse hardware plugins such as Intel Gaudi, IBM Spyre and Huawei Ascend.
- Prefix caching support
- Multi-LoRA support

vLLM seamlessly supports most popular open-source models on HuggingFace, including:

- Transformer-like LLMs (e.g., Llama)
- Mixture-of-Expert LLMs (e.g., Mixtral, Deepseek-V2 and V3)
- Embedding Models (e.g., E5-Mistral)
- Multi-modal LLMs (e.g., LLaVA)

Find the full list of supported models [here](https://docs.vllm.ai/en/latest/models/supported_models.html).

## Getting Started

Install vLLM with `pip` or [from source](https://docs.vllm.ai/en/latest/getting_started/installation/gpu/index.html#build-wheel-from-source):

```bash
pip install vllm
```

Visit our [documentation](https://docs.vllm.ai/en/latest/) to learn more.

- [Installation](https://docs.vllm.ai/en/latest/getting_started/installation.html)
- [Quickstart](https://docs.vllm.ai/en/latest/getting_started/quickstart.html)
- [List of Supported Models](https://docs.vllm.ai/en/latest/models/supported_models.html)

## Contributing

We welcome and value any contributions and collaborations.
Please check out [Contributing to vLLM](https://docs.vllm.ai/en/latest/contributing/index.html) for how to get involved.

## Citation

If you use vLLM for your research, please cite our [paper](https://arxiv.org/abs/2309.06180):

```bibtex
@inproceedings{kwon2023efficient,
  title={Efficient Memory Management for Large Language Model Serving with PagedAttention},
  author={Woosuk Kwon and Zhuohan Li and Siyuan Zhuang and Ying Sheng and Lianmin Zheng and Cody Hao Yu and Joseph E. Gonzalez and Hao Zhang and Ion Stoica},
  booktitle={Proceedings of the ACM SIGOPS 29th Symposium on Operating Systems Principles},
  year={2023}
}
```

## Contact Us

<!-- --8<-- [start:contact-us] -->
- For technical questions and feature requests, please use GitHub [Issues](https://github.com/vllm-project/vllm/issues)
- For discussing with fellow users, please use the [vLLM Forum](https://discuss.vllm.ai)
- For coordinating contributions and development, please use [Slack](https://slack.vllm.ai)
- For security disclosures, please use GitHub's [Security Advisories](https://github.com/vllm-project/vllm/security/advisories) feature
- For collaborations and partnerships, please contact us at [collaboration@vllm.ai](mailto:collaboration@vllm.ai)
<!-- --8<-- [end:contact-us] -->

## Media Kit

- If you wish to use vLLM's logo, please refer to [our media kit repo](https://github.com/vllm-project/media-kit)
