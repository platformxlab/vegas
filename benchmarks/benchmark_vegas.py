from datasets import load_dataset
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams


_SEED = 42
_MODEL_NAME = "Qwen/Qwen3-8B"
_NUM_SPECULATIVE_TOKENS = 6
_SAMPLING_ARGS = {
    "max_tokens": 40960,
    "temperature": 0.6,
    "top_p": 0.95,
    "top_k": 20,
}
_MAX_NUM_SEQS = 128


tokenizer = AutoTokenizer.from_pretrained(_MODEL_NAME, trust_remote_code=True)


def apply_chat_template(prompt: str) -> str:
    messages = [
        {"role": "user", "content": prompt}
    ]
    return tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )


# Use Vegas for speculative decoding.
speculative_config = {
    "method": "sparse_attn",
    "num_speculative_tokens": _NUM_SPECULATIVE_TOKENS,
    "sparse_attn_algorithm": "vegas",   # Also supported: "streamingllm"
    "sparse_attn_ratio": 0.07,
}
# speculative_config = None     # Uncomment to disable speculative decoding.

llm = LLM(
    model=_MODEL_NAME,
    max_num_seqs=_MAX_NUM_SEQS,
    max_model_len=_SAMPLING_ARGS["max_tokens"],
    seed=_SEED,
    speculative_config=speculative_config,
)
sampling_params = SamplingParams(**_SAMPLING_ARGS)
questions = load_dataset("math-ai/aime25")["test"]["problem"]
prompts = [apply_chat_template(q) for q in questions] * 32
outputs = llm.generate(prompts=prompts, sampling_params=sampling_params)
