# Recirculation

!!! warning
    Recirculation support is experimental. Compatible model implementations
    must opt in to the engine capability; the optimized wavefront path has the
    additional restrictions listed below.

[Recirculation](https://arxiv.org/abs/2608.17981) feeds a norm-matched deep
residual-stream activation back into a shallower layer. vLLM returns the logits
from the token block's normal pass, then reruns the layers above the destination
with the mixed residual. The rerun overwrites the block's upper-layer KV cache,
so later tokens attend to the recirculated representations.

Enable fixed recirculation through `--hf-overrides`. For Gemma 3 1B PT, the
paper's perplexity configuration is:

```bash
vllm serve google/gemma-3-1b-pt \
  --hf-overrides '{
    "recirculation_config": {
      "source_layer": 11,
      "destination_layer": 4,
      "alpha": 0.15,
      "ramp_tokens": 10
    }
  }' \
  --long-prefill-token-threshold 1
```

When `beta` is omitted, vLLM uses the convex coefficient
`beta = 1 - alpha`. The ramp scales `alpha` from zero to its configured value
over the first `ramp_tokens` positions and adjusts the convex `beta`
accordingly.

An identity configuration with `alpha = 0` and `beta` omitted or set to `1`
disables Recirculation. It therefore agrees exactly with the baseline and does
not incur a redundant upper-layer pass.

## Model capability

The scheduler, recurrent-state buffers, KV-slot remapping, and CUDA-graph
specialization are engine-level features. Model implementations opt in through
the `SupportsRecirculation` interface and use a shared residual-decoder
execution mixin. This keeps scheduling behavior common while allowing a model
family to override its layer execution with a specialized implementation.

The reviewed adapters cover Gemma 3/4, Llama and direct aliases, Llama 4,
Mistral, Mixtral, Qwen 2, Qwen 3, Qwen 3 MoE, DeepSeek V2/V3, GLM-4 MoE,
GLM-4.7-Flash, GPT-OSS, MiniMax-M2, MiMo-V2, and Step-3.5. Architectures with
incompatible attention backends use the serial path; ordinary residual
decoders can also use wavefront execution. An unreviewed subclass does not
inherit support automatically and fails during model loading when
Recirculation is requested.

DeepSeek-V3.2 and GLM-5-family DSA decoders use a dedicated serial adapter that
reconstructs tensor-parallel residuals before mixing and does not reuse the
normal pass's DSA attention input during the rerun.

## Wavefront execution

Set `"wavefront": true` to execute exact tokenwise Recirculation as a
two-token wavefront. After the first-token warmup, the layers above the
destination process the previous token's recurrent state and the current
token's normal state in one layer call. Each upper attention layer first
overwrites the previous token's KV entry, so the current token attends to the
same recurrent cache that the serial implementation would have produced.

```bash
vllm serve google/gemma-3-1b-pt \
  --hf-overrides '{
    "recirculation_config": {
      "source_layer": 11,
      "destination_layer": 4,
      "alpha": 0.15,
      "ramp_tokens": 10,
      "wavefront": true
    }
  }' \
  --max-num-seqs 1 \
  --long-prefill-token-threshold 1 \
  --no-enable-prefix-caching
```

Wavefront mode captures a dedicated one-token CUDA graph whose upper stack has
the internal two-token batch. Torch compilation remains enabled unless
`--enforce-eager` is also set.

The paper reports the following fixed configurations for its pretrained-model
perplexity evaluation:

| Model | Source | Destination | Alpha | Beta | Ramp tokens |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gemma 3 1B PT | 11 | 4 | 0.15 | `1 - alpha` | 10 |
| Gemma 3 4B PT | 18 | 9 | 0.15 | 1.0 | 0 |
| Gemma 3 12B PT | 35 | 16 | 0.15 | 1.0 | 0 |

Set `"beta": 1.0` explicitly for the non-convex 4B and 12B configurations.

## Exact and blockwise execution

Each forward call recirculates the scheduled tokens as one block. A one-token
block implements the paper's tokenwise recurrence. Larger prefill chunks use
the blockwise approximation proposed in the paper: logits within a block come
from the normal pass, and the recirculated cache affects later blocks.

For exact tokenwise evaluation, set `--long-prefill-token-threshold 1` and do
not enable speculative decoding. For throughput experiments, increase the
threshold to sweep the block size and measure the quality-throughput tradeoff.

## Current restrictions

- Multimodal wrappers are not yet supported.
- Gemma 4 YOCO fast prefill is unsupported. Gemma 4 per-layer embeddings are
  serial only.
- The DeepSeek-V3.2/GLM-5 DSA adapter is serial only and rejects sequence
  parallel execution.
- Pipeline parallelism is not supported.
- Only fixed scalar coefficients and source norm matching are implemented.
- Wavefront execution currently requires one sequence, one scheduled token per
  step, FlashAttention, no prefix caching or speculative decoding, and no data,
  decode-context, sequence, or pipeline parallelism.
- Serial execution remains available when `"wavefront"` is omitted or false.
