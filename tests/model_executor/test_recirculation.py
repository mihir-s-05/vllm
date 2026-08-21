# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from typing import Any, cast

import pytest
import torch
from torch import nn

from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model
from vllm.model_executor.models.gemma3 import Gemma3ForCausalLM, Gemma3Model
from vllm.model_executor.models.gemma4 import Gemma4Model
from vllm.model_executor.models.glm4_moe import Glm4MoeModel
from vllm.model_executor.models.glm4_moe_lite import Glm4MoeLiteModel
from vllm.model_executor.models.gpt_oss import GptOssModel
from vllm.model_executor.models.interfaces import supports_recirculation
from vllm.model_executor.models.kimi_audio import KimiAudioForConditionalGeneration
from vllm.model_executor.models.kimi_k25 import KimiK25ForConditionalGeneration
from vllm.model_executor.models.kimi_vl import KimiVLForConditionalGeneration
from vllm.model_executor.models.llama import LlamaForCausalLM, LlamaModel
from vllm.model_executor.models.llama4 import Llama4Model
from vllm.model_executor.models.mimo_v2 import MiMoV2Model
from vllm.model_executor.models.minimax_m2 import MiniMaxM2Model
from vllm.model_executor.models.mistral import MistralModel
from vllm.model_executor.models.mixtral import MixtralModel
from vllm.model_executor.models.qwen2 import Qwen2Model
from vllm.model_executor.models.qwen3 import Qwen3Model
from vllm.model_executor.models.qwen3_5 import Qwen3_5Model
from vllm.model_executor.models.qwen3_moe import Qwen3MoeModel
from vllm.model_executor.models.qwen3_next import (
    Qwen3NextDecoderLayer,
    Qwen3NextModel,
)
from vllm.model_executor.models.recirculation import (
    RecirculationCapabilities,
    RecirculationConfig,
    RecirculationDecoderMixin,
)
from vllm.model_executor.models.step3p5 import Step3p5Model
from vllm.models.kimi_k3.amd import linear as kimi_k3_amd_linear
from vllm.models.kimi_k3.nvidia import model as kimi_k3_model
from vllm.models.kimi_k3.nvidia.model import (
    KimiK3ForConditionalGeneration,
    KimiLinearForCausalLM,
    KimiLinearModel,
)

pytestmark = pytest.mark.skip_global_cleanup


def test_recirculation_mix_matches_destination_norm() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.25,
    )
    source = torch.tensor([[3.0, 4.0]])
    destination = torch.tensor([[0.0, 10.0]])

    mixed = config.mix(source, destination, torch.tensor([8]))

    torch.testing.assert_close(mixed, torch.tensor([[1.5, 9.5]]))


def test_recirculation_mix_ramps_convex_coefficients_by_position() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        ramp_tokens=10,
    )
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0]])

    mixed = config.mix(source, destination, torch.tensor([0, 5, 10]))

    expected = torch.tensor([[0.0, 1.0], [0.1, 0.9], [0.2, 0.8]])
    torch.testing.assert_close(mixed, expected)


def test_recirculation_mix_uses_first_mrope_position_row() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        ramp_tokens=10,
    )
    source = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0], [0.0, 1.0]])
    positions = torch.tensor([[5, 10], [50, 60], [70, 80]])

    mixed = config.mix(source, destination, positions)

    assert mixed.shape == source.shape
    torch.testing.assert_close(
        mixed,
        torch.tensor([[0.1, 0.9], [0.2, 0.8]]),
    )


def test_recirculation_mix_supports_nonconvex_beta() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        beta=1.0,
    )
    source = torch.tensor([[1.0, 0.0]])
    destination = torch.tensor([[0.0, 1.0]])

    mixed = config.mix(source, destination, torch.tensor([4]))

    torch.testing.assert_close(mixed, torch.tensor([[0.2, 1.0]]))


@pytest.mark.parametrize("beta", [None, 1.0])
def test_recirculation_config_disables_identity_mix(beta: float | None) -> None:
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        recirculation_config={
            "source_layer": 2,
            "destination_layer": 0,
            "alpha": 0.0,
            "beta": beta,
        },
    )

    assert RecirculationConfig.from_hf_config(hf_config) is None


@pytest.mark.parametrize(
    "raw_config",
    [
        {"source_layer": 2, "destination_layer": 2},
        {"source_layer": 3, "destination_layer": 0},
        {"source_layer": 2, "destination_layer": 0, "alpha": 1.1},
        {"source_layer": 2, "destination_layer": 0, "ramp_tokens": -1},
        {"source_layer": 2, "destination_layer": 0, "wavefront": 1},
        {
            "source_layer": 2,
            "destination_layer": 0,
            "attn_res_mode": "inverse",
        },
        {"source_layer": 2, "destination_layer": 0, "unexpected": True},
    ],
)
def test_recirculation_config_rejects_invalid_values(raw_config: dict) -> None:
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        recirculation_config=raw_config,
    )

    with pytest.raises(ValueError):
        RecirculationConfig.from_hf_config(hf_config)


class _AdditiveLayer(nn.Module):
    def __init__(self, layer_idx: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.calls = calls

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(self.layer_idx)
        residual = hidden_states if residual is None else hidden_states + residual
        return torch.full_like(residual, self.layer_idx + 1), residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
    ) -> tuple[torch.Tensor, None]:
        assert residual is not None
        return hidden_states + residual, None


class _KimiAdditiveLayer(nn.Module):
    def __init__(self, layer_idx: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.calls = calls

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor | None,
        **kwargs,
    ) -> tuple[torch.Tensor, None, torch.Tensor]:
        self.calls.append(self.layer_idx)
        residual = hidden_states if residual is None else hidden_states + residual
        hidden_states = torch.full_like(residual, self.layer_idx + 1)
        return hidden_states, None, residual


class _FakeAttnResLayer(nn.Module):
    def __init__(self, layer_idx: int, block_size: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.block_write_idx = layer_idx // block_size
        self.is_block_write_layer = layer_idx % block_size == 0
        self.prev_valid_blocks = (layer_idx + block_size - 1) // block_size
        self.calls = calls
        self.self_attention_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.ones(4)), variance_epsilon=1e-6
        )
        self.self_attention_res_proj = SimpleNamespace(
            weight=nn.Parameter(torch.zeros(1, 4))
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor | None,
        residual: torch.Tensor,
        prefix_sum: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        self.calls.append(self.layer_idx)
        prefix_sum = prefix_sum if hidden_states is None else prefix_sum + hidden_states
        if self.is_block_write_layer:
            residual[:, self.block_write_idx, :].copy_(prefix_sum)
        attention_output = prefix_sum + float(self.layer_idx + 1)
        prefix_sum = (
            attention_output
            if self.is_block_write_layer
            else prefix_sum + attention_output
        )
        mlp_output = torch.full_like(prefix_sum, 0.25 * (self.layer_idx + 1))
        return mlp_output, prefix_sum, residual


class _FakeAmdAttnResLayer(nn.Module):
    def __init__(self, layer_idx: int, block_size: int, calls: list[int]) -> None:
        super().__init__()
        self.layer_idx = layer_idx
        self.block_write_idx = layer_idx // block_size
        self.is_block_write_layer = layer_idx % block_size == 0
        self.prev_valid_blocks = (layer_idx + block_size - 1) // block_size
        self.calls = calls
        self.self_attention_res_norm = SimpleNamespace(
            weight=nn.Parameter(torch.ones(4)), variance_epsilon=1e-6
        )
        self.self_attention_res_proj = SimpleNamespace(
            weight=nn.Parameter(torch.zeros(1, 4))
        )

    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.calls.append(self.layer_idx)
        if self.is_block_write_layer:
            residual[:, self.block_write_idx, :].copy_(hidden_states)
        attention_output = hidden_states + float(self.layer_idx + 1)
        prefix_sum = (
            attention_output
            if self.is_block_write_layer
            else hidden_states + attention_output
        )
        prefix_sum = prefix_sum + 0.25 * (self.layer_idx + 1)
        return prefix_sum, residual


def _make_llama_model(
    monkeypatch: pytest.MonkeyPatch,
    model_type: type[LlamaModel] = LlamaModel,
) -> tuple[LlamaModel, list[int]]:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.llama.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = cast(LlamaModel, object.__new__(model_type))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.layers = nn.ModuleList([_AdditiveLayer(i, calls) for i in range(3)])
    model.norm = _FinalNorm()
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
        wavefront=True,
    )
    return model, calls


def test_llama_uses_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_llama_top_level_advertises_engine_capability(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, _ = _make_llama_model(monkeypatch)
    causal_lm = LlamaForCausalLM.__new__(LlamaForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model

    assert supports_recirculation(causal_lm)


def test_gemma3_top_level_advertises_engine_capability() -> None:
    model = cast(Gemma3Model, object.__new__(Gemma3Model))
    causal_lm = Gemma3ForCausalLM.__new__(Gemma3ForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model

    assert supports_recirculation(causal_lm)


@pytest.mark.parametrize(
    ("is_causal", "start_layer", "end_layer", "error"),
    [
        (False, 0, 3, "causal attention"),
        (True, 1, 3, "pipeline parallelism"),
        (True, 0, 2, "pipeline parallelism"),
    ],
)
def test_model_initialization_rejects_unsupported_execution(
    is_causal: bool,
    start_layer: int,
    end_layer: int,
    error: str,
) -> None:
    model = cast(LlamaModel, object.__new__(LlamaModel))
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity() for _ in range(3)])
    config = SimpleNamespace(
        is_causal=is_causal,
        num_hidden_layers=3,
        recirculation_config={
            "source_layer": 1,
            "destination_layer": 0,
            "alpha": 0.2,
        },
    )

    with pytest.raises(ValueError, match=error):
        model._init_recirculation(config, start_layer, end_layer)


def test_attnres_mode_requires_model_capability() -> None:
    model = cast(LlamaModel, object.__new__(LlamaModel))
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([nn.Identity() for _ in range(3)])
    config = SimpleNamespace(
        num_hidden_layers=3,
        recirculation_config={
            "source_layer": 1,
            "destination_layer": 0,
            "attn_res_mode": "prefix",
        },
    )

    with pytest.raises(ValueError, match="does not support attn_res_mode"):
        model._init_recirculation(config, 0, 3)


@pytest.mark.parametrize(
    "wrapper_type",
    [
        KimiAudioForConditionalGeneration,
        KimiK25ForConditionalGeneration,
        KimiK3ForConditionalGeneration,
        KimiVLForConditionalGeneration,
    ],
)
def test_kimi_wrappers_advertise_nested_text_capability(
    wrapper_type: type[nn.Module],
) -> None:
    capabilities = RecirculationCapabilities(adapter="nested_kimi", wavefront=False)

    class NestedLanguageModel(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.received_wavefront: tuple[object, object, object] | None = None

        def get_recirculation_capabilities(self) -> RecirculationCapabilities:
            return capabilities

        def forward(
            self,
            input_ids: torch.Tensor | None,
            positions: torch.Tensor,
            intermediate_tensors: object = None,
            inputs_embeds: torch.Tensor | None = None,
            recirculation_wavefront_warmup: bool | None = None,
            recirculation_wavefront_positions: torch.Tensor | None = None,
            recirculation_wavefront_pending: torch.Tensor | None = None,
        ) -> torch.Tensor:
            self.received_wavefront = (
                recirculation_wavefront_warmup,
                recirculation_wavefront_positions,
                recirculation_wavefront_pending,
            )
            return positions

    nested_model = NestedLanguageModel()
    if wrapper_type is KimiAudioForConditionalGeneration:

        class AudioLanguageModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.model = nested_model

            def get_recirculation_capabilities(
                self,
            ) -> RecirculationCapabilities:
                return capabilities

        language_model: nn.Module = AudioLanguageModel()
    else:
        language_model = nested_model
    model: Any = object.__new__(wrapper_type)
    nn.Module.__init__(model)
    model.language_model = language_model

    assert supports_recirculation(model)
    assert model.get_recirculation_capabilities() is capabilities

    positions = torch.tensor([3])
    wavefront_positions = torch.tensor([2, 3])
    wavefront_pending = torch.ones(1, 2)
    output = model.forward(
        input_ids=torch.tensor([1]),
        positions=positions,
        recirculation_wavefront_warmup=False,
        recirculation_wavefront_positions=wavefront_positions,
        recirculation_wavefront_pending=wavefront_pending,
    )

    assert output is positions
    assert nested_model.received_wavefront is not None
    warmup, received_positions, received_pending = nested_model.received_wavefront
    assert warmup is False
    assert received_positions is wavefront_positions
    assert received_pending is wavefront_pending


def test_kimi_linear_top_level_advertises_serial_engine_capability() -> None:
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    causal_lm = KimiLinearForCausalLM.__new__(KimiLinearForCausalLM)
    nn.Module.__init__(causal_lm)
    causal_lm.model = model

    assert supports_recirculation(causal_lm)
    capabilities = causal_lm.get_recirculation_capabilities()
    assert capabilities is not None
    assert capabilities.adapter == "kimi_linear_hybrid"
    assert not capabilities.wavefront


def test_mistral_delegates_to_shared_wavefront_execution(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_llama_model(monkeypatch, MistralModel)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        intermediate_tensors=None,
        inputs_embeds=torch.zeros(1, 2),
        t_cond=None,
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_unvalidated_llama_subclass_does_not_inherit_adapter() -> None:
    class UnvalidatedLlamaModel(LlamaModel):
        pass

    model = UnvalidatedLlamaModel.__new__(UnvalidatedLlamaModel)

    assert not model.has_recirculation_adapter()


def test_engine_capability_rejects_incomplete_forward() -> None:
    class IncompleteModel:
        supports_recirculation = True

        def forward(
            self, input_ids: torch.Tensor, positions: torch.Tensor
        ) -> torch.Tensor:
            return input_ids

    assert not supports_recirculation(IncompleteModel())


@pytest.mark.parametrize(
    ("model_type", "adapter", "wavefront"),
    [
        (DeepseekV2Model, "deepseek_moe", False),
        (Gemma4Model, "gemma4", True),
        (Glm4MoeModel, "glm4_moe", True),
        (Glm4MoeLiteModel, "glm4_moe_lite", False),
        (GptOssModel, "gpt_oss_moe", False),
        (KimiLinearModel, "kimi_linear_hybrid", False),
        (Llama4Model, "llama4_moe", True),
        (MiniMaxM2Model, "minimax_m2_moe", True),
        (MiMoV2Model, "mimo_v2_moe", True),
        (MixtralModel, "mixtral", True),
        (Qwen2Model, "qwen2", True),
        (Qwen3Model, "qwen3", True),
        (Qwen3MoeModel, "qwen3_moe", True),
        (Qwen3NextModel, "qwen3_next_hybrid", False),
        (Qwen3_5Model, "qwen3_5_hybrid", False),
        (Step3p5Model, "step3p5_moe", True),
    ],
)
def test_reviewed_family_capabilities(
    model_type: type[RecirculationDecoderMixin],
    adapter: str,
    wavefront: bool,
) -> None:
    model = cast(RecirculationDecoderMixin, object.__new__(model_type))
    if isinstance(model, Gemma4Model):
        model.hidden_size_per_layer_input = 0

    capabilities = model.get_recirculation_capabilities()

    assert capabilities is not None
    assert capabilities.adapter == adapter
    assert capabilities.serial
    assert capabilities.wavefront is wavefront


def test_gemma4_per_layer_embeddings_are_serial_only() -> None:
    model = cast(Gemma4Model, object.__new__(Gemma4Model))
    model.hidden_size_per_layer_input = 16

    capabilities = model.get_recirculation_capabilities()

    assert capabilities is not None
    assert capabilities.serial
    assert not capabilities.wavefront


def test_nested_text_config_is_used_for_recirculation() -> None:
    text_config = SimpleNamespace(
        num_hidden_layers=4,
        recirculation_config={
            "source_layer": 2,
            "destination_layer": 1,
            "alpha": 0.1,
        },
    )
    wrapper_config = SimpleNamespace(get_text_config=lambda: text_config)

    config = RecirculationConfig.from_hf_config(wrapper_config)

    assert config is not None
    assert config.source_layer == 2
    assert config.destination_layer == 1


def test_qwen_next_restores_active_gdn_state_before_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeLinearAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = "model.layers.0.linear_attn"
            self.kv_cache = (
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
            )

    layer = cast(Qwen3NextDecoderLayer, object.__new__(Qwen3NextDecoderLayer))
    nn.Module.__init__(layer)
    layer.linear_attn = FakeLinearAttention()
    model = cast(Qwen3NextModel, object.__new__(Qwen3NextModel))
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([layer])
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=torch.tensor([2, 0], dtype=torch.int64),
        num_prefills=1,
        num_decodes=1,
    )
    context = SimpleNamespace(attn_metadata={"model.layers.0.linear_attn": metadata})
    monkeypatch.setattr(
        "vllm.model_executor.models.qwen3_next.get_forward_context",
        lambda: context,
    )

    snapshot = model._capture_recirculation_layer_state(0)
    untouched = tuple(state[1].clone() for state in layer.linear_attn.kv_cache)
    for state in layer.linear_attn.kv_cache:
        state.add_(100)
    model._restore_recirculation_layer_state(0, snapshot)

    assert snapshot is not None
    state_indices, captured = snapshot
    for cache, saved, untouched_row in zip(
        layer.linear_attn.kv_cache, captured, untouched
    ):
        torch.testing.assert_close(cache.index_select(0, state_indices), saved)
        torch.testing.assert_close(cache[1], untouched_row + 100)


def test_kimi_linear_uses_serial_recirculation() -> None:
    calls: list[int] = []
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.use_attn_res = False
    model.layers = nn.ModuleList([_KimiAdditiveLayer(i, calls) for i in range(3)])
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
    )

    output = model._forward_recirculation(
        positions=torch.tensor([0]),
        hidden_states=torch.zeros(1, 2),
        residual=None,
        wavefront_warmup=None,
        wavefront_positions=None,
        wavefront_pending=None,
    )

    torch.testing.assert_close(output, torch.full((1, 2), 6.0))
    assert calls == [0, 1, 2, 1, 2]


def test_kimi_linear_restores_active_kda_state_before_rerun(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FakeAttention(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.prefix = "model.layers.0.self_attn"
            self.kv_cache = (
                torch.arange(12, dtype=torch.float32).reshape(3, 4),
                torch.arange(18, dtype=torch.float32).reshape(3, 2, 3),
            )

    class FakeLayer(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.self_attn = FakeAttention()

    monkeypatch.setattr(kimi_k3_model, "KimiK3DeltaAttention", FakeAttention)
    monkeypatch.setattr(
        kimi_k3_model, "KimiLinearGatedDeltaNetAttention", FakeAttention
    )
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    nn.Module.__init__(model)
    model.layers = nn.ModuleList([FakeLayer()])
    metadata = SimpleNamespace(
        spec_sequence_masks=None,
        non_spec_state_indices_tensor=torch.tensor([2, 0], dtype=torch.int64),
        num_prefills=1,
        num_decodes=1,
    )
    context = SimpleNamespace(attn_metadata={"model.layers.0.self_attn": metadata})
    monkeypatch.setattr(kimi_k3_model, "get_forward_context", lambda: context)

    snapshot = model._capture_recirculation_layer_state(0)
    attention = model.layers[0].self_attn
    untouched = tuple(state[1].clone() for state in attention.kv_cache)
    for state in attention.kv_cache:
        state.add_(100)
    model._restore_recirculation_layer_state(0, snapshot)

    assert snapshot is not None
    state_indices, captured = snapshot
    for cache, saved, untouched_row in zip(attention.kv_cache, captured, untouched):
        torch.testing.assert_close(cache.index_select(0, state_indices), saved)
        torch.testing.assert_close(cache[1], untouched_row + 100)


@pytest.mark.parametrize("mode", ["prefix", "bank", "broadcast"])
def test_kimi_k3_accepts_explicit_attnres_recirculation_mode(mode: str) -> None:
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    model.use_attn_res = True
    model.use_sequence_parallel = False

    model._validate_recirculation_model_config(
        object(),
        RecirculationConfig(
            source_layer=2,
            destination_layer=0,
            attn_res_mode=cast(Any, mode),
        ),
    )


def test_kimi_k3_requires_explicit_attnres_recirculation_mode() -> None:
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    model.use_attn_res = True
    model.use_sequence_parallel = False

    with pytest.raises(ValueError, match="requires attn_res_mode"):
        model._validate_recirculation_model_config(
            object(), RecirculationConfig(source_layer=2, destination_layer=0)
        )


def test_attnres_mode_is_rejected_without_attnres() -> None:
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    model.use_attn_res = False
    model.use_sequence_parallel = False

    with pytest.raises(ValueError, match="requires Kimi-K3 AttnRes"):
        model._validate_recirculation_model_config(
            object(),
            RecirculationConfig(
                source_layer=2,
                destination_layer=0,
                attn_res_mode="prefix",
            ),
        )


@pytest.mark.parametrize("mode", ["prefix", "bank", "broadcast"])
def test_attnres_state_mappings(mode: str) -> None:
    config = RecirculationConfig(
        source_layer=4,
        destination_layer=2,
        alpha=1.0,
        attn_res_mode=cast(Any, mode),
    )
    source = torch.tensor([[0.0, 2.0]])
    destination = torch.tensor([[3.0, 0.0]])
    original_bank = torch.tensor([[[1.0, 0.0], [0.0, 2.0], [3.0, 0.0]]])
    bank = original_bank.clone()

    recirculated = config.mix_attn_res_state(
        source,
        destination,
        bank,
        positions=torch.tensor([7]),
        source_num_blocks=3,
        destination_num_blocks=2,
    )

    torch.testing.assert_close(recirculated, torch.tensor([[0.0, 3.0]]))
    if mode == "prefix":
        torch.testing.assert_close(bank, original_bank)
    elif mode == "bank":
        torch.testing.assert_close(
            bank,
            torch.tensor([[[0.0, 1.0], [2.0, 0.0], [3.0, 0.0]]]),
        )
    else:
        torch.testing.assert_close(
            bank,
            torch.tensor([[[0.0, 3.0], [0.0, 3.0], [3.0, 0.0]]]),
        )


def test_recirculation_ramp_broadcasts_over_attnres_blocks() -> None:
    config = RecirculationConfig(
        source_layer=2,
        destination_layer=0,
        alpha=0.2,
        ramp_tokens=10,
    )
    source = torch.zeros(2, 3, 4)
    source[..., 0] = 1.0
    destination = torch.zeros(2, 3, 4)
    destination[..., 1] = 1.0

    mixed = config.mix(source, destination, torch.tensor([5, 10]))

    assert mixed.shape == source.shape
    expected = destination.clone()
    expected[0, :, 0] = 0.1
    expected[0, :, 1] = 0.9
    expected[1, :, 0] = 0.2
    expected[1, :, 1] = 0.8
    torch.testing.assert_close(mixed, expected)


@pytest.mark.parametrize("mode", ["prefix", "bank", "broadcast"])
def test_kimi_k3_runs_attnres_recirculation_without_weights(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    def fake_attn_res(
        prefix: torch.Tensor,
        delta: torch.Tensor | None,
        blocks: torch.Tensor,
        norm_weight: torch.Tensor,
        qk_weight: torch.Tensor,
        output_norm_weight: torch.Tensor | None,
        num_blocks: int,
        block_write_idx: int,
        eps: float,
        output_norm_eps: float,
    ) -> torch.Tensor:
        del norm_weight, qk_weight, output_norm_weight, eps, output_norm_eps
        prefix = prefix if delta is None else prefix + delta
        if block_write_idx >= 0:
            blocks[:, block_write_idx, :].copy_(prefix)
        sources = torch.cat((blocks[:, :num_blocks, :], prefix.unsqueeze(1)), dim=1)
        return sources.mean(dim=1)

    monkeypatch.setattr(kimi_k3_model, "attn_res", fake_attn_res)
    calls: list[int] = []
    block_size = 2
    model = cast(KimiLinearModel, object.__new__(KimiLinearModel))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 5
    model.use_attn_res = True
    model.use_sequence_parallel = False
    model.attn_res_block_size = block_size
    model.num_attn_res_blocks = 3
    model.layers = nn.ModuleList(
        [_FakeAttnResLayer(i, block_size, calls) for i in range(model.end_layer)]
    )
    model.output_attn_res_norm = SimpleNamespace(
        weight=nn.Parameter(torch.ones(4)), variance_epsilon=1e-6
    )
    model.output_attn_res_proj = SimpleNamespace(weight=nn.Parameter(torch.zeros(1, 4)))
    model.recirculation_config = RecirculationConfig(
        source_layer=3,
        destination_layer=1,
        alpha=0.2,
        attn_res_mode=cast(Any, mode),
    )

    output = model._forward_recirculation(
        positions=torch.tensor([4]),
        hidden_states=torch.ones(1, 4),
        residual=None,
        wavefront_warmup=None,
        wavefront_positions=None,
        wavefront_pending=None,
    )

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()
    assert calls == [0, 1, 2, 3, 4, 2, 3, 4]


@pytest.mark.parametrize("mode", ["prefix", "bank", "broadcast"])
def test_kimi_k3_amd_runs_attnres_recirculation_without_weights(
    monkeypatch: pytest.MonkeyPatch,
    mode: str,
) -> None:
    def fake_apply_attn_res(
        prefix_sum: torch.Tensor,
        block_residual: torch.Tensor,
        proj: object,
        norm: object,
        num_valid_blocks: int,
        **kwargs,
    ) -> torch.Tensor:
        del proj, norm, kwargs
        sources = torch.cat(
            (
                block_residual[:, :num_valid_blocks, :],
                prefix_sum.unsqueeze(1),
            ),
            dim=1,
        )
        return sources.mean(dim=1)

    monkeypatch.setattr(kimi_k3_amd_linear, "_apply_attn_res", fake_apply_attn_res)
    calls: list[int] = []
    block_size = 2
    model_type = kimi_k3_amd_linear.KimiLinearModel
    model = cast(Any, object.__new__(model_type))
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 5
    model.config = SimpleNamespace(attn_res_block_size=block_size)
    model.layers = nn.ModuleList(
        [_FakeAmdAttnResLayer(i, block_size, calls) for i in range(model.end_layer)]
    )
    model.output_attn_res_norm = SimpleNamespace(
        weight=nn.Parameter(torch.ones(4)), variance_epsilon=1e-6
    )
    model.output_attn_res_proj = SimpleNamespace(weight=nn.Parameter(torch.zeros(1, 4)))
    model.recirculation_config = RecirculationConfig(
        source_layer=3,
        destination_layer=1,
        alpha=0.2,
        attn_res_mode=cast(Any, mode),
    )

    output = model._forward_recirculation(
        positions=torch.tensor([4]),
        hidden_states=torch.ones(1, 4),
        residual=None,
        wavefront_warmup=None,
        wavefront_positions=None,
        wavefront_pending=None,
    )

    assert output.shape == (1, 4)
    assert torch.isfinite(output).all()
    assert calls == [0, 1, 2, 3, 4, 2, 3, 4]
