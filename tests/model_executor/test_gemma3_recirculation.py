# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace

import pytest
import torch
from torch import nn

from vllm.model_executor.models.gemma3 import Gemma3Model
from vllm.model_executor.models.recirculation import RecirculationConfig

pytestmark = pytest.mark.skip_global_cleanup


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
        hidden_states = torch.full_like(residual, self.layer_idx + 1)
        if (kv_cache := kwargs.get("kv_cache")) is not None:
            kv_cache[self.layer_idx] = (hidden_states + residual).clone()
        return hidden_states, residual


class _FinalNorm(nn.Module):
    def forward(
        self,
        hidden_states: torch.Tensor,
        residual: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return hidden_states + residual, residual


def test_gemma3_recirculation_overwrites_only_upper_layer_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.gemma3.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = Gemma3Model.__new__(Gemma3Model)
    nn.Module.__init__(model)
    model.start_layer = 0
    model.end_layer = 3
    model.layers = nn.ModuleList([_AdditiveLayer(i, calls) for i in range(3)])
    model.norm = _FinalNorm()
    model.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
    )
    kv_cache: dict[int, torch.Tensor] = {}

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.tensor([[1.0, 0.0]]),
        kv_cache=kv_cache,
    )

    destination = torch.tensor([[2.0, 1.0]])
    source = torch.tensor([[4.0, 3.0]])
    recirculated = model.recirculation_config.mix(
        source, destination, torch.tensor([0])
    )
    torch.testing.assert_close(output, torch.tensor([[7.0, 6.0]]))
    torch.testing.assert_close(kv_cache[0], destination)
    torch.testing.assert_close(kv_cache[1], recirculated + 2.0)
    torch.testing.assert_close(kv_cache[2], recirculated + 5.0)
    assert calls == [0, 1, 2, 1, 2]


def test_identity_config_uses_baseline_stack_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_wavefront_model(monkeypatch)
    hf_config = SimpleNamespace(
        num_hidden_layers=3,
        recirculation_config={
            "source_layer": 1,
            "destination_layer": 0,
            "alpha": 0.0,
        },
    )
    model.recirculation_config = RecirculationConfig.from_hf_config(hf_config)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, 2),
    )

    torch.testing.assert_close(output, torch.full((1, 2), 6.0))
    assert calls == [0, 1, 2]


def _make_wavefront_model(
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[Gemma3Model, list[int]]:
    pp_group = SimpleNamespace(is_first_rank=True, is_last_rank=True)
    monkeypatch.setattr(
        "vllm.model_executor.models.gemma3.get_pp_group", lambda: pp_group
    )
    calls: list[int] = []
    model = Gemma3Model.__new__(Gemma3Model)
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


def test_gemma3_wavefront_warms_up_without_rerunning_upper_layers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_wavefront_model(monkeypatch)

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=torch.zeros(1, 2),
        recirculation_wavefront_warmup=True,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 6.0))
    torch.testing.assert_close(output[1:2], torch.ones(1, 2))
    assert calls == [0, 1, 2]


def test_gemma3_wavefront_batches_previous_and_current_upper_stacks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model, calls = _make_wavefront_model(monkeypatch)
    kv_cache: dict[int, torch.Tensor] = {}

    output = model.forward(
        input_ids=None,
        positions=torch.tensor([1]),
        inputs_embeds=torch.ones(1, 2),
        recirculation_wavefront_warmup=False,
        recirculation_wavefront_positions=torch.tensor([0, 1]),
        recirculation_wavefront_pending=torch.ones(1, 2),
        kv_cache=kv_cache,
    )

    torch.testing.assert_close(output[0:1], torch.full((1, 2), 7.0))
    torch.testing.assert_close(output[1:2], torch.full((1, 2), 2.0))
    assert kv_cache[0].shape[0] == 1
    assert kv_cache[1].shape[0] == 2
    assert kv_cache[2].shape[0] == 2
    assert calls == [0, 1, 2]


def test_serial_and_wavefront_match_outputs_and_effective_upper_cache(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    serial, _ = _make_wavefront_model(monkeypatch)
    serial.recirculation_config = RecirculationConfig(
        source_layer=1,
        destination_layer=0,
        alpha=0.2,
    )
    wavefront, _ = _make_wavefront_model(monkeypatch)
    serial_cache: dict[int, torch.Tensor] = {}
    wavefront_cache: dict[int, torch.Tensor] = {}
    inputs = [
        torch.tensor([[1.0, 0.0]]),
        torch.tensor([[0.0, 1.0]]),
        torch.tensor([[2.0, 1.0]]),
    ]

    serial.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=inputs[0],
        kv_cache=serial_cache,
    )
    serial_output = serial.forward(
        input_ids=None,
        positions=torch.tensor([1]),
        inputs_embeds=inputs[1],
        kv_cache=serial_cache,
    )
    serial_upper_cache = {
        layer_idx: serial_cache[layer_idx].clone() for layer_idx in (1, 2)
    }

    warmup = wavefront.forward(
        input_ids=None,
        positions=torch.tensor([0]),
        inputs_embeds=inputs[0],
        recirculation_wavefront_warmup=True,
        kv_cache=wavefront_cache,
    )
    wavefront_output = wavefront.forward(
        input_ids=None,
        positions=torch.tensor([1]),
        inputs_embeds=inputs[1],
        recirculation_wavefront_warmup=False,
        recirculation_wavefront_positions=torch.tensor([0, 1]),
        recirculation_wavefront_pending=warmup[1:],
        kv_cache=wavefront_cache,
    )
    wavefront.forward(
        input_ids=None,
        positions=torch.tensor([2]),
        inputs_embeds=inputs[2],
        recirculation_wavefront_warmup=False,
        recirculation_wavefront_positions=torch.tensor([1, 2]),
        recirculation_wavefront_pending=wavefront_output[1:],
        kv_cache=wavefront_cache,
    )

    torch.testing.assert_close(wavefront_output[:1], serial_output)
    for layer_idx in (1, 2):
        torch.testing.assert_close(
            wavefront_cache[layer_idx][0:1],
            serial_upper_cache[layer_idx],
        )
