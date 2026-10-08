import os
from pathlib import Path

import pytest
import torch

import robust_cfi_is.benchmarks.resnet as resnet_module
from robust_cfi_is.benchmarks.resnet import (
    G7_DATA_SHA256,
    G7ResNetEvent,
    RESNET18_PARAMETER_NAMES,
    explicit_device,
    validate_g7_payload,
    verify_file,
)


def _payload():
    return {
        "X": torch.zeros(24, 3, 224, 224, dtype=torch.float16),
        "Y": torch.tensor([0] * 10 + [1] * 10 + [2] * 4, dtype=torch.float16),
    }


def test_g7_payload_envelope_and_device_are_strict(tmp_path):
    x, y = validate_g7_payload(_payload())
    assert x.is_contiguous() and y.is_contiguous()
    assert explicit_device("cpu") == torch.device("cpu")
    for value in ("cuda", "auto", "cuda:x", "mps"):
        with pytest.raises(ValueError, match="exactly"):
            explicit_device(value)
    bad = _payload()
    bad["extra"] = torch.tensor(1)
    with pytest.raises(ValueError, match="exactly X and Y"):
        validate_g7_payload(bad)
    path = tmp_path / "bad.pt"
    path.write_bytes(b"not an asset")
    with pytest.raises(ValueError, match="size mismatch"):
        verify_file(path, name="g7 tensor", size=7_227_208, sha256=G7_DATA_SHA256)


def test_g7_problem_threshold_and_strict_event_direction(monkeypatch):
    class FakeEvent:
        asset_record = object()

        def __init__(self, *_args, **_kwargs):
            pass

        def __call__(self, samples):
            return samples[:, 0]

    monkeypatch.setattr(resnet_module, "G7ResNetEvent", FakeEvent)
    problem, _ = resnet_module.get_g7_benchmark("data", "weights", device="cpu")
    assert problem.dimension == 62
    assert problem.gamma == 5.157151232471544
    values = torch.zeros(2, 62)
    values[:, 0] = torch.tensor([problem.gamma, problem.gamma + 1e-6])
    assert problem.indicator(values).tolist() == [False, True]


@pytest.mark.skipif(
    not (os.environ.get("ROBUST_CFI_IS_G7_DATA") and os.environ.get("ROBUST_CFI_IS_RESNET_WEIGHTS")),
    reason="certified local g7 assets were not supplied",
)
def test_g7_direct_authoritative_cpu_parity(monkeypatch, tmp_path):
    # Any accidental torchvision download is a hard test failure.
    monkeypatch.setattr(
        torch.hub, "load_state_dict_from_url",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network call")),
    )
    data = Path(os.environ["ROBUST_CFI_IS_G7_DATA"])
    weights = Path(os.environ["ROBUST_CFI_IS_RESNET_WEIGHTS"])
    event = G7ResNetEvent(data, weights, "cpu")

    from torchvision.models import resnet18
    payload = torch.load(data, map_location="cpu", weights_only=True)
    model = resnet18(weights=None)
    model.load_state_dict(torch.load(weights, map_location="cpu", weights_only=True), strict=True)
    model.eval()
    names = tuple(name for name, _ in model.named_parameters())
    assert names == RESNET18_PARAMETER_NAMES
    criterion = torch.nn.CrossEntropyLoss(reduction="mean")
    loss = criterion(model(payload["X"].float()), payload["Y"].long())
    loss.backward()
    reference = torch.stack([(p.grad * p).sum().detach() for p in model.parameters()])
    torch.testing.assert_close(event.coefficients, reference, rtol=0, atol=0)
    inputs = torch.stack((torch.zeros(62), torch.linspace(-0.2, 0.2, 62)))
    torch.testing.assert_close(event(inputs), 0.2 * (inputs @ reference), rtol=0, atol=0)
    bad_weights = tmp_path / "resnet18-f37072fd.pth"
    bad_weights.write_bytes(b"wrong")
    with pytest.raises(ValueError, match="size mismatch"):
        G7ResNetEvent(data, bad_weights, "cpu")
