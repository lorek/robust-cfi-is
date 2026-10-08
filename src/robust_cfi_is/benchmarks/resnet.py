"""Explicit, offline implementation of the paper's g7 ResNet benchmark."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Any

import torch

from robust_cfi_is.benchmarks.analytic import RareEventProblem
from robust_cfi_is.distributions import GaussianDistribution


G7_DATA_NAME = "imagenetv2-matched-frequency-format-val_first24.pt"
G7_DATA_SIZE = 7_227_208
G7_DATA_SHA256 = "3eed840225285c10291673dea3a247504d6c225fd617648ba7b139ee5aa63c32"
RESNET18_WEIGHTS_NAME = "resnet18-f37072fd.pth"
RESNET18_WEIGHTS_SIZE = 46_830_571
RESNET18_WEIGHTS_SHA256 = "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec"
G7_DIMENSION = 62
G7_NOISE_SCALE = 0.2
G7_THRESHOLD = 5.157151232471544

RESNET18_PARAMETER_NAMES = (
    "conv1.weight", "bn1.weight", "bn1.bias",
    "layer1.0.conv1.weight", "layer1.0.bn1.weight", "layer1.0.bn1.bias",
    "layer1.0.conv2.weight", "layer1.0.bn2.weight", "layer1.0.bn2.bias",
    "layer1.1.conv1.weight", "layer1.1.bn1.weight", "layer1.1.bn1.bias",
    "layer1.1.conv2.weight", "layer1.1.bn2.weight", "layer1.1.bn2.bias",
    "layer2.0.conv1.weight", "layer2.0.bn1.weight", "layer2.0.bn1.bias",
    "layer2.0.conv2.weight", "layer2.0.bn2.weight", "layer2.0.bn2.bias",
    "layer2.0.downsample.0.weight", "layer2.0.downsample.1.weight",
    "layer2.0.downsample.1.bias", "layer2.1.conv1.weight",
    "layer2.1.bn1.weight", "layer2.1.bn1.bias", "layer2.1.conv2.weight",
    "layer2.1.bn2.weight", "layer2.1.bn2.bias", "layer3.0.conv1.weight",
    "layer3.0.bn1.weight", "layer3.0.bn1.bias", "layer3.0.conv2.weight",
    "layer3.0.bn2.weight", "layer3.0.bn2.bias",
    "layer3.0.downsample.0.weight", "layer3.0.downsample.1.weight",
    "layer3.0.downsample.1.bias", "layer3.1.conv1.weight",
    "layer3.1.bn1.weight", "layer3.1.bn1.bias", "layer3.1.conv2.weight",
    "layer3.1.bn2.weight", "layer3.1.bn2.bias", "layer4.0.conv1.weight",
    "layer4.0.bn1.weight", "layer4.0.bn1.bias", "layer4.0.conv2.weight",
    "layer4.0.bn2.weight", "layer4.0.bn2.bias",
    "layer4.0.downsample.0.weight", "layer4.0.downsample.1.weight",
    "layer4.0.downsample.1.bias", "layer4.1.conv1.weight",
    "layer4.1.bn1.weight", "layer4.1.bn1.bias", "layer4.1.conv2.weight",
    "layer4.1.bn2.weight", "layer4.1.bn2.bias", "fc.weight", "fc.bias",
)
_EXPECTED_Y = (0.0,) * 10 + (1.0,) * 10 + (2.0,) * 4


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def verify_file(path: str | Path, *, name: str, size: int, sha256: str) -> Path:
    candidate = Path(path)
    if not candidate.is_file():
        raise ValueError(f"{name} is not a readable local file: {candidate}")
    if candidate.stat().st_size != size:
        raise ValueError(f"{name} size mismatch")
    if _sha256(candidate) != sha256:
        raise ValueError(f"{name} SHA-256 mismatch")
    return candidate


def validate_g7_payload(payload: Any) -> tuple[torch.Tensor, torch.Tensor]:
    if not isinstance(payload, dict) or set(payload) != {"X", "Y"}:
        raise ValueError("g7 tensor must be a mapping with exactly X and Y")
    x, y = payload["X"], payload["Y"]
    if not isinstance(x, torch.Tensor) or not isinstance(y, torch.Tensor):
        raise ValueError("g7 X and Y must be tensors")
    if x.shape != (24, 3, 224, 224) or x.dtype != torch.float16:
        raise ValueError("g7 X identity mismatch")
    if y.shape != (24,) or y.dtype != torch.float16:
        raise ValueError("g7 Y identity mismatch")
    if not x.is_contiguous() or not y.is_contiguous():
        raise ValueError("g7 tensors must be contiguous")
    if not bool(torch.isfinite(x).all()) or not bool(torch.isfinite(y).all()):
        raise ValueError("g7 tensors contain non-finite values")
    if tuple(y.tolist()) != _EXPECTED_Y:
        raise ValueError("g7 Y representation mismatch")
    return x, y


def load_g7_tensor(path: str | Path) -> tuple[torch.Tensor, torch.Tensor]:
    verified = verify_file(
        path, name="g7 tensor", size=G7_DATA_SIZE, sha256=G7_DATA_SHA256
    )
    payload = torch.load(verified, map_location="cpu", weights_only=True)
    return validate_g7_payload(payload)


def explicit_device(value: str) -> torch.device:
    if value == "cpu":
        return torch.device("cpu")
    if not value.startswith("cuda:") or not value[5:].isdigit():
        raise ValueError("event device must be exactly 'cpu' or 'cuda:N'")
    index = int(value[5:])
    if not torch.cuda.is_available() or index >= torch.cuda.device_count():
        raise ValueError(f"requested event device is unavailable: {value}")
    return torch.device(value)


@dataclass(frozen=True)
class G7AssetRecord:
    data_sha256: str
    weights_sha256: str
    event_device: str
    torch_version: str
    torchvision_version: str
    parameter_order_sha256: str

    def to_dict(self) -> dict[str, str]:
        return self.__dict__.copy()


class G7ResNetEvent:
    """Linearized loss degradation, exactly following the authoritative source."""

    def __init__(self, data_path: str | Path, weights_path: str | Path, device: str):
        event_device = explicit_device(device)
        x, y = load_g7_tensor(data_path)
        weights = verify_file(
            weights_path,
            name="ResNet18 weights",
            size=RESNET18_WEIGHTS_SIZE,
            sha256=RESNET18_WEIGHTS_SHA256,
        )
        try:
            import torchvision
            from torchvision.models import resnet18
        except ImportError as exc:
            raise RuntimeError("g7 requires the optional 'g7' dependency extra") from exc

        model = resnet18(weights=None)
        state = torch.load(weights, map_location="cpu", weights_only=True)
        model.load_state_dict(state, strict=True)
        names = tuple(name for name, _ in model.named_parameters())
        if names != RESNET18_PARAMETER_NAMES or len(names) != G7_DIMENSION:
            raise RuntimeError("ResNet18 parameter-tensor ordering mismatch")
        parameters = list(model.parameters())
        if any(p is not named for p, (_, named) in zip(parameters, model.named_parameters())):
            raise RuntimeError("ResNet18 parameters() and named_parameters() order differ")

        self.device = event_device
        self.model = model.to(event_device).eval()
        self.x = x.to(event_device).float()
        self.y = y.to(event_device).long()
        criterion = torch.nn.CrossEntropyLoss(reduction="mean")
        for parameter in parameters:
            parameter.requires_grad_(True)
        parameters = list(self.model.parameters())
        loss = criterion(self.model(self.x), self.y)
        loss.backward()
        coefficients = []
        for parameter in parameters:
            if parameter.grad is None:
                raise RuntimeError("ResNet18 parameter gradient is missing")
            coefficients.append((parameter.grad * parameter).sum().detach())
        self.model.zero_grad(set_to_none=True)
        for parameter in parameters:
            parameter.requires_grad_(False)
        self.coefficients = torch.stack(coefficients).to(event_device)
        order_digest = hashlib.sha256("\n".join(names).encode()).hexdigest()
        self.asset_record = G7AssetRecord(
            G7_DATA_SHA256,
            RESNET18_WEIGHTS_SHA256,
            str(event_device),
            torch.__version__,
            torchvision.__version__,
            order_digest,
        )

    @torch.inference_mode()
    def __call__(self, noise_vectors: torch.Tensor) -> torch.Tensor:
        original_device = noise_vectors.device
        values = torch.as_tensor(noise_vectors)
        if values.ndim == 1:
            values = values.unsqueeze(0)
        if values.ndim != 2 or values.shape[1] != G7_DIMENSION:
            raise ValueError("g7 expects noise vectors with shape (n, 62)")
        result = G7_NOISE_SCALE * (
            values.to(self.device, dtype=torch.float32) @ self.coefficients
        )
        return result.to(original_device)


def get_g7_benchmark(
    data_path: str | Path, weights_path: str | Path, *, device: str
) -> tuple[RareEventProblem, G7AssetRecord]:
    event = G7ResNetEvent(data_path, weights_path, device)
    problem = RareEventProblem(
        name="g7_resnet",
        dimension=G7_DIMENSION,
        gamma=G7_THRESHOLD,
        nominal=GaussianDistribution(
            mean=torch.zeros(G7_DIMENSION), covariance=torch.eye(G7_DIMENSION)
        ),
        event_score=event,
        reference_probability=6e-5,
    )
    return problem, event.asset_record
