from pathlib import Path

import pytest

from robust_cfi_is.config import load_config
from robust_cfi_is.naming import (
    get_method,
    supported_methods,
)


ROOT = Path(__file__).resolve().parents[2]


EXPECTED_METHODS = {
    "cfi_c": ("CFI-c", "coverage_first_constrained", True),
    "cfi_c_ffkl_c": ("CFI-c+FKL-c", "cfi_then_forward_kl", True),
    "cfi_c_frkl_c": ("CFI-c+RKL-c", "cfi_then_reverse_kl", True),
    "cfi_c_frkl_c_flow": ("CFI-c+RKL-c+F", "cfi_then_reverse_kl_then_flow", False),
    "cfi_c_ffkl_c_flow": ("CFI-c+FKL-c+F", "cfi_then_forward_kl_then_flow", False),
    "cfi_u": ("CFI-u", "coverage_first", False),
    "ffkl_c": ("fFKL-c", "forward_kl", True),
    "ffkl_u": ("fFKL-u", "forward_kl", False),
    "frkl_c": ("fRKL-c", "reverse_kl", True),
    "frkl_d": ("fRKL-d", "reverse_kl", False),
    "frkl_u": ("fRKL-u", "reverse_kl", False),
}

def test_public_naming_registry_has_exact_mathematical_identities():
    methods = {item.slug: item for item in supported_methods()}
    assert set(methods) == set(EXPECTED_METHODS)
    for slug, (display, objective, constrained) in EXPECTED_METHODS.items():
        method = get_method(slug)
        assert method.display_name == display
        assert method.objective == objective
        assert method.constrained is constrained


def test_example_config_loads_and_is_cpu_offline():
    config = load_config(ROOT / "configs" / "examples" / "g1_cfi_c.yaml")
    assert config.method == "cfi_c"
    assert config.components == 6
    assert config.device == "cpu"
    assert config.tracking is False
    assert config.network is False


def test_config_rejects_unknown_fields(tmp_path):
    source = (ROOT / "configs" / "examples" / "g1_cfi_c.yaml").read_text()
    path = tmp_path / "bad.yaml"
    path.write_text(source + "unexpected: true\n")
    with pytest.raises(ValueError, match="unexpected"):
        load_config(path)


def test_config_rejects_unknown_method_as_normal_user_input(tmp_path):
    source = (ROOT / "configs" / "examples" / "g1_cfi_c.yaml").read_text()
    path = tmp_path / "unknown.yaml"
    path.write_text(source.replace("method: cfi_c", "method: unsupported_method"))
    with pytest.raises(ValueError, match="Unknown public method"):
        load_config(path)


@pytest.mark.parametrize("method", sorted(set(EXPECTED_METHODS) - {"frkl_d"}))
def test_config_accepts_every_public_method_slug(tmp_path, method):
    source = (ROOT / "configs" / "examples" / "g1_cfi_c.yaml").read_text()
    path = tmp_path / f"{method}.yaml"
    path.write_text(source.replace("method: cfi_c", f"method: {method}"))
    assert load_config(path).method == method


def test_defensive_method_uses_explicit_paper_config():
    config = load_config(ROOT / "configs/paper/defensive_g1.yaml")
    assert config.methods == ("frkl_d",)
    assert config.fitting["frkl_d"]["defensive_weights"] == [0.1, 0.2, 0.3, 0.4, 0.5]


@pytest.mark.parametrize("slug", ["cfi_c_frkl_c_flow", "cfi_c_ffkl_c_flow"])
def test_flow_metadata_never_extends_gmm_certification(slug):
    method = get_method(slug)
    assert method.constrained_gmm_backbone is True
    assert method.final_proposal_family == "flow"
    assert method.theorem_certified_final_proposal is False
