import json
from pathlib import Path

from robust_cfi_is.cli import main
from robust_cfi_is.artifacts import ArtifactManifest
from robust_cfi_is.config import PaperConfig, load_config
from robust_cfi_is.naming import get_method


ROOT = Path(__file__).parents[2]


def test_all_public_paper_configs_validate(capsys):
    configs = sorted((ROOT / "configs/paper").glob("*.yaml"))
    assert [path.name for path in configs] == [
        "defensive_g1.yaml", "g1.yaml", "g10_20k.yaml", "g10_40k.yaml",
        "g10_500k.yaml", "g2.yaml", "g3.yaml", "g5.yaml", "g6.yaml",
        "g7.yaml",
    ]
    for path in configs:
        config = load_config(path)
        assert isinstance(config, PaperConfig)
        assert config.execution["network"] is False
        assert config.execution["tracking"] is False
        assert all(get_method(method).slug == method for method in config.methods)
        assert main(["validate", str(path)]) == 0
    capsys.readouterr()


def test_reproduction_plan_is_non_executing(capsys):
    config = ROOT / "configs/paper/g6.yaml"
    assert main(["reproduce", "plan", str(config), "--method", "ffkl_c", "--k", "16"]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["experiment_started"] is False
    assert plan["benchmark"] == "g6_papaioannou_100d_Sigma1"
    assert plan["evaluation"]["sample_size"] == 100000


def test_supported_constrained_config_cells_have_manifest_identities():
    manifest = ArtifactManifest.load(ROOT / "reproducibility/artifacts.json")
    identities = {
        (record.benchmark, record.method, record.components)
        for record in manifest.records
        if record.workflow_status in {"supported_analytic", "deferred_g7"}
    }
    constrained = {"frkl_c", "ffkl_c", "cfi_c", "cfi_c_frkl_c", "cfi_c_ffkl_c"}
    for path in (ROOT / "configs/paper").glob("*.yaml"):
        config = load_config(path)
        if config.methods == ("frkl_d",):
            continue
        for method in constrained:
            for components in config.components:
                assert (config.benchmark, method, components) in identities


def test_g7_plan_is_asset_free(capsys):
    config = ROOT / "configs/paper/g7.yaml"
    assert main([
        "reproduce", "plan", str(config), "--method", "cfi_c", "--k", "6"
    ]) == 0
    plan = json.loads(capsys.readouterr().out)
    assert plan["experiment_started"] is False
    assert plan["benchmark"] == "g7_resnet"
    assert plan["execution"]["event_device"] == "explicit_cli_required"
