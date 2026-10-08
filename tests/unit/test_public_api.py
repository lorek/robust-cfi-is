import inspect

import robust_cfi_is


def test_top_level_public_api_exposes_fitting_surface():
    expected = (
        "fit_cfi",
        "fit_forward_kl",
        "fit_fixed_proposal_flow",
        "fit_public_method",
        "fit_reverse_kl",
        "refine_forward_kl",
        "refine_reverse_kl",
    )
    for name in expected:
        assert callable(getattr(robust_cfi_is, name))


def test_public_flow_defaults_match_authoritative_paper_budgets():
    parameters = inspect.signature(robust_cfi_is.fit_public_method).parameters
    assert parameters["flow_initial_samples"].default == 25_000
    assert parameters["flow_training_samples"].default == 25_000
    assert parameters["flow_epochs"].default == 1_000
    assert parameters["flow_learning_rate"].default == 1e-5
