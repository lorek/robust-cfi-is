import torch

from robust_cfi_is.methods.forward_kl import mean_one_adaptive_ce_weights
from robust_cfi_is.methods.reverse_kl import reverse_kl_surrogate
from robust_cfi_is.naming import get_method


def test_ffkl_c_retains_corrected_log_space_weights_under_extreme_ratios():
    method = get_method("ffkl_c")
    assert method.objective == "forward_kl"
    log_target = torch.tensor([-2000.0, -1999.0, -1998.0])
    log_proposal = torch.tensor([0.0, 0.0, 0.0])
    assert torch.count_nonzero(torch.exp(log_target - log_proposal)) == 0
    weights = mean_one_adaptive_ce_weights(
        log_target,
        log_proposal,
        output_dtype=torch.float32,
    )
    assert torch.isfinite(weights).all()
    assert weights.sum() > 0
    assert torch.allclose(weights.mean(), torch.tensor(1.0))


def test_frkl_c_remains_reverse_kl_and_does_not_use_forward_weights():
    method = get_method("frkl_c")
    assert method.objective == "reverse_kl"
    log_q = torch.tensor([-2.0, -1.0, -3.0])
    log_f = torch.tensor([-1.0, -2.0, -2.5])
    scores = torch.tensor([-0.2, 0.4, 0.8])
    _, eta = reverse_kl_surrogate(
        log_q,
        log_f,
        scores,
        gamma=0.0,
        temperature=0.25,
        baseline=0.0,
    )
    forward = mean_one_adaptive_ce_weights(
        log_f,
        log_q,
        output_dtype=torch.float32,
    )
    assert not torch.allclose(1.0 + eta, forward)
