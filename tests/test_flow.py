"""Correctness tests for the RealNVP flow (Stage-2 second model)."""
import math
import torch

from src.models.flow import RealNVP, FlowConfig


def _model(c=1, hw=8, n=6, seed=0):
    torch.manual_seed(seed)
    m = RealNVP(FlowConfig(in_channels=c, image_size=hw, n_coupling=n, hidden_channels=16))
    # randomise away from the identity init so tests exercise a non-trivial map
    with torch.no_grad():
        for cpl in m.couplings:
            cpl.net.c3.weight.normal_(0, 0.1); cpl.net.c3.bias.normal_(0, 0.1)
            cpl.net.scale.normal_(0, 0.3)
    return m.eval()


def test_invertibility():
    m = _model(c=1, hw=8)
    x = torch.rand(4, 1, 8, 8)
    z, _ = m.encode(x)
    x2 = m.decode(z)
    assert torch.allclose(x, x2, atol=1e-4), f"max err {(x-x2).abs().max():.2e}"


def test_invertibility_rgb():
    m = _model(c=3, hw=8)           # exercises channel masks too
    x = torch.rand(3, 3, 8, 8)
    z, _ = m.encode(x)
    assert torch.allclose(x, m.decode(z), atol=1e-4)


def test_change_of_variables_logdet():
    """Model's logdet must equal the log|det| of the encode Jacobian (autograd)."""
    m = _model(c=1, hw=4, n=5)
    x = torch.rand(1, 1, 4, 4, dtype=torch.float64)
    m = m.double()
    def enc_flat(v):
        z, _ = m.encode(v.reshape(1, 1, 4, 4))
        return z.reshape(-1)
    J = torch.autograd.functional.jacobian(enc_flat, x.reshape(-1))
    autograd_logdet = torch.slogdet(J).logabsdet
    _, model_logdet = m._forward_logdet(x)
    assert abs(autograd_logdet.item() - model_logdet.item()) < 1e-6, \
        f"autograd {autograd_logdet.item():.6f} vs model {model_logdet.item():.6f}"


def test_log_prob_matches_base_plus_logdet():
    m = _model(c=1, hw=8)
    x = torch.rand(5, 1, 8, 8)
    z, logdet = m._forward_logdet(x)
    log_pz = (-0.5 * (z ** 2 + math.log(2 * math.pi))).flatten(1).sum(1)
    assert torch.allclose(m.log_prob(x), log_pz + logdet, atol=1e-5)
    assert torch.isfinite(m.log_prob(x)).all()


def test_nll_trains_one_step():
    torch.manual_seed(1)
    m = RealNVP(FlowConfig(in_channels=1, image_size=8, n_coupling=6, hidden_channels=16))
    x = torch.rand(16, 1, 8, 8)
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    l0 = m(x).loss.item()
    for _ in range(50):
        opt.zero_grad(); out = m(x); out.loss.backward(); opt.step()
    l1 = m(x).loss.item()
    assert math.isfinite(l0) and math.isfinite(l1)
    assert l1 < l0, f"NLL should drop on an overfit batch: {l0:.1f} -> {l1:.1f}"


def test_identity_at_init():
    """Zero-init couplings => the coupling stack is the identity at init: z equals the
    preprocessed input and the couplings add zero logdet (so training starts stable)."""
    torch.manual_seed(2)
    m = RealNVP(FlowConfig(in_channels=1, image_size=8, n_coupling=8, hidden_channels=16)).eval()
    x = torch.rand(4, 1, 8, 8)
    y, pre_ld = m._preprocess(x, stochastic=False)
    z, logdet = m._forward_logdet(x)
    assert torch.allclose(z, y, atol=1e-6), "couplings should be identity at init"
    assert torch.allclose(logdet, pre_ld, atol=1e-6), "couplings should add zero logdet at init"


def test_nll_finite_on_binary():
    """Regression: a continuous flow on binary {0,1} pixels diverged (NLL -> -inf -> NaN)
    before dequantization was added. With levels=2 the NLL must stay finite."""
    torch.manual_seed(3)
    m = RealNVP(FlowConfig(in_channels=1, image_size=8, n_coupling=6, hidden_channels=16, levels=2))
    x = (torch.rand(32, 1, 8, 8) > 0.7).float()   # sparse binary, like dSprites
    opt = torch.optim.Adam(m.parameters(), lr=1e-3)
    for _ in range(30):
        opt.zero_grad(); out = m(x); out.loss.backward(); opt.step()
        assert torch.isfinite(out.loss), "NLL went non-finite on binary data"
    assert torch.isfinite(m.log_prob(x)).all()
