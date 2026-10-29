"""Properties the U-LiRA scorer must have (ledger D4/E29)."""
import numpy as np
from scipy.stats import norm

from src.evaluation import ulira as U


def _mats(rng, n=4, N=2000, mu_in=0.0, mu_out=0.0, s_in=1.0, s_out=1.0):
    base = rng.normal(0, 3, N)
    return (base + mu_in + rng.normal(0, s_in, (n, N)),
            base + mu_out + rng.normal(0, s_out, (n, N)))


def test_identical_distributions_give_no_advantage():
    rng = np.random.default_rng(0)
    a, b = _mats(rng)
    d = U.summarize(*U.loo_llrs(a, b))
    assert abs(d["adv"]) < 0.08, d
    assert 0.35 < d["post_in"] < 0.65 and 0.35 < d["post_out"] < 0.65, d
    assert d["auc"] < 0.56, d


def test_separable_distributions_give_full_advantage():
    rng = np.random.default_rng(1)
    a, b = _mats(rng, mu_out=40.0)
    d = U.summarize(*U.loo_llrs(a, b))
    assert d["adv"] > 0.99 and d["auc"] > 0.999, d


def test_advantage_is_symmetric_in_which_side_is_which():
    rng = np.random.default_rng(2)
    a, b = _mats(rng, mu_out=3.0)
    d1 = U.summarize(*U.loo_llrs(a, b))
    d2 = U.summarize(*U.loo_llrs(b, a))
    assert abs(d1["adv"] - d2["adv"]) < 0.05


def test_leave_one_out_removes_self_inclusion():
    """With few shadows a model scored against a fit that CONTAINS it looks more IN-like."""
    rng = np.random.default_rng(3)
    a, b = _mats(rng, n=3, N=3000, mu_out=1.0, s_in=2.0, s_out=2.0)
    loo_in, _ = U.loo_llrs(a, b)
    incl = np.stack([U.llr(a[i], U.fit(a), U.fit(b)) for i in range(3)])
    assert incl.mean() > loo_in.mean() + 0.02


def test_unbiased_sigma_matches_known_value():
    m = np.array([[0.0, 0.0], [2.0, 2.0]])           # per-example var (ddof=1) = 2
    assert abs(U.fit(m, unbiased=True)[1] - np.sqrt(2.0)) < 1e-9
    assert abs(U.fit(m, unbiased=False)[1] - 1.0) < 1e-9


def test_log_scale_is_finite_and_monotone():
    x = np.array([-5.0, 0.0, 1e-9, 1.0, 10.0, 1e4])
    y = U.log_scale(x)
    assert np.isfinite(y).all() and (np.diff(y) >= 0).all()


def test_target_posterior_reproduces_legacy_v3_formula():
    """raw scale + biased sigma + fixed fits must equal the original run_ulira.py math."""
    rng = np.random.default_rng(4)
    a, b = _mats(rng, mu_out=5.0, s_out=4.0)
    x = a[0] + 1.0
    mu_i, s_i = a.mean(0), float(np.sqrt(((a - a.mean(0)) ** 2).mean()))
    mu_o, s_o = b.mean(0), float(np.sqrt(((b - b.mean(0)) ** 2).mean()))
    lam = norm.logpdf(x, mu_i, s_i) - norm.logpdf(x, mu_o, s_o)
    legacy = float(np.mean(1 / (1 + np.exp(-lam))))
    assert abs(U.target_posterior(x, a, b, unbiased=False) - legacy) < 1e-12


def test_old_failure_mode_is_a_calibration_property_not_a_mean_shift():
    """E29 mechanism: tight IN + wide OUT puts the Lambda=0 boundary a few sigma_in from IN.
    A target only ~2.5% of the way to OUT gets posterior ~0 against UNTOUCHED-IN fits;
    scored against an IN set that contains the same shift (the Hayes definition), ~1."""
    rng = np.random.default_rng(5)
    N = 3000
    base = rng.normal(0, 3, N)
    untouched = base + rng.normal(0, 1.0, (4, N))
    oracle = base + 400 + rng.normal(0, 240, (4, N))
    unlearned = base + 10.0 + rng.normal(0, 1.0, (4, N))   # +10 nats of a 400-nat gap
    tgt = unlearned[0]
    assert U.target_posterior(tgt, untouched, oracle, unbiased=False) < 0.01
    # not 1.0: with only 3 shadows the target deviates from their mean with variance
    # (1 + 1/3) sigma^2, so ~1% of examples fall far enough out to lose to the wide OUT fit
    assert U.target_posterior(tgt, unlearned[1:], oracle, unbiased=False) > 0.95
