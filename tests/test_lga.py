"""
tests/test_lga.py
=================
Pin the Latent Geometry Alignment (LGA) contract (ledger D3). LGA is the single
proposed metric; these tests assert the invariants the write-up relies on so a
future edit cannot silently change what "LGA" means:

  * LGA(baseline) == 0 and LGA(oracle) == 1 exactly (the anchors);
  * within a dataset LGA is an affine reweighting of (cka_oracle, oracle_gap) with
    a CONSTANT scale, so it preserves rank order (the justification for using the
    baseline->oracle ratio despite D2 rule 2);
  * a degenerate baseline (== oracle) fails loud rather than emitting inf/nan.
"""
import numpy as np
import pytest

from src.evaluation.lga import inject_lga, lga_value


def _rows(cka_base=0.6, gap_base=0.5, method_pts=None):
    """A minimal dataset row list: baseline + oracle + some method points."""
    rows = [
        {"method": "baseline", "cka_oracle": cka_base, "oracle_gap": gap_base},
        {"method": "oracle",   "cka_oracle": 1.0,      "oracle_gap": 0.0},
    ]
    for c, g in (method_pts or [(0.7, 0.3), (0.5, 0.8), (0.9, 0.1)]):
        rows.append({"method": "neg_grad", "cka_oracle": c, "oracle_gap": g})
    return rows


def test_anchors_are_zero_and_one():
    rows = _rows()
    inject_lga(rows)
    base = next(r for r in rows if r["method"] == "baseline")
    orc = next(r for r in rows if r["method"] == "oracle")
    assert base["lga"] == pytest.approx(0.0)
    assert orc["lga"] == pytest.approx(1.0)


def test_matches_closed_form():
    cka_base, gap_base = 0.6, 0.5
    rows = _rows(cka_base, gap_base, [(0.8, 0.2)])
    inject_lga(rows)
    r = rows[-1]
    expect = 0.5 * (0.8 - cka_base) / (1 - cka_base) + 0.5 * (1 - 0.2 / gap_base)
    assert r["lga"] == pytest.approx(expect)
    assert lga_value(0.8, 0.2, cka_base, gap_base) == pytest.approx(expect)


def test_affine_constant_scale_preserves_rank():
    """The core D3 justification: LGA is a fixed linear blend of the two components
    with dataset-CONSTANT weights, so its within-dataset ordering is a deterministic
    function of the components and does not depend on the (blow-up-prone) denominator
    magnitude. Two different anchor scales must give the SAME ranking of the same
    method points."""
    pts = [(0.70, 0.30), (0.50, 0.80), (0.90, 0.10), (0.65, 0.45)]
    r1 = _rows(0.6, 0.5, pts); inject_lga(r1)
    r2 = _rows(0.95, 0.02, pts); inject_lga(r2)   # near-degenerate -> huge magnitudes
    m1 = [r["lga"] for r in r1 if r["method"] == "neg_grad"]
    m2 = [r["lga"] for r in r2 if r["method"] == "neg_grad"]
    assert np.max(np.abs(m2)) > 10           # the CelebA-style blow-up really happens
    assert list(np.argsort(m1)) == list(np.argsort(m2))   # yet ranking is identical


def test_missing_components_give_none():
    rows = _rows()
    rows.append({"method": "rmu", "cka_oracle": None, "oracle_gap": 0.2})
    inject_lga(rows)
    assert rows[-1]["lga"] is None


def test_degenerate_baseline_fails_loud():
    rows = [{"method": "baseline", "cka_oracle": 1.0, "oracle_gap": 0.0},
            {"method": "oracle", "cka_oracle": 1.0, "oracle_gap": 0.0}]
    with pytest.raises(ValueError):
        inject_lga(rows)


def test_no_baseline_fails_loud():
    rows = [{"method": "neg_grad", "cka_oracle": 0.7, "oracle_gap": 0.3}]
    with pytest.raises(ValueError):
        inject_lga(rows)


# --- dedup guard (ledger E22): finetune/neg_grad shared the _ElboAscent impl, so a
# finetune run at lambda>0 was byte-identical to the neg_grad run of the same config
# and appeared as two operating points. This asserts the dense sets never again carry
# two DIFFERENT-method rows for one checkpoint (identical cka_oracle+oracle_gap AND
# attack -- feature_unlearn shares the baseline's ENCODER geometry but not the attack,
# so it is correctly not flagged). Skips datasets whose dense file is absent.
import json as _json
from pathlib import Path as _Path

import pytest as _pytest

_ROOT = _Path(__file__).resolve().parent.parent


@_pytest.mark.parametrize("dataset", ["dsprites", "celeba", "mpi3d"])
def test_no_duplicate_operating_points_across_methods(dataset):
    fp = _ROOT / f"results/{dataset}/dense_recovery.json"
    if not fp.exists():
        _pytest.skip(f"no dense_recovery.json for {dataset}")
    rows = _json.loads(fp.read_text())
    seen = {}
    for r in rows:
        if r.get("cka_oracle") is None or r.get("attack_at_r") is None:
            continue
        key = (round(r["cka_oracle"], 6), round(r["oracle_gap"], 6),
               round(r["attack_at_r"], 6))
        if key in seen and seen[key] != r["method"]:
            raise AssertionError(
                f"{dataset}: identical operating point under two methods "
                f"'{seen[key]}' and '{r['method']}' (run {r['run']}) -- "
                f"finetune/neg_grad-style duplication (ledger E22)")
        seen[key] = r["method"]
