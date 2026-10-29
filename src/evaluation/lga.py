"""
src/evaluation/lga.py
=====================
Latent Geometry Alignment (LGA) -- the project's single proposed evaluation
metric, defined in ONE place so no script can drift from another (the divergence
that produced the geo_suite triplication, and historically let the saturated-judge
bug hide behind three copies of the same code).

LGA answers one question: *how far has an unlearned model's latent geometry
moved from the untouched baseline toward the retrained oracle?* It is a
continuous, model-INDEPENDENT score anchored to two fixed references per dataset
-- the baseline (no unlearning) and the oracle (retrained without the concept):

    LGA = 0.5 * (CKA_model - CKA_base) / (1 - CKA_base)          # global axis
        + 0.5 * (1 - gap_model / gap_base)                       # concept axis

where
    CKA_x  = linear CKA between model x's latents and the oracle's (cka_oracle),
             so the oracle's own value is 1 by construction;
    gap_x  = |Mahalanobis_x - Mahalanobis_oracle|  (oracle_gap), the absolute
             concept-separation distance to the oracle, so the oracle's is 0.

Each term is the FRACTION of the baseline->oracle change achieved on that axis:
    0  = geometry unchanged from the baseline,
    1  = geometry reached the oracle.
By construction LGA(baseline)=0 and LGA(oracle)=1 exactly.

It is deliberately NOT clamped. >1 means the geometry overshot past the oracle;
<0 means it moved away from the oracle / blew up. Large magnitudes are honest
signals, not errors.

This is "Option B" (baseline->oracle progress), chosen 2026-09 over the earlier
rank-normalised suite because it measures geometry change on a continuous,
model-independent scale rather than ranking one unlearning method against the
others (see the thesis, Chapter 4).

Cross-dataset note (a finding, not a bug): when baseline ~= oracle -- e.g. CelebA,
where only ~6.5% of data carries the concept so retraining barely moves the
geometry -- both denominators (1 - CKA_base, gap_base) are near zero and LGA
takes large magnitudes. That correctly reports "there was almost no baseline->
oracle gap to move along". LGA magnitudes are therefore NOT comparable across
datasets; the denominators are per-dataset CONSTANTS, so within a dataset LGA is
an affine reweighting of (cka_oracle, oracle_gap) that preserves within-dataset
ranking and correlation. Report per dataset, and rank-normalise WITHIN a dataset
before pooling.
"""
from __future__ import annotations


def _anchor(rows):
    """(cka_base, gap_base) from the dataset's baseline row -- LGA's references.
    The oracle row supplies the far anchors implicitly (CKA=1, gap=0)."""
    base = next((r for r in rows if r.get("method") == "baseline"), None)
    if base is None:
        raise ValueError("LGA needs the 'baseline' operating point to anchor; "
                         "pass the full row list (refs included) for this dataset.")
    cka_base = base.get("cka_oracle")
    gap_base = base.get("oracle_gap")
    if cka_base is None or gap_base is None:
        raise ValueError("baseline row is missing cka_oracle / oracle_gap")
    if cka_base >= 1.0 or gap_base == 0.0:
        # A degenerate baseline (already identical to the oracle) has no gap to
        # measure progress along -- fail loud rather than emit inf/nan silently.
        raise ValueError(f"degenerate LGA anchor (cka_base={cka_base}, "
                         f"gap_base={gap_base}); baseline == oracle?")
    return float(cka_base), float(gap_base)


def lga_value(cka_oracle, oracle_gap, cka_base, gap_base):
    """Scalar LGA for one model given the dataset anchors. Unclamped."""
    cka_term = (cka_oracle - cka_base) / (1.0 - cka_base)
    gap_term = 1.0 - oracle_gap / gap_base
    return 0.5 * cka_term + 0.5 * gap_term


def inject_lga(rows):
    """Inject row['lga'] for every row in ONE dataset's operating-point list
    (baseline + oracle + method rows). Rows missing cka_oracle/oracle_gap get
    lga=None. Returns (cka_base, gap_base). Mutates in place.

    Pass the FULL list including baseline and oracle rows -- the anchors come
    from them. Drop refs AFTER calling this if the analysis excludes them.
    """
    cka_base, gap_base = _anchor(rows)
    for r in rows:
        c, g = r.get("cka_oracle"), r.get("oracle_gap")
        r["lga"] = None if (c is None or g is None) else lga_value(c, g, cka_base, gap_base)
    return cka_base, gap_base
