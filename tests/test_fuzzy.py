from itertools import product

import pytest

from dms.config import Config
from dms.risk.driver_risk import DriverRiskModel, tier_for
from dms.risk.fuzzy import FuzzySet, FuzzyVariable, MamdaniSystem, Rule
from dms.risk.rules import FATIGUE_LEVELS, LEVELS, all_rule_blocks, risk_variable, unit_variable
from dms.types import ContextReading, Severity, SeverityReport


def test_fuzzification_reproduces_table_1_8():
    v = unit_variable("x")
    assert v.fuzzify(0.62) == pytest.approx({"Low": 0.0, "Moderate": 0.40, "High": 0.10})
    assert v.fuzzify(0.55) == pytest.approx({"Low": 0.0, "Moderate": 0.75, "High": 0.0})
    assert v.fuzzify(0.68) == pytest.approx({"Low": 0.0, "Moderate": 0.10, "High": 0.40})


def test_firing_strengths_reproduce_table_1_9():
    f, d, c = (unit_variable(n).fuzzify(x) for n, x in (("F", 0.62), ("D", 0.55), ("C", 0.68)))
    assert min(f["Moderate"], d["Moderate"], c["Moderate"]) == pytest.approx(0.10)
    assert min(f["Moderate"], d["Moderate"], c["High"]) == pytest.approx(0.40)
    assert min(f["High"], d["Moderate"], c["Moderate"]) == pytest.approx(0.10)
    assert min(f["High"], d["Moderate"], c["High"]) == pytest.approx(0.10)


def test_triangle_matches_paper_formula():
    a, b, c = 0.3, 0.5, 0.7
    s = FuzzySet("t", [(a, 0), (b, 1), (c, 0)])
    for x in (0.1, 0.35, 0.5, 0.64, 0.9):
        assert float(s.mu(x)) == pytest.approx(max(min((x - a) / (b - a), (c - x) / (c - b)), 0))


@pytest.mark.parametrize("name,n_inputs", [("fatigue", 3), ("distraction", 2), ("context", 2)])
def test_rule_bases_are_complete_and_unique(name, n_inputs):
    rules = all_rule_blocks()[name]
    assert len(rules) == 3**n_inputs
    combos = {tuple(r.antecedent.values()) for r in rules}
    assert combos == set(product(LEVELS, repeat=n_inputs))


def test_overall_rule_base_has_four_fatigue_levels():
    rules = all_rule_blocks()["overall"]
    assert len(rules) == 4 * 3 * 3
    combos = {tuple(r.antecedent.values()) for r in rules}
    assert combos == set(product(FATIGUE_LEVELS, LEVELS, LEVELS))


def test_centroid_of_single_fully_fired_rule():
    out = FuzzyVariable("y", 0, 1, {"Moderate": FuzzySet("Moderate", [(0.3, 0), (0.5, 1), (0.7, 0)])})
    x = FuzzyVariable("x", 0, 1, {"On": FuzzySet("On", [(0, 1), (1, 1)])})
    sys = MamdaniSystem("t", [x], out, [Rule({"x": "On"}, "Moderate")], resolution=1001)
    assert sys.infer({"x": 0.5}).value == pytest.approx(0.5, abs=1e-3)


def test_risk_output_sets_centroids_fall_in_their_tier():
    bounds = Config().risk.tier_bounds
    var = risk_variable()
    for name in ("NONE", "LOW", "MEDIUM", "HIGH"):
        sys = MamdaniSystem("t", [FuzzyVariable("x", 0, 1, {"On": FuzzySet("On", [(0, 1), (1, 1)])})], var, [Rule({"x": "On"}, name)])
        assert tier_for(sys.infer({"x": 0.5}).value, bounds) == name


def _assess(d=0.0, y=0.0, g=0.0, p=0.0, speed=40.0, motion=0.0, hours=0.0):
    sev = SeverityReport(Severity(d, ""), Severity(y, ""), Severity(g, ""), Severity(p, ""))
    return DriverRiskModel(Config()).assess(sev, ContextReading(speed, motion, hours * 3600))


def test_behaviour_to_tier_mapping_of_section_1_7_1():
    assert _assess().tier == "NONE"
    assert _assess(y=0.4).tier == "LOW"  # single yawn
    assert _assess(y=0.7).tier == "MEDIUM"  # frequent yawning
    assert _assess(d=0.7).tier == "MEDIUM"  # drowsiness
    assert _assess(g=0.7).tier == "HIGH"  # distraction (gaze)
    assert _assess(p=0.7).tier == "HIGH"  # distraction (phone)
    assert _assess(motion=0.5).tier == "HIGH"  # sudden change of motion


def test_context_modulates_risk():
    assert _assess(p=0.7, speed=0.0).tier == "MEDIUM"  # stopped: one tier lower
    assert _assess(p=0.7, speed=None).tier == "HIGH"  # GPS missing: assume moving
    assert _assess(y=0.4, hours=2.5).tier == "MEDIUM"  # long drive escalates moderate fatigue
    assert _assess(hours=3.0).tier == "NONE"  # duration alone never alarms
    assert _assess(speed=110.0).tier == "NONE"  # an alert driver at speed is not an alarm


def test_drowsy_and_yawning_together_raise_the_risk():
    single_yawn, drowsy = 0.4, 0.7
    assert _assess(y=single_yawn).tier == "LOW"
    assert _assess(d=drowsy).tier == "MEDIUM"
    combined = _assess(d=drowsy, y=single_yawn)
    assert combined.tier == "HIGH"
    assert combined.fatigue_level == "Severe"
    # moderate drowsiness + one yawn: each alone is LOW, together MEDIUM
    assert _assess(d=0.45).tier == "LOW"
    assert _assess(d=0.45, y=single_yawn).tier == "MEDIUM"


def test_fatigue_levels_reported():
    assert _assess().fatigue_level == "Low"
    assert _assess(y=0.4).fatigue_level == "Moderate"
    assert _assess(d=0.7).fatigue_level == "High"
    assert _assess(d=0.7, hours=2.5).fatigue_level == "Severe"  # long drive escalates


def test_risk_tier_hysteresis():
    from dms.risk.driver_risk import TierTracker

    tr = TierTracker([25.0, 52.5, 77.5], margin=5.0)
    assert [tr.update(r) for r in (30, 24, 21, 19, 53, 50, 48, 46, 80, 10)] == \
        ["LOW", "LOW", "LOW", "NONE", "MEDIUM", "MEDIUM", "MEDIUM", "LOW", "HIGH", "NONE"]
    assert _assess(d=0.7, hours=2.5).tier == "HIGH"
