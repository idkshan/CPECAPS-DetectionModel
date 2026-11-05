"""Stage 5: a small Mamdani fuzzy-inference engine (Section 1.7.18).

    1. Fuzzification  - membership degree of each input in each fuzzy set
    2. Firing strength - AND of a rule's antecedents = min (Zadeh)
    3. Implication    - each rule's output set is clipped at its strength (min)
    4. Aggregation    - OR of all clipped sets = max (Zadeh)
    5. Defuzzification - centroid  R* = sum(y * mu(y)) / sum(mu(y))

Membership functions are piecewise linear, given as ``[(x, mu), ...]``;
outside the listed range the end values extend flat, which yields the
shoulder shapes of Table 1.7 (e.g. "High" stays at 1 above 0.8). A triangle
(a, b, c) is ``[(a, 0), (b, 1), (c, 0)]`` and equals the paper's
mu(x) = max(min((x - a)/(b - a), (c - x)/(c - b)), 0).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class FuzzySet:
    name: str
    points: Sequence[Tuple[float, float]]

    def mu(self, x):
        xs = [p[0] for p in self.points]
        ys = [p[1] for p in self.points]
        return np.interp(x, xs, ys)


@dataclass
class FuzzyVariable:
    name: str
    lo: float
    hi: float
    sets: Dict[str, FuzzySet]

    def fuzzify(self, x: float) -> Dict[str, float]:
        x = float(min(self.hi, max(self.lo, x)))
        return {n: float(s.mu(x)) for n, s in self.sets.items()}


@dataclass
class Rule:
    antecedent: Dict[str, str]  # variable name -> set name
    consequent: str  # output set name

    def text(self, output_name: str) -> str:
        cond = " AND ".join(f"{v} is {s}" for v, s in self.antecedent.items())
        return f"IF {cond} THEN {output_name} is {self.consequent}"


@dataclass
class InferenceResult:
    value: float
    memberships: Dict[str, Dict[str, float]]
    firing: List[Tuple[Rule, float]]


class MamdaniSystem:
    def __init__(self, name: str, inputs: List[FuzzyVariable], output: FuzzyVariable, rules: List[Rule], resolution: int = 201):
        self.name = name
        self.inputs = {v.name: v for v in inputs}
        self.output = output
        self.rules = rules
        self.universe = np.linspace(output.lo, output.hi, resolution)
        self._out_mu = {n: s.mu(self.universe) for n, s in output.sets.items()}
        for r in rules:
            unknown = [v for v in r.antecedent if v not in self.inputs]
            if unknown or r.consequent not in output.sets:
                raise ValueError(f"Rule references unknown variable/set: {r}")

    def infer(self, values: Dict[str, float]) -> InferenceResult:
        memberships = {n: v.fuzzify(values[n]) for n, v in self.inputs.items()}
        aggregated = np.zeros_like(self.universe)
        firing = []
        for rule in self.rules:
            alpha = min(memberships[v][s] for v, s in rule.antecedent.items())
            firing.append((rule, alpha))
            if alpha > 0:
                aggregated = np.maximum(aggregated, np.minimum(alpha, self._out_mu[rule.consequent]))
        area = aggregated.sum()
        if area <= 1e-12:  # no rule fired (cannot happen with a complete rule base)
            value = self.output.lo
        else:
            value = float((self.universe * aggregated).sum() / area)
        return InferenceResult(value=value, memberships=memberships, firing=firing)
