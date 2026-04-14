"""Registry of recovery benchmark definitions with explicit metadata."""

from __future__ import annotations

from typing import Callable

import numpy as np

from etree.ast import Constant, ENode, Variable
from etree.benchmarks import RecoveryCase

TargetFactory = Callable[[np.ndarray], np.ndarray]


x = Variable("x")
one = Constant(1.0)


# Tier A: exact E-native recovery cases.
TIER_A_NATIVE_CASES: tuple[RecoveryCase, ...] = (
    RecoveryCase(
        name="identity",
        tier="native_e",
        family="identity/compression",
        target=x,
        max_depth=1,
    ),
    RecoveryCase(
        name="constant_one",
        tier="native_e",
        family="identity/compression",
        target=one,
        max_depth=1,
    ),
    RecoveryCase(
        name="e_x",
        tier="native_e",
        family="growth/saturation",
        target=ENode(x, one),
        max_depth=2,
    ),
    RecoveryCase(
        name="nested",
        tier="native_e",
        family="growth/saturation",
        target=ENode(x, ENode(x, one)),
        max_depth=3,
    ),
    RecoveryCase(
        name="stacked_growth",
        tier="native_e",
        family="growth/saturation",
        target=ENode(ENode(x, one), one),
        max_depth=3,
    ),
    RecoveryCase(
        name="balanced_native",
        tier="native_e",
        family="geometry-like",
        target=ENode(ENode(x, one), ENode(x, one)),
        max_depth=3,
    ),
    RecoveryCase(
        name="offset_left",
        tier="native_e",
        family="geometry-like",
        target=ENode(one, ENode(x, one)),
        max_depth=3,
    ),
    RecoveryCase(
        name="positive_domain_right_leaf",
        tier="native_e",
        family="geometry-like",
        target=ENode(ENode(x, one), x),
        x_min=0.2,
        x_max=1.2,
        max_depth=3,
    ),
)


NON_NATIVE_CASES: tuple[RecoveryCase, ...] = (
    RecoveryCase(
        name="quadratic_bowl",
        tier="compressible_non_native",
        family="geometry-like",
        target_factory=lambda x: x**2 + 1.0,
    ),
    RecoveryCase(
        name="cubic_growth",
        tier="compressible_non_native",
        family="growth/saturation",
        target_factory=lambda x: x**3 + 0.5,
    ),
    RecoveryCase(
        name="sigmoid",
        tier="compressible_non_native",
        family="growth/saturation",
        target_factory=lambda x: 1.0 / (1.0 + np.exp(-x)),
    ),
    RecoveryCase(
        name="softplus_shift",
        tier="compressible_non_native",
        family="growth/saturation",
        target_factory=lambda x: np.log1p(np.exp(x - 0.4)),
    ),
    RecoveryCase(
        name="abs_plus",
        tier="compressible_non_native",
        family="identity/compression",
        target_factory=lambda x: np.abs(x) + 0.5,
    ),
    RecoveryCase(
        name="harmonic_mix",
        tier="compressible_non_native",
        family="identity/compression",
        target_factory=lambda x: 0.7 * x + 0.3 * np.sin(2.0 * x),
    ),
    RecoveryCase(
        name="log_shift",
        tier="compressible_non_native",
        family="geometry-like",
        x_min=-0.9,
        x_max=1.2,
        target_factory=lambda x: np.log1p(x + 1.2),
    ),
)


NEGATIVE_CONTROL_CASES: tuple[RecoveryCase, ...] = (
    RecoveryCase(
        name="high_freq_sine",
        tier="negative_control",
        family="adversarial",
        target_factory=lambda x: np.sin(25.0 * x),
    ),
    RecoveryCase(
        name="step_jump",
        tier="negative_control",
        family="adversarial",
        target_factory=lambda x: np.where(x >= 0.0, 3.0, -1.0),
    ),
    RecoveryCase(
        name="sawtooth",
        tier="negative_control",
        family="adversarial",
        target_factory=lambda x: (x - np.floor(x)) - 0.5,
    ),
    RecoveryCase(
        name="deterministic_noise",
        tier="negative_control",
        family="adversarial",
        target_factory=lambda x: np.random.default_rng(7).normal(0.0, 0.7, size=x.shape),
    ),
)


BENCHMARK_REGISTRY: tuple[RecoveryCase, ...] = (
    *TIER_A_NATIVE_CASES,
    *NON_NATIVE_CASES,
    *NEGATIVE_CONTROL_CASES,
)


def get_benchmark_registry() -> list[RecoveryCase]:
    """Return all benchmark cases as mutable list for downstream filtering."""
    return list(BENCHMARK_REGISTRY)
