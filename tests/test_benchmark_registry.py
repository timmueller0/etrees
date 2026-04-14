from etree.benchmark_registry import get_benchmark_registry
from etree.benchmarks import run_recovery_case


def test_registry_spans_all_tiers_and_families() -> None:
    cases = get_benchmark_registry()

    assert 15 <= len(cases) <= 25
    assert {case.tier for case in cases} == {
        "native_e",
        "compressible_non_native",
        "negative_control",
    }
    assert {case.family for case in cases} == {
        "growth/saturation",
        "geometry-like",
        "identity/compression",
        "adversarial",
    }


def test_non_native_case_uses_target_factory_path() -> None:
    case = next(c for c in get_benchmark_registry() if c.tier == "compressible_non_native")

    out = run_recovery_case(case)

    assert out.name == case.name
    assert out.tier == "compressible_non_native"
    assert out.exact_recovered is False
