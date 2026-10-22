"""How often the cap rule scales, as a function of the true cost, on fresh draws of the generator."""

from __future__ import annotations

from dataclasses import dataclass

from budget_rule.cost import cost_against_cap

from reconcile.dgp import MAX_COST_PER_INCREMENTAL, N_TEST_GEO, TEST_WEEKS, experiment_cost, generate
from reconcile.methods import geo_incremental

TRUE_COST_GRID = (0.03, 0.05, 0.08, 0.10, 0.12, 0.15)


@dataclass(frozen=True)
class OperatingPoint:
    true_cost_per_incremental: float
    increment_per_geo_week: float
    n_seeds: int
    scale_share: float
    coverage: float

    def as_dict(self) -> dict[str, float | int]:
        return dict(self.__dict__)


def increment_for_cost(true_cost: float, test_cost: float) -> float:
    """Planted increment per treated geo-week that puts the true cost at `true_cost`."""
    if true_cost <= 0 or test_cost <= 0:
        raise ValueError("Costs must be positive.")
    return test_cost / true_cost / (N_TEST_GEO * TEST_WEEKS)


def operating_point(
    true_cost: float,
    *,
    n_seeds: int,
    first_seed: int,
    max_cost_per_incremental: float = MAX_COST_PER_INCREMENTAL,
) -> OperatingPoint:
    if n_seeds < 1:
        raise ValueError("Need at least one seed.")
    test_cost = experiment_cost(generate(first_seed, include_paths=False))
    increment = increment_for_cost(true_cost, test_cost)
    scaled = 0
    covered = 0
    for seed in range(first_seed, first_seed + n_seeds):
        world = generate(seed, increment=increment, include_paths=False)
        estimate = geo_incremental(world)
        scaled += int(cost_against_cap(estimate.low, experiment_cost(world), max_cost_per_incremental).clears)
        covered += int(estimate.low <= world.true_incremental <= estimate.high)
    return OperatingPoint(
        true_cost_per_incremental=true_cost,
        increment_per_geo_week=increment,
        n_seeds=n_seeds,
        scale_share=scaled / n_seeds,
        coverage=covered / n_seeds,
    )


def operating_curve(
    *,
    grid: tuple[float, ...] = TRUE_COST_GRID,
    n_seeds: int,
    first_seed: int,
) -> list[OperatingPoint]:
    return [operating_point(cost, n_seeds=n_seeds, first_seed=first_seed) for cost in grid]
