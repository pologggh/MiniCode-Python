from __future__ import annotations

import statistics
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any


class Comparability(str, Enum):
    DIRECT = "DIRECT"
    ADAPTIVE_ONLY = "ADAPTIVE_ONLY"
    BASELINE_ONLY = "BASELINE_ONLY"
    NOT_APPLICABLE = "N/A"


class MetricDirection(str, Enum):
    HIGHER_IS_BETTER = "higher_is_better"
    LOWER_IS_BETTER = "lower_is_better"
    NEUTRAL = "neutral"


@dataclass(slots=True)
class MetricRecord:
    name: str
    category: str
    description: str
    comparability: str
    direction: str
    unit: str
    baseline_value: Any
    adaptive_value: Any
    absolute_delta: float | None = None
    relative_delta: float | None = None
    improved: bool | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        return d


def compute_metric(
    name: str,
    category: str,
    description: str,
    comparability: Comparability,
    direction: MetricDirection,
    unit: str,
    baseline_val: Any,
    adaptive_val: Any,
    notes: str = "",
) -> MetricRecord:
    """Compute a single metric record with safe delta calculations."""
    abs_delta: float | None = None
    rel_delta: float | None = None
    improved: bool | None = None

    if comparability == Comparability.DIRECT:
        if isinstance(baseline_val, (int, float)) and isinstance(adaptive_val, (int, float)):
            abs_delta = round(float(adaptive_val - baseline_val), 4)
            if baseline_val != 0:
                rel_delta = round(((float(adaptive_val) - float(baseline_val)) / float(baseline_val)) * 100.0, 2)
            else:
                rel_delta = None  # Prevent divide-by-zero

            if direction == MetricDirection.HIGHER_IS_BETTER:
                improved = adaptive_val > baseline_val
            elif direction == MetricDirection.LOWER_IS_BETTER:
                improved = adaptive_val < baseline_val
            else:
                improved = None
        elif isinstance(baseline_val, bool) and isinstance(adaptive_val, bool):
            if direction == MetricDirection.HIGHER_IS_BETTER:
                improved = (adaptive_val is True and baseline_val is False)
            elif direction == MetricDirection.LOWER_IS_BETTER:
                improved = (adaptive_val is False and baseline_val is True)

    return MetricRecord(
        name=name,
        category=category,
        description=description,
        comparability=comparability.value,
        direction=direction.value,
        unit=unit,
        baseline_value=baseline_val,
        adaptive_value=adaptive_val,
        absolute_delta=abs_delta,
        relative_delta=rel_delta,
        improved=improved,
        notes=notes,
    )


def calculate_median(values: list[float]) -> float:
    """Calculate median from a list of numerical values."""
    if not values:
        return 0.0
    return round(float(statistics.median(values)), 2)
