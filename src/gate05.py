from collections.abc import Iterable
from math import inf, sqrt
from statistics import median


def active_in_window(observed: Iterable[bool], min_weeks: int) -> bool:
    """Return True when price/availability is observed in at least min_weeks."""
    if min_weeks < 1:
        raise ValueError("min_weeks must be >= 1")
    values = list(observed)
    if not values:
        return False
    return sum(bool(x) for x in values) >= min_weeks


def classify_future_continuation(observed: Iterable[bool], min_weeks: int) -> bool:
    """Classify whether selling presence continues in a future window."""
    return active_in_window(observed, min_weeks=min_weeks)


def window_bounds(origin: int, width: int, offset_end: int) -> tuple[int, int]:
    """Return [start, end) indices for a fixed-width window ending at origin+offset_end."""
    if width < 1:
        raise ValueError("width must be >= 1")
    end = origin + offset_end + 1
    start = end - width
    return start, end


def standardized_mean_difference(group_a: Iterable[float], group_b: Iterable[float]) -> float:
    """Cohen-style standardized difference: mean(group_b)-mean(group_a) over pooled SD."""
    a = [float(x) for x in group_a]
    b = [float(x) for x in group_b]
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    mean_a = sum(a) / len(a)
    mean_b = sum(b) / len(b)
    var_a = sum((x - mean_a) ** 2 for x in a) / (len(a) - 1)
    var_b = sum((x - mean_b) ** 2 for x in b) / (len(b) - 1)
    pooled_var = ((len(a) - 1) * var_a + (len(b) - 1) * var_b) / (len(a) + len(b) - 2)
    if pooled_var == 0:
        if mean_b == mean_a:
            return 0.0
        return inf if mean_b > mean_a else -inf
    return (mean_b - mean_a) / sqrt(pooled_var)


def gate05_decision(
    minority_shares: Iterable[float],
    median_abs_smd_by_feature: dict[str, float],
    aucs: Iterable[float],
) -> dict[str, float | str | dict[str, float]]:
    """Pre-registered Gate-0.5 decision rule for the 52-week continuation analysis."""
    shares = [float(x) for x in minority_shares]
    auc_values = [float(x) for x in aucs]
    if not shares:
        raise ValueError("minority_shares must not be empty")

    median_minority_share = median(shares)
    max_median_abs_smd = max(median_abs_smd_by_feature.values(), default=float("nan"))
    median_auc = median(auc_values) if auc_values else float("nan")

    if median_minority_share < 0.05:
        status = "HARD_KILL"
    elif median_minority_share < 0.10:
        status = "WEAK_HOLD"
    elif max_median_abs_smd >= 0.20 or median_auc >= 0.60:
        status = "PASS_GATE_0_5"
    else:
        status = "HOLD"

    return {
        "status": status,
        "median_minority_share": median_minority_share,
        "max_median_abs_smd": max_median_abs_smd,
        "median_auc": median_auc,
        "median_abs_smd_by_feature": dict(median_abs_smd_by_feature),
    }
