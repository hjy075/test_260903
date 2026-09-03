from collections.abc import Iterable


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
