"""Server-owned policies applied after checking and before persistence."""

from __future__ import annotations

from boxcheck.types import AnalysisResult, Verdict


__all__ = ["apply_review_policy"]


def apply_review_policy(result: AnalysisResult, attempt: int) -> str:
    """Convert a repeated insufficient result to a server review verdict.

    The standalone checker remains the owner of whether evidence is
    insufficient. The server owns the workflow decision that a second or
    later inconclusive attempt should be reviewed by a person.
    """
    if not isinstance(result, AnalysisResult):
        raise TypeError("result must be an AnalysisResult")
    if isinstance(attempt, bool) or not isinstance(attempt, int):
        raise TypeError("attempt must be an integer")
    if attempt < 1:
        raise ValueError("attempt must be at least 1")

    if result.verdict is Verdict.INSUFFICIENT and attempt >= 2:
        return "review"
    return result.verdict.value
