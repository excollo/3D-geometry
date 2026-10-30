"""Tests for server-owned pre-persistence result policies."""

from pathlib import Path

import pytest

from app.policies import apply_review_policy
from boxcheck.types import AnalysisResult, Verdict


def _result(verdict: Verdict) -> AnalysisResult:
    return AnalysisResult(
        verdict=verdict,
        score=0.5,
        confidence=0.4,
        reason="Test result.",
        debug_frames=[Path("debug.jpg")],
    )


def test_second_insufficient_attempt_returns_review_verdict() -> None:
    checker_result = _result(Verdict.INSUFFICIENT)

    transformed = apply_review_policy(checker_result, 2)

    assert transformed == "review"
    assert checker_result.verdict is Verdict.INSUFFICIENT


@pytest.mark.parametrize("attempt", [1])
def test_first_insufficient_attempt_is_unchanged(attempt: int) -> None:
    checker_result = _result(Verdict.INSUFFICIENT)

    assert apply_review_policy(checker_result, attempt) == "insufficient"


@pytest.mark.parametrize("verdict", [Verdict.THREE_D, Verdict.FLAT])
def test_conclusive_verdict_is_unchanged(verdict: Verdict) -> None:
    checker_result = _result(verdict)

    assert apply_review_policy(checker_result, 3) == verdict.value


@pytest.mark.parametrize(
    "attempt,exception_type,message",
    [
        (True, TypeError, "attempt must be an integer"),
        (1.5, TypeError, "attempt must be an integer"),
        (0, ValueError, "attempt must be at least 1"),
    ],
)
def test_review_policy_validates_attempt(
    attempt: object,
    exception_type: type[Exception],
    message: str,
) -> None:
    with pytest.raises(exception_type, match=message):
        apply_review_policy(_result(Verdict.INSUFFICIENT), attempt)  # type: ignore[arg-type]


def test_review_policy_validates_result() -> None:
    with pytest.raises(TypeError, match="result must be an AnalysisResult"):
        apply_review_policy(object(), 2)  # type: ignore[arg-type]
