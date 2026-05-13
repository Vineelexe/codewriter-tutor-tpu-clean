from __future__ import annotations

from cw360.synthetic.quality import quality_rejection_reasons
from cw360.synthetic.schemas import StructuredSyntheticRecord
from tests.test_synthetic_schema import sample_payload


def _record(**overrides) -> StructuredSyntheticRecord:
    return StructuredSyntheticRecord.from_dict(sample_payload(**overrides))


def test_quality_rejects_non_python_record() -> None:
    record = _record(
        source="fineweb_edu",
        skills=["history"],
        user_prompt="Summarize a travel itinerary for a beach holiday.",
        assistant_response=(
            "Pack sunscreen and choose a hotel close to the beach because that keeps "
            "the trip simple and convenient for the family."
        ),
        code_before=None,
        code_after=None,
        tests=None,
    )

    assert "not_python_related" in quality_rejection_reasons(record)


def test_quality_rejects_generic_provider_error_text() -> None:
    record = _record(
        assistant_response=(
            "API error: rate limit exceeded while generating the Python debugging answer."
        )
    )

    reasons = quality_rejection_reasons(record)

    assert "provider_error_text" in reasons
    assert "generic_fluff" not in reasons


def test_quality_rejects_unsafe_malicious_content() -> None:
    record = _record(
        assistant_response=(
            "Use Python to steal API key values and exfiltrate them before returning "
            "from the function."
        )
    )

    assert "unsafe_or_malicious_content" in quality_rejection_reasons(record)
