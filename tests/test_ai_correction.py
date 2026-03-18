import pytest

from core.correction import ai_suggest_corrections


def test_ai_suggest_corrections_requires_api_key() -> None:
    with pytest.raises(ValueError, match="API key"):
        ai_suggest_corrections(["A", "a"], api_key="")
