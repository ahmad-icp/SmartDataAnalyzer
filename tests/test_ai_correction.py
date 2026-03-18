from core.correction import ai_suggest_corrections


def test_ai_suggest_corrections_returns_mapping_without_api() -> None:
    mapping = ai_suggest_corrections(["Fast Nuces", "FAST-NUCES", "fast_nuces"])
    assert mapping
    assert set(mapping.keys()) == {"Fast Nuces", "FAST-NUCES", "fast_nuces"}
