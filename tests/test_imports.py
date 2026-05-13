from __future__ import annotations


def test_imports_cleanly() -> None:
    import cw360
    import cw360.config
    import cw360.constants
    import cw360.tokenizer
    import cw360.tokenizer.loader
    import cw360.tokenizer.special_tokens
    import cw360.tokenizer.validate
    import cw360.utils.device
    import cw360.utils.env
    import cw360.utils.hashing
    import cw360.utils.logging
    import cw360.utils.seed

    assert cw360.MODEL_SHORT_NAME == "cw360"
