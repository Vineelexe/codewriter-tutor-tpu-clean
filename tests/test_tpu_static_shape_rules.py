from __future__ import annotations

import sys

import pytest

from cw360.train.validate import validate_static_shape_rules


def test_static_shape_rules_require_drop_last_and_static_shapes() -> None:
    with pytest.raises(ValueError, match="drop_last"):
        validate_static_shape_rules(
            max_seq_len=1024,
            batch_size=8,
            drop_last=False,
            static_shapes=True,
        )
    with pytest.raises(ValueError, match="static_shapes"):
        validate_static_shape_rules(
            max_seq_len=1024,
            batch_size=8,
            drop_last=True,
            static_shapes=False,
        )


def test_tpu_modules_do_not_import_torch_xla_on_local_import() -> None:
    sys.modules.pop("torch_xla", None)

    import cw360.train.tpu_trainer  # noqa: F401
    import cw360.train.xla_utils  # noqa: F401

    assert "torch_xla" not in sys.modules
