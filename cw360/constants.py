from __future__ import annotations

MODEL_SHORT_NAME = "cw360"
CHECKPOINT_NAME_TEMPLATE = "{model_short_name}-{size_label}-{training_stage}-{trainer}-{step:08d}"

VALID_TRAINERS = frozenset({"vineel", "sarang"})

VALID_PLATFORMS = frozenset(
    {
        "local-cpu-test",
        "cloud-cpu-prepack",
        "kaggle-cpu-prepack",
        "colab-cpu-prepack",
        "codespaces-cpu-prepack",
        "kaggle-tpu",
        "kaggle-tpu-background",
        "kaggle-t4x2-legacy",
        "lightning-cpu",
    }
)

VALID_TRAINING_STAGES = frozenset(
    {
        "base_pretrain",
        "fim_train",
        "instruction_tune",
        "eval_only",
    }
)

VALID_MODEL_SIZE_LABELS = frozenset({"354m", "420m", "480m", "tiny"})
