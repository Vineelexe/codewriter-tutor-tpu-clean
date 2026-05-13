from __future__ import annotations


def main() -> None:
    print(
        "Before launch, verify Kaggle's current TPU quota, accelerator availability, "
        "and session/runtime limits in each account UI and current Kaggle docs. "
        "Use the config session_time_limit_minutes as a conservative guard, not as a quota claim."
    )


if __name__ == "__main__":
    main()
