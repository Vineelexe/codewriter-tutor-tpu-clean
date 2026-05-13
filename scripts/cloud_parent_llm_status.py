from __future__ import annotations

import argparse
import json

from cw360.teacher.engine import load_teacher_engine_config
from cw360.teacher.manifest import load_manifest, manifest_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Print parent-LLM generation manifest status.")
    parser.add_argument("--config", default="configs/teacher_generation.yaml")
    args = parser.parse_args()

    config = load_teacher_engine_config(args.config)
    path = manifest_path(config.local_temp_dir / config.run_id, config.run_id)
    manifest = load_manifest(path)
    if manifest is None:
        print(f"manifest_missing path={path}")
        return
    print(json.dumps(manifest.to_dict(), indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
