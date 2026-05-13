from __future__ import annotations

import argparse

from cw360.teacher.engine import load_teacher_engine_config
from cw360.teacher.remote_store import HuggingFaceDatasetStore, MockRemoteStore


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Push structured synthetic shards to remote storage."
    )
    parser.add_argument("--config", default="configs/teacher_generation.yaml")
    parser.add_argument("--mock", action="store_true")
    args = parser.parse_args()

    config = load_teacher_engine_config(args.config)
    output_dir = config.local_temp_dir / config.run_id
    if args.mock:
        result = MockRemoteStore().upload_dir(output_dir, delete_local_after_upload=False)
    else:
        result = HuggingFaceDatasetStore(
            str(config.storage["repo_id"]),
            private=bool(config.storage.get("private", True)),
        ).upload_dir(
            output_dir,
            delete_local_after_upload=bool(config.storage.get("delete_local_after_upload", False)),
        )
    print(
        f"uploaded={result.uploaded} destination={result.destination} files={len(result.files)}"
    )


if __name__ == "__main__":
    main()
