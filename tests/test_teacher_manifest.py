from __future__ import annotations

from cw360.teacher.manifest import TeacherRunManifest, load_manifest, save_manifest


def test_manifest_tracks_completed_and_failed_candidates(tmp_path) -> None:
    manifest = TeacherRunManifest(
        run_id="run-1",
        provider="mock",
        model="mock-teacher",
        output_dir=str(tmp_path),
    )
    manifest.mark_failed("cand-1", "bad json")
    manifest.mark_completed("cand-2", "synth-2")

    path = tmp_path / "manifest.json"
    save_manifest(manifest, path)
    loaded = load_manifest(path)

    assert loaded is not None
    assert loaded.failed_candidates == {"cand-1": "bad json"}
    assert loaded.completed_candidates == {"cand-2": "synth-2"}
    assert loaded.total_requests == 2
