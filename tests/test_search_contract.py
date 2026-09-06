"""Lab 5 reliability contract for a persisted search index."""
import json
from pathlib import Path
from bayan.search.index import build_index


def test_index_manifest_pins_required_versions(tmp_path: Path):
    prefix = tmp_path / "case_index_v1"
    build_index(prefix=str(prefix), limit=20)
    manifest_path = Path(f"{prefix}_manifest.json")
    assert manifest_path.exists()
    manifest = json.loads(manifest_path.read_text())
    for key in ["model", "preproc_version", "n_vectors", "dim"]:
        assert key in manifest
    assert manifest["n_vectors"] > 0
