import zipfile

import pytest

from nereid.weights import resolve_weights


def test_result_archive_selects_best_checkpoint(tmp_path):
    archive = tmp_path / "results.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("weights/last.pt", b"last")
        handle.writestr("weights/best.pt", b"best")
    with resolve_weights(archive) as weights:
        assert weights.read_bytes() == b"best"
        extracted = weights
    assert not extracted.exists()


def test_ambiguous_checkpoints_are_rejected(tmp_path):
    archive = tmp_path / "ambiguous.zip"
    with zipfile.ZipFile(archive, "w") as handle:
        handle.writestr("a/best.pt", b"a")
        handle.writestr("b/best.pt", b"b")
    with pytest.raises(ValueError, match="exactly one"):
        with resolve_weights(archive):
            pass

