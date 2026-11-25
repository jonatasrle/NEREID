from contextlib import contextmanager
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import zipfile


@contextmanager
def resolve_weights(source):
    source = Path(source).expanduser().resolve()
    if not source.is_file():
        raise ValueError(f"Weights not found: {source}")
    if source.suffix.lower() == ".pt":
        yield source
        return
    if source.suffix.lower() != ".zip":
        raise ValueError("Weights must be a .pt file or a training-results ZIP.")
    with tempfile.TemporaryDirectory(prefix="nereid_weights_") as directory:
        destination = Path(directory) / "best.pt"
        with zipfile.ZipFile(source) as archive:
            candidates = [item for item in archive.infolist()
                          if not item.is_dir()
                          and PurePosixPath(item.filename.replace("\\", "/")).name == "best.pt"]
            if len(candidates) != 1:
                raise ValueError("The archive must contain exactly one best.pt file.")
            with archive.open(candidates[0]) as src, destination.open("wb") as dst:
                shutil.copyfileobj(src, dst)
        yield destination

