import os
import sys
from pathlib import Path


def _add_src_to_path() -> None:
    root = Path(__file__).resolve().parents[1]
    src_path = root / "src"
    sys.path.insert(0, str(src_path))


_add_src_to_path()
