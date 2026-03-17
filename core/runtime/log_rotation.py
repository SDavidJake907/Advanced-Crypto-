from __future__ import annotations

import os
from pathlib import Path


def rotate_jsonl_if_needed(path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.exists():
        return
    max_mb = float(os.getenv("LOG_ROTATE_MAX_MB", "25"))
    keep_files = max(int(os.getenv("LOG_ROTATE_KEEP_FILES", "3")), 1)
    max_bytes = int(max_mb * 1024 * 1024)
    if max_bytes <= 0 or file_path.stat().st_size < max_bytes:
        return

    oldest = file_path.with_suffix(file_path.suffix + f".{keep_files}")
    if oldest.exists():
        oldest.unlink()
    for idx in range(keep_files - 1, 0, -1):
        src = file_path.with_suffix(file_path.suffix + f".{idx}")
        dst = file_path.with_suffix(file_path.suffix + f".{idx + 1}")
        if src.exists():
            src.replace(dst)
    file_path.replace(file_path.with_suffix(file_path.suffix + ".1"))
