from __future__ import annotations

import os
from pathlib import Path
import time


def rotate_jsonl_if_needed(path: str | Path) -> None:
    file_path = Path(path)
    if not file_path.exists():
        return
    max_mb = float(os.getenv("LOG_ROTATE_MAX_MB", "10"))
    keep_files = max(int(os.getenv("LOG_ROTATE_KEEP_FILES", "2")), 1)
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


def prune_directory_files(
    directory: str | Path,
    *,
    pattern: str = "*",
    keep_latest: int = 0,
    max_age_days: float | None = None,
) -> None:
    target_dir = Path(directory)
    if not target_dir.exists():
        return
    files = [path for path in target_dir.glob(pattern) if path.is_file()]
    if not files:
        return
    files.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    now = time.time()
    protected = set(files[: max(keep_latest, 0)])
    max_age_sec = max_age_days * 86400.0 if max_age_days is not None and max_age_days >= 0 else None
    for path in files:
        if path in protected:
            continue
        if max_age_sec is not None:
            age_sec = now - path.stat().st_mtime
            if age_sec < max_age_sec:
                continue
        try:
            path.unlink()
        except OSError:
            continue
