import os
from pathlib import Path

from huggingface_hub import snapshot_download


def download(repo_id: str, target_dir: Path) -> None:
    token = os.getenv("HUGGINGFACE_TOKEN") or None
    target_dir.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        local_dir=target_dir,
        local_dir_use_symlinks=False,
        token=token,
        resume_download=True,
    )


def main() -> None:
    base = Path("models")
    download("Qwen/Qwen2.5-7B-Instruct", base / "Qwen2.5-7B-Instruct")
    download("Qwen/Qwen2.5-1.5B-Instruct", base / "Qwen2.5-1.5B-Instruct")


if __name__ == "__main__":
    main()
