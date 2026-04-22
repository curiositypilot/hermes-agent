from __future__ import annotations

from pathlib import Path


def ensure_directory(path: str | Path) -> Path:
    directory = Path(path)
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def discover_media_files(input_dir: str | Path, supported_extensions: list[str], recursive: bool = False) -> list[Path]:
    root = Path(input_dir)
    if not root.exists():
        return []
    pattern = "**/*" if recursive else "*"
    allowed = {extension.lower() for extension in supported_extensions}
    return sorted(
        path for path in root.glob(pattern) if path.is_file() and path.suffix.lower() in allowed
    )


def output_stem(source_file: str | Path) -> str:
    return Path(source_file).stem
