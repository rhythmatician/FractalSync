"""Restore one pinned, unshipped geometry experiment into a new directory."""
from pathlib import Path, PurePosixPath
import argparse
import hashlib
import json
import zipfile


def main() -> None:
    location = Path(__file__).resolve()
    repository = location.parents[4]
    record = json.loads((location.parents[2] / "geometry-145-refinement.json").read_text(encoding="utf-8"))
    snapshots: dict[str, dict[str, str]] = record["snapshots"]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--snapshot", required=True, choices=sorted(snapshots))
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    output = args.output.resolve()
    if output.exists():
        raise SystemExit(f"Output must be a new directory: {output}")
    archive_path = location.parent / "sources.zip"
    digest = hashlib.sha256(archive_path.read_bytes()).hexdigest()
    if digest != record["source_archive_sha256"]:
        raise SystemExit("Source archive hash mismatch")

    def save(relative: str, data: bytes, expected: str) -> None:
        parts = PurePosixPath(relative)
        if parts.is_absolute() or ".." in parts.parts:
            raise SystemExit(f"Invalid snapshot path: {relative}")
        target = output.joinpath(*parts.parts).resolve()
        if not target.is_relative_to(output):
            raise SystemExit(f"Snapshot path escapes output: {relative}")
        if hashlib.sha256(data).hexdigest() != expected:
            raise SystemExit(f"Snapshot content hash mismatch: {relative}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)

    with zipfile.ZipFile(archive_path) as archive:
        for relative, expected in snapshots[args.snapshot].items():
            save(relative, archive.read(f"blobs/{expected}"), expected)
    scaffold: dict[str, str] = record["repository_scaffold"]
    for relative, expected in scaffold.items():
        save(relative, (repository / relative).read_bytes(), expected)
    print(f"Restored {args.snapshot} with verified source hashes to {output}")


if __name__ == "__main__":
    main()
