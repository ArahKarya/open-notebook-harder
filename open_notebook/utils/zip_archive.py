"""Flatten a ZIP archive of text/code files into a single Markdown document.

content-core cannot read archives, so uploads of ``.zip`` would be rejected
with HTTP 415. This module turns an archive into one Markdown document (file
tree + the text content of every readable file) so it can be ingested like any
other document, searched, and used as chat context.

Entries are read in memory only; nothing is extracted to disk.
"""

import io
import zipfile

from open_notebook.exceptions import InvalidInputError

SKIP_DIRS = frozenset(
    {
        ".git",
        "node_modules",
        "__pycache__",
        ".venv",
        "venv",
        "dist",
        "build",
        ".next",
        "target",
        ".idea",
        ".vscode",
    }
)
MAX_FILE_BYTES = 200 * 1024
MAX_TOTAL_BYTES = 5 * 1024 * 1024
MAX_UNCOMPRESSED_BYTES = 200 * 1024 * 1024
MAX_ENTRIES = 5000


def is_zip_upload(filename: str, content: bytes) -> bool:
    return filename.lower().endswith(".zip") and content[:4] == b"PK\x03\x04"


def _skipped(name: str) -> bool:
    parts = name.replace("\\", "/").split("/")
    return (
        name.startswith("/")
        or ".." in parts
        or any(p in SKIP_DIRS for p in parts)
        or parts[-1].startswith(".DS_Store")
    )


def zip_to_markdown(filename: str, content: bytes) -> tuple[bytes, str]:
    """Return ``(markdown_bytes, new_filename)`` for a ZIP archive."""
    try:
        zf = zipfile.ZipFile(io.BytesIO(content))
    except zipfile.BadZipFile:
        raise InvalidInputError("Invalid ZIP archive")

    infos = [i for i in zf.infolist() if not i.is_dir()]
    if len(infos) > MAX_ENTRIES:
        raise InvalidInputError(f"ZIP has too many files (max {MAX_ENTRIES})")
    if sum(i.file_size for i in infos) > MAX_UNCOMPRESSED_BYTES:
        raise InvalidInputError("ZIP too large when uncompressed (max 200MB)")

    infos = sorted(
        (i for i in infos if not _skipped(i.filename)), key=lambda i: i.filename
    )
    tree = "\n".join(f"- {i.filename} ({i.file_size} B)" for i in infos)
    out = [
        f"# Archive: {filename}\n",
        "## File tree\n",
        tree or "(empty)",
        "\n## Files\n",
    ]
    total = 0
    omitted: list[str] = []
    for i in infos:
        if i.file_size > MAX_FILE_BYTES or total >= MAX_TOTAL_BYTES:
            omitted.append(f"{i.filename} (too large or limit reached)")
            continue
        raw = zf.read(i)
        if b"\x00" in raw[:4096]:
            omitted.append(f"{i.filename} (binary)")
            continue
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            omitted.append(f"{i.filename} (not UTF-8)")
            continue
        total += len(raw)
        ext = i.filename.rsplit(".", 1)[-1] if "." in i.filename else ""
        fence = "~~~~" if "```" in text else "```"
        out.append(
            f"\n### {i.filename}\n\n{fence}{ext}\n{text.rstrip()}\n{fence}\n"
        )
    if omitted:
        out.append("\n## Omitted files\n\n" + "\n".join(f"- {o}" for o in omitted))
    return "\n".join(out).encode("utf-8"), filename.rsplit(".", 1)[0] + ".md"
