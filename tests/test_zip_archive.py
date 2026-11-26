"""Tests for ZIP-archive upload flattening (open_notebook.utils.zip_archive)."""

import io
import zipfile

import pytest

from open_notebook.exceptions import InvalidInputError
from open_notebook.utils.zip_archive import is_zip_upload, zip_to_markdown


def _zip(files: dict[str, bytes]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_flattens_text_and_skips_noise():
    data = _zip(
        {
            "proj/README.md": b"# Hello",
            "proj/src/a.py": b"print('hi')",
            "proj/node_modules/x/i.js": b"junk",
            "proj/.git/config": b"junk",
            "proj/blob.bin": b"\x00\x01\x02" * 100,
        }
    )
    md, name = zip_to_markdown("proj.zip", data)
    text = md.decode()
    assert name == "proj.md"
    assert "proj/README.md" in text and "print('hi')" in text
    assert "node_modules" not in text and ".git/config" not in text
    assert "proj/blob.bin (binary)" in text


def test_skips_path_traversal_entries():
    data = _zip({"../evil.txt": b"x", "ok.txt": b"fine"})
    text = zip_to_markdown("a.zip", data)[0].decode()
    assert "evil" not in text and "fine" in text


def test_oversized_file_is_omitted():
    data = _zip({"big.txt": b"a" * (300 * 1024), "small.txt": b"ok"})
    text = zip_to_markdown("a.zip", data)[0].decode()
    assert "big.txt (too large or limit reached)" in text and "ok" in text


def test_nested_fence_uses_tildes():
    data = _zip({"d.md": b"```py\nx=1\n```"})
    assert "~~~~md" in zip_to_markdown("a.zip", data)[0].decode()


def test_invalid_zip_raises():
    with pytest.raises(InvalidInputError):
        zip_to_markdown("a.zip", b"PK\x03\x04not really a zip")


def test_is_zip_upload():
    assert is_zip_upload("a.ZIP", b"PK\x03\x04rest")
    assert not is_zip_upload("a.txt", b"PK\x03\x04rest")
    assert not is_zip_upload("a.zip", b"notzip")
