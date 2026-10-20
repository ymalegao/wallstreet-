from __future__ import annotations

import hashlib
import json
import zipfile
from datetime import date
from pathlib import Path

import pytest

from ws.data.downloads import download_command
from ws.fundamentals.industry import parse_ff12_sic_zip


class _Response:
    status_code = 200
    is_error = False

    def __enter__(self) -> _Response:
        return self

    def __exit__(self, *_args: object) -> None:
        return None

    def iter_bytes(self, *, chunk_size: int):
        assert chunk_size > 0
        yield b'{"small":'
        yield b" fixture}\n"


class _Client:
    def __init__(self) -> None:
        self.request: tuple[str, str, dict[str, str]] | None = None

    def stream(
        self,
        method: str,
        url: str,
        *,
        headers: dict[str, str],
        follow_redirects: bool,
    ) -> _Response:
        assert follow_redirects is True
        self.request = (method, url, headers)
        return _Response()


def test_sec_download_refuses_missing_user_agent(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("SEC_USER_AGENT", raising=False)
    monkeypatch.chdir(tmp_path)

    with pytest.raises(ValueError, match="SEC_USER_AGENT"):
        download_command("sec-tickers", root=tmp_path / "data", snapshot_date=date(2026, 1, 2))


def test_mocked_download_writes_manifest_with_content_hash(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    body = b'{"small": fixture}\n'
    client = _Client()
    monkeypatch.setenv("SEC_USER_AGENT", "Example Researcher researcher@example.test")

    result = download_command(
        "sec-tickers",
        root=tmp_path / "data",
        snapshot_date=date(2026, 1, 2),
        client=client,  # type: ignore[arg-type]
    )

    target = Path(str(result["path"]))
    manifest = json.loads((target.parent / "manifest.json").read_text(encoding="utf-8"))
    assert target.read_bytes() == body
    assert result["sha256"] == hashlib.sha256(body).hexdigest()
    assert manifest["files"][target.name]["bytes"] == len(body)
    assert manifest["files"][target.name]["sha256"] == hashlib.sha256(body).hexdigest()
    assert client.request is not None
    assert client.request[0] == "GET"
    assert client.request[2]["User-Agent"] == "Example Researcher researcher@example.test"


def test_ff12_parser_expands_the_sic_ranges(tmp_path: Path) -> None:
    content = "\n".join(
        [
            "1 NoDur Consumer Nondurables",
            "          0100-0102",
            *(f"{number} Industry{number}" for number in range(2, 13)),
            "",
        ]
    )
    archive_path = tmp_path / "Siccodes12.zip"
    with zipfile.ZipFile(archive_path, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("Siccodes12.txt", content)

    mapping = parse_ff12_sic_zip(archive_path)

    assert mapping == {100: "NoDur", 101: "NoDur", 102: "NoDur"}
