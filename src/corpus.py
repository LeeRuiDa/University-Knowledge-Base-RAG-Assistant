from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import httpx

from src.config import Settings

ACTIVE_STATUSES = {"active", "ready"}
USER_AGENT = "University-Knowledge-Base-RAG-Assistant/0.1 (+https://openai.com)"


@dataclass(frozen=True)
class CorpusEntry:
    doc_id: str
    url: str
    file_path: str
    title: str
    doc_type: str
    year: int | None
    program: str | None
    status: str
    notes: str | None = None

    @property
    def output_path(self) -> Path:
        return Path(self.file_path)

    @property
    def is_active(self) -> bool:
        return self.status.strip().lower() in ACTIVE_STATUSES


def load_corpus_manifest(manifest_path: Path) -> list[CorpusEntry]:
    if not manifest_path.exists():
        return []

    with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        return [
            CorpusEntry(
                doc_id=row["doc_id"].strip(),
                url=row["url"].strip(),
                file_path=row["file_path"].strip(),
                title=row["title"].strip(),
                doc_type=row["doc_type"].strip(),
                year=_parse_year(row.get("year", "")),
                program=_empty_to_none(row.get("program", "")),
                status=row.get("status", "active").strip() or "active",
                notes=_empty_to_none(row.get("notes", "")),
            )
            for row in reader
            if row.get("doc_id") and row.get("url") and row.get("file_path")
        ]


def sync_corpus_entries(
    entries: list[CorpusEntry],
    settings: Settings,
    force: bool = False,
) -> list[CorpusEntry]:
    settings.ensure_directories()
    active_entries = [entry for entry in entries if entry.is_active]
    if not active_entries:
        return []

    with httpx.Client(
        follow_redirects=True,
        headers={"User-Agent": USER_AGENT},
        timeout=45.0,
    ) as client:
        for entry in active_entries:
            output_path = entry.output_path
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if output_path.exists() and not force:
                continue

            response = client.get(entry.url)
            response.raise_for_status()

            if output_path.suffix.lower() == ".pdf":
                output_path.write_bytes(response.content)
            else:
                output_path.write_text(response.text, encoding="utf-8")

    return active_entries


def build_metadata_overrides(entries: list[CorpusEntry]) -> dict[str, dict[str, object]]:
    overrides: dict[str, dict[str, object]] = {}
    for entry in entries:
        source = _relative_source(entry.output_path)
        overrides[source] = {
            "doc_id": entry.doc_id,
            "title": entry.title,
            "doc_type": entry.doc_type,
            "year": entry.year,
            "url": entry.url,
            "program": entry.program,
        }
    return overrides


def sync_manifest_corpus(
    settings: Settings,
    force: bool = False,
) -> tuple[list[CorpusEntry], dict[str, dict[str, object]]]:
    entries = load_corpus_manifest(settings.corpus_manifest_path)
    active_entries = sync_corpus_entries(entries, settings=settings, force=force)
    return active_entries, build_metadata_overrides(active_entries)


def _relative_source(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _parse_year(value: str) -> int | None:
    value = value.strip()
    return int(value) if value else None


def _empty_to_none(value: str) -> str | None:
    value = value.strip()
    return value or None
