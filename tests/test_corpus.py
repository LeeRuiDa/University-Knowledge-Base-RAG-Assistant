from pathlib import Path

from src.corpus import build_metadata_overrides, load_corpus_manifest


def test_load_corpus_manifest_parses_rows(tmp_path: Path) -> None:
    manifest_path = tmp_path / "corpus_manifest.csv"
    manifest_path.write_text(
        (
            "doc_id,url,file_path,title,doc_type,year,program,status,notes\n"
            "unl_cs,https://example.edu/doc,data/raw/unl/doc.html,UNL CS,degree_requirements,"
            "2025,computer_science_undergraduate,active,Primary program page\n"
        ),
        encoding="utf-8",
    )

    entries = load_corpus_manifest(manifest_path)

    assert len(entries) == 1
    assert entries[0].doc_id == "unl_cs"
    assert entries[0].year == 2025
    assert entries[0].is_active is True


def test_build_metadata_overrides_uses_manifest_values(tmp_path: Path) -> None:
    output_path = tmp_path / "data" / "raw" / "unl" / "doc.html"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text("<html><body>Test</body></html>", encoding="utf-8")

    entries = load_corpus_manifest(
        _write_manifest(
            tmp_path,
            file_path=output_path.as_posix(),
        )
    )

    overrides = build_metadata_overrides(entries)

    key = next(iter(overrides))
    assert overrides[key]["doc_type"] == "degree_requirements"
    assert overrides[key]["program"] == "computer_science_undergraduate"
    assert overrides[key]["url"] == "https://example.edu/doc"


def _write_manifest(tmp_path: Path, file_path: str) -> Path:
    manifest_path = tmp_path / "corpus_manifest.csv"
    manifest_path.write_text(
        (
            "doc_id,url,file_path,title,doc_type,year,program,status,notes\n"
            f"unl_cs,https://example.edu/doc,{file_path},UNL CS,degree_requirements,"
            "2025,computer_science_undergraduate,active,\n"
        ),
        encoding="utf-8",
    )
    return manifest_path
