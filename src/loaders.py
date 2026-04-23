from __future__ import annotations

import re
from pathlib import Path
from typing import Any

from bs4 import BeautifulSoup
from langchain_core.documents import Document
from markdown import markdown
from pypdf import PdfReader

SUPPORTED_EXTENSIONS = {".pdf", ".html", ".htm", ".md", ".markdown", ".txt"}

DOC_TYPE_PATTERNS = {
    "program_handbook": ("handbook", "program"),
    "course_catalog": ("catalog", "course"),
    "graduation_requirements": ("graduation", "degree"),
    "internship_rules": ("internship", "placement"),
    "thesis_guidelines": ("thesis", "capstone", "project"),
    "fee_payment_faq": ("fee", "payment", "tuition"),
    "academic_calendar": ("calendar", "semester"),
    "student_services_faq": ("services", "student-service", "support"),
}

YEAR_PATTERN = re.compile(r"(20\d{2}|19\d{2})")
MARKDOWN_HEADING_PATTERN = re.compile(r"^(#{1,3})\s+(.*)$")
NOISE_HEADINGS = {
    "log in",
    "search form",
    "menu",
    "visit",
    "apply to",
    "give to",
    "contact us",
    "related links",
    "campus links",
    "policies & reports",
    "outline",
}
NOISE_LINE_PREFIXES = (
    "skip to main content",
    "search form",
    "submit",
    "close",
    "log in",
)


def load_documents_from_path(
    input_dir: Path,
    metadata_overrides: dict[str, dict[str, Any]] | None = None,
) -> list[Document]:
    if not input_dir.exists():
        raise FileNotFoundError(f"Input directory does not exist: {input_dir}")

    documents: list[Document] = []
    for path in sorted(input_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue

        metadata_override = _override_for_path(path, metadata_overrides)
        suffix = path.suffix.lower()
        if suffix == ".pdf":
            documents.extend(_load_pdf(path, metadata_override=metadata_override))
        elif suffix in {".html", ".htm"}:
            documents.extend(_load_html(path, metadata_override=metadata_override))
        else:
            documents.extend(
                _load_markdown_or_text(
                    path,
                    metadata_override=metadata_override,
                )
            )

    return documents


def _load_pdf(
    path: Path,
    metadata_override: dict[str, Any] | None = None,
) -> list[Document]:
    reader = PdfReader(str(path))
    title = _clean_text(
        (reader.metadata.title if reader.metadata and reader.metadata.title else "") or path.stem
    )
    base_metadata = _build_metadata(
        path,
        title=title,
        section=None,
        page=None,
        metadata_override=metadata_override,
    )

    documents: list[Document] = []
    for page_number, page in enumerate(reader.pages, start=1):
        text = _clean_text(page.extract_text() or "")
        if not text:
            continue
        metadata = {**base_metadata, "page": page_number, "section": f"Page {page_number}"}
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _load_html(
    path: Path,
    metadata_override: dict[str, Any] | None = None,
) -> list[Document]:
    html = path.read_text(encoding="utf-8")
    soup = BeautifulSoup(html, "html.parser")
    content_root = _select_html_content_root(soup)
    _strip_html_noise(content_root)
    title = _clean_text(
        (metadata_override or {}).get("title", "")
        or (content_root.find("h1").get_text(" ", strip=True) if content_root.find("h1") else "")
        or
        (soup.title.string if soup.title and soup.title.string else "")
        or path.stem
    )
    base_metadata = _build_metadata(
        path,
        title=title,
        page=None,
        metadata_override=metadata_override,
        section=None,
    )

    sections = _split_html_sections(content_root, fallback_title=title)
    documents: list[Document] = []
    for heading, section_text in sections:
        text = _clean_text(section_text)
        if not text or (heading and heading == title and text == title):
            continue
        metadata = {**base_metadata, "section": heading or title}
        documents.append(Document(page_content=text, metadata=metadata))

    if documents:
        return documents

    text = _clean_text(content_root.get_text("\n"))
    if not text:
        return []
    metadata = {**base_metadata, "section": title}
    return [Document(page_content=text, metadata=metadata)]


def _load_markdown_or_text(
    path: Path,
    metadata_override: dict[str, Any] | None = None,
) -> list[Document]:
    raw_text = path.read_text(encoding="utf-8")
    title, sections = _split_markdown_sections(raw_text, fallback_title=path.stem)
    base_metadata = _build_metadata(
        path,
        title=title,
        section=None,
        page=None,
        metadata_override=metadata_override,
    )

    documents: list[Document] = []
    for heading, section_text in sections:
        if path.suffix.lower() != ".txt":
            text = _markdown_to_text(section_text)
        else:
            text = _clean_text(section_text)
        if not text:
            continue
        if heading and heading == title and text.strip() == title.strip():
            continue
        metadata = {**base_metadata, "section": heading or title}
        documents.append(Document(page_content=text, metadata=metadata))

    return documents


def _build_metadata(
    path: Path,
    title: str,
    section: str | None,
    page: int | None,
    metadata_override: dict[str, Any] | None = None,
) -> dict[str, object]:
    metadata = {
        "doc_id": None,
        "source": _relative_source(path),
        "url": None,
        "title": title,
        "section": section,
        "page": page,
        "doc_type": infer_doc_type(path),
        "year": infer_year(path),
        "program": None,
    }
    if metadata_override:
        metadata.update(
            {key: value for key, value in metadata_override.items() if value is not None}
        )
    return metadata


def infer_doc_type(path: Path) -> str:
    haystack = path.stem.lower().replace("_", "-")
    for doc_type, markers in DOC_TYPE_PATTERNS.items():
        if any(marker in haystack for marker in markers):
            return doc_type
    return "general_policy"


def infer_year(path: Path) -> int | None:
    match = YEAR_PATTERN.search(path.stem)
    return int(match.group(1)) if match else None


def _split_markdown_sections(
    text: str,
    fallback_title: str,
) -> tuple[str, list[tuple[str | None, str]]]:
    lines = text.splitlines()
    sections: list[tuple[str | None, str]] = []
    title = fallback_title.replace("_", " ").strip()

    current_heading: str | None = None
    current_lines: list[str] = []

    for line in lines:
        match = MARKDOWN_HEADING_PATTERN.match(line)
        if match:
            heading = match.group(2).strip()
            if match.group(1) == "#" and title == fallback_title.replace("_", " ").strip():
                title = heading
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines).strip()))
            current_heading = heading
            current_lines = [line]
        else:
            current_lines.append(line)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines).strip()))

    if not sections:
        return title, [(title, text)]

    return title, sections


def _markdown_to_text(markdown_text: str) -> str:
    html = markdown(markdown_text)
    soup = BeautifulSoup(html, "html.parser")
    return _clean_text(soup.get_text("\n"))


def _relative_source(path: Path) -> str:
    try:
        return path.relative_to(Path.cwd()).as_posix()
    except ValueError:
        return path.as_posix()


def _override_for_path(
    path: Path,
    metadata_overrides: dict[str, dict[str, Any]] | None,
) -> dict[str, Any] | None:
    if not metadata_overrides:
        return None
    return metadata_overrides.get(_relative_source(path))


def _select_html_content_root(soup: BeautifulSoup) -> BeautifulSoup:
    selectors = [
        "main article",
        "main",
        "article",
        "[role='main']",
        "#main-content",
        "#content",
        ".main-content",
        ".page-content",
        ".layout-content",
    ]
    for selector in selectors:
        node = soup.select_one(selector)
        if node is not None:
            return node
    return soup


def _strip_html_noise(root: BeautifulSoup) -> None:
    selectors = [
        "script",
        "style",
        "noscript",
        "svg",
        "img",
        "button",
        "form",
        "nav",
        "header",
        "footer",
        "aside",
        ".breadcrumb",
        ".breadcrumbs",
        ".search-form",
        ".menu",
        ".site-header",
        ".site-footer",
        ".sidebar",
        ".share",
        ".social",
    ]
    for selector in selectors:
        for node in root.select(selector):
            node.decompose()


def _split_html_sections(
    root: BeautifulSoup,
    fallback_title: str,
) -> list[tuple[str | None, str]]:
    sections: list[tuple[str | None, str]] = []
    current_heading: str | None = fallback_title
    current_lines: list[str] = []

    for element in root.find_all(["h1", "h2", "h3", "p", "li", "tr", "dd", "dt"]):
        text = _clean_text(element.get_text(" ", strip=True))
        if not text or _is_noise_line(text):
            continue

        if element.name in {"h1", "h2", "h3"}:
            if text.lower() in NOISE_HEADINGS:
                continue
            if current_lines:
                sections.append((current_heading, "\n".join(current_lines)))
            current_heading = text
            current_lines = [text]
            continue

        current_lines.append(text)

    if current_lines:
        sections.append((current_heading, "\n".join(current_lines)))

    return sections


def _is_noise_line(text: str) -> bool:
    lowered = text.lower()
    if lowered in NOISE_HEADINGS:
        return True
    return any(lowered.startswith(prefix) for prefix in NOISE_LINE_PREFIXES)


def _clean_text(value: str) -> str:
    value = value.replace("\u00a0", " ")
    value = re.sub(r"\r\n?", "\n", value)
    value = re.sub(r"[ \t]+", " ", value)
    value = re.sub(r"\n{3,}", "\n\n", value)
    return value.strip()
