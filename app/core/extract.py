"""Turn an uploaded file into plain text.

PDF and DOCX support are optional at runtime: if the library is missing we raise
a clear, actionable error instead of crashing with an ImportError traceback.
"""

from __future__ import annotations

import bisect
import csv
import io
import json
import re
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple



class ExtractionError(RuntimeError):
    """Raised when a document cannot be converted to text."""


def _extract_text(data: bytes) -> str:
    for encoding in ("utf-8", "utf-16", "latin-1"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    raise ExtractionError("Could not decode file as text in utf-8, utf-16, or latin-1.")


def _pdf_pages(data: bytes):
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ExtractionError(
            "PDF support requires 'pypdf'. Install it with: pip install pypdf"
        ) from exc
    try:
        reader = PdfReader(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open PDF: {exc}") from exc
    pages = []
    for index, page in enumerate(reader.pages, start=1):
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        pages.append((index, text))
    return pages


def _extract_pdf_doc(data: bytes) -> "Extracted":
    pages = [(n, t) for n, t in _pdf_pages(data) if t.strip()]
    if not pages:
        # Scanned PDF: try OCR if the optional stack is installed.
        ocr = _ocr_pdf(data)
        if ocr is not None:
            return ocr
        raise ExtractionError(
            "No selectable text found in this PDF. It is probably a scan; "
            "install the OCR extras (pytesseract, pdf2image, Tesseract, Poppler) "
            "or upload a text-based version."
        )
    b = _Builder()
    for n, t in pages:
        b.mark_page(n)
        b.add(t.strip())
    return b.build()


def _extract_pdf(data: bytes) -> str:
    return _extract_pdf_doc(data).text


def _extract_docx(data: bytes) -> str:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover - depends on install extras
        raise ExtractionError(
            "DOCX support requires 'python-docx'. Install it with: "
            "pip install python-docx"
        ) from exc

    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc

    parts = [p.text for p in document.paragraphs if p.text.strip()]
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                parts.append(" | ".join(cells))

    if not parts:
        raise ExtractionError("The DOCX file contains no readable text.")
    return "\n".join(parts)


def _extract_csv(data: bytes) -> str:
    text = _extract_text(data)
    reader = csv.reader(io.StringIO(text))
    rows = list(reader)
    if not rows:
        raise ExtractionError("The CSV file is empty.")

    header = rows[0]
    lines = []
    for row in rows[1:]:
        pairs = [
            "{}: {}".format(header[i] if i < len(header) else "col{}".format(i), value)
            for i, value in enumerate(row)
            if value.strip()
        ]
        if pairs:
            lines.append("; ".join(pairs))
    # Keep the header so the model knows the column vocabulary.
    return "Columns: {}\n\n{}".format(", ".join(header), "\n".join(lines))


def _extract_json(data: bytes) -> str:
    text = _extract_text(data)
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ExtractionError("Invalid JSON: {}".format(exc)) from exc
    return json.dumps(parsed, indent=2, ensure_ascii=False)


# --------------------------------------------------------------------------
# Structured extraction (M2)
# --------------------------------------------------------------------------
#
# ``extract_doc`` returns the text plus position markers measured in
# characters of that text: where each PDF page / PPTX slide starts, where each
# transcript segment sits in time, and where each heading is. The chunker cuts
# spans from the same text, so a chunk's page, slide, timestamp and section
# are looked up from its start offset.


@dataclass
class Segment:
    start: int
    end: int
    t_start_ms: int
    t_end_ms: int


@dataclass
class Extracted:
    text: str
    pages: List[Tuple[int, int]] = field(default_factory=list)    # (offset, page_no)
    slides: List[Tuple[int, int]] = field(default_factory=list)   # (offset, slide_no)
    segments: List[Segment] = field(default_factory=list)
    headings: List[Tuple[int, int, str]] = field(default_factory=list)  # (offset, level, title)
    title: str = ""

    @staticmethod
    def _last_le(marks, pos):
        i = bisect.bisect_right([m[0] for m in marks], pos) - 1
        return marks[i] if i >= 0 else None

    def page_at(self, pos: int) -> Optional[int]:
        m = self._last_le(self.pages, pos)
        return m[1] if m else None

    def slide_at(self, pos: int) -> Optional[int]:
        m = self._last_le(self.slides, pos)
        return m[1] if m else None

    def time_span(self, start: int, end: int) -> Tuple[Optional[int], Optional[int]]:
        hit = [s for s in self.segments if s.end > start and s.start < end]
        if not hit:
            return None, None
        return min(s.t_start_ms for s in hit), max(s.t_end_ms for s in hit)

    def section_at(self, pos: int) -> Optional[str]:
        path: List[Tuple[int, str]] = []
        for off, level, title in self.headings:
            if off > pos:
                break
            while path and path[-1][0] >= level:
                path.pop()
            path.append((level, title))
        return " > ".join(t for _, t in path) if path else None


class _Builder:
    """Accumulates text blocks while recording marker offsets."""

    def __init__(self):
        self.parts: List[str] = []
        self.len = 0
        self.doc = Extracted(text="")

    def _sep(self):
        if self.parts:
            self.parts.append("\n\n")
            self.len += 2

    def mark_page(self, n):
        self.doc.pages.append((self.len + (2 if self.parts else 0), n))

    def mark_slide(self, n):
        self.doc.slides.append((self.len + (2 if self.parts else 0), n))

    def heading(self, level: int, title: str):
        self._sep()
        self.doc.headings.append((self.len, level, title))
        self.parts.append(title)
        self.len += len(title)

    def add(self, block: str, t_ms: Optional[Tuple[int, int]] = None):
        block = block.strip()
        if not block:
            return
        self._sep()
        start = self.len
        self.parts.append(block)
        self.len += len(block)
        if t_ms is not None:
            self.doc.segments.append(Segment(start, self.len, t_ms[0], t_ms[1]))

    def build(self) -> Extracted:
        self.doc.text = "".join(self.parts)
        return self.doc


_MD_HEADING = re.compile(r"^(#{1,6})[ \t]+(.+?)[ \t#]*$", re.M)


def _extract_markdown_doc(data: bytes) -> Extracted:
    text = _extract_text(data)
    doc = Extracted(text=text)
    doc.headings = [(m.start(), len(m.group(1)), m.group(2).strip())
                    for m in _MD_HEADING.finditer(text)]
    return doc


def _extract_docx_doc(data: bytes) -> Extracted:
    try:
        import docx  # python-docx
    except ImportError as exc:  # pragma: no cover
        raise ExtractionError(
            "DOCX support requires 'python-docx'. Install it with: pip install python-docx"
        ) from exc
    try:
        document = docx.Document(io.BytesIO(data))
    except Exception as exc:
        raise ExtractionError(f"Could not open DOCX: {exc}") from exc
    b = _Builder()
    for p in document.paragraphs:
        if not p.text.strip():
            continue
        style = (p.style.name if p.style is not None else "") or ""
        m = re.match(r"Heading (\d)", style)
        if m or style == "Title":
            b.heading(int(m.group(1)) if m else 1, p.text.strip())
        else:
            b.add(p.text)
    for table in document.tables:
        for row in table.rows:
            cells = [c.text.strip() for c in row.cells if c.text.strip()]
            if cells:
                b.add(" | ".join(cells))
    doc = b.build()
    if not doc.text.strip():
        raise ExtractionError("The DOCX file contains no readable text.")
    return doc


# --- PPTX (stdlib: a .pptx is a zip of XML) ---------------------------------

_A_NS = "{http://schemas.openxmlformats.org/drawingml/2006/main}"


def _xml_paragraphs(xml_bytes: bytes) -> List[str]:
    root = ET.fromstring(xml_bytes)
    out = []
    for p in root.iter(_A_NS + "p"):
        t = "".join(x.text or "" for x in p.iter(_A_NS + "t")).strip()
        if t:
            out.append(t)
    return out


def _extract_pptx_doc(data: bytes) -> Extracted:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
    except zipfile.BadZipFile as exc:
        raise ExtractionError("Could not open PPTX: not a valid zip archive.") from exc
    slide_re = re.compile(r"^ppt/slides/slide(\d+)\.xml$")
    slides = sorted((int(m.group(1)), n) for n in z.namelist() for m in [slide_re.match(n)] if m)
    if not slides:
        raise ExtractionError("The PPTX file contains no slides.")
    b = _Builder()
    for num, name in slides:
        try:
            paras = _xml_paragraphs(z.read(name))
        except ET.ParseError:
            paras = []
        notes = []
        nname = "ppt/notesSlides/notesSlide{}.xml".format(num)
        if nname in z.namelist():
            try:
                notes = [p for p in _xml_paragraphs(z.read(nname)) if not p.isdigit()]
            except ET.ParseError:
                notes = []
        if not paras and not notes:
            continue
        b.mark_slide(num)
        title = paras[0] if paras else "Slide {}".format(num)
        b.heading(1, "Slide {}: {}".format(num, title))
        if len(paras) > 1:
            b.add("\n".join(paras[1:]))
        if notes:
            b.add("Speaker notes: " + " ".join(notes))
    doc = b.build()
    if not doc.text.strip():
        raise ExtractionError("The PPTX file contains no readable text.")
    return doc


# --- HTML / EPUB -------------------------------------------------------------


class _HTMLText(HTMLParser):
    """Visible text with headings, skipping scripts, styles and navigation."""

    SKIP = {"script", "style", "noscript", "nav", "footer", "header", "aside", "form",
            "svg", "template"}
    BLOCK = {"p", "div", "li", "tr", "section", "article", "blockquote", "pre",
             "br", "dd", "dt", "table", "ul", "ol", "main"}

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.b = _Builder()
        self.buf: List[str] = []
        self.skip = 0
        self.heading_level = 0
        self.title = ""
        self._in_title = False

    def _flush(self):
        t = re.sub(r"\s+", " ", "".join(self.buf)).strip()
        self.buf = []
        if not t:
            return
        if self.heading_level:
            self.b.heading(self.heading_level, t)
        else:
            self.b.add(t)

    def handle_starttag(self, tag, attrs):
        if tag in self.SKIP:
            self.skip += 1
        elif tag == "title":
            self._in_title = True
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self.heading_level = int(tag[1])
        elif tag in self.BLOCK:
            self._flush()

    def handle_endtag(self, tag):
        if tag in self.SKIP:
            self.skip = max(0, self.skip - 1)
        elif tag == "title":
            self._in_title = False
        elif re.fullmatch(r"h[1-6]", tag):
            self._flush()
            self.heading_level = 0
        elif tag in self.BLOCK:
            self._flush()

    def handle_data(self, data):
        if self._in_title:
            self.title += data
        elif not self.skip:
            self.buf.append(data)

    def result(self) -> Extracted:
        self._flush()
        doc = self.b.build()
        doc.title = " ".join(self.title.split())
        return doc


def html_to_doc(html: str) -> Extracted:
    p = _HTMLText()
    p.feed(html)
    p.close()
    return p.result()


def _merge(docs: List[Extracted]) -> Extracted:
    b = _Builder()
    out = b.doc
    for d in docs:
        if not d.text.strip():
            continue
        base = b.len + (2 if b.parts else 0)
        b.add(d.text)
        out.headings.extend((base + o, lv, t) for o, lv, t in d.headings)
    return b.build()


def _extract_epub_doc(data: bytes) -> Extracted:
    try:
        z = zipfile.ZipFile(io.BytesIO(data))
        container = ET.fromstring(z.read("META-INF/container.xml"))
        rootfile = next(e for e in container.iter() if e.tag.endswith("rootfile"))
        opf_path = rootfile.attrib["full-path"]
        opf = ET.fromstring(z.read(opf_path))
    except (zipfile.BadZipFile, KeyError, StopIteration, ET.ParseError) as exc:
        raise ExtractionError("Could not open EPUB: {}".format(exc)) from exc
    base = opf_path.rsplit("/", 1)[0] + "/" if "/" in opf_path else ""
    manifest = {e.attrib.get("id"): e.attrib.get("href") for e in opf.iter()
                if e.tag.endswith("item")}
    spine = [e.attrib.get("idref") for e in opf.iter() if e.tag.endswith("itemref")]
    title_el = next((e for e in opf.iter() if e.tag.endswith("}title")), None)
    docs = []
    for idref in spine:
        href = manifest.get(idref)
        if not href:
            continue
        try:
            html = z.read(base + href).decode("utf-8", errors="replace")
        except KeyError:
            continue
        docs.append(html_to_doc(html))
    doc = _merge(docs)
    if not doc.text.strip():
        raise ExtractionError("The EPUB file contains no readable text.")
    doc.title = (title_el.text or "").strip() if title_el is not None else ""
    return doc


def _extract_html_doc(data: bytes) -> Extracted:
    doc = html_to_doc(_extract_text(data))
    if not doc.text.strip():
        raise ExtractionError("No readable text found in this HTML page.")
    return doc


# --- OCR and speech (optional, lazily imported) ------------------------------


def _ocr_image_bytes(data: bytes) -> str:
    try:
        import pytesseract
        from PIL import Image
    except ImportError as exc:
        raise ExtractionError(
            "Image text recognition requires the OCR extras: pip install pytesseract "
            "Pillow, plus the Tesseract binary (brew install tesseract / apt install "
            "tesseract-ocr)."
        ) from exc
    try:
        return pytesseract.image_to_string(Image.open(io.BytesIO(data)))
    except Exception as exc:
        raise ExtractionError("OCR failed: {}".format(exc)) from exc


def _extract_image_doc(data: bytes) -> Extracted:
    text = _ocr_image_bytes(data)
    if not text.strip():
        raise ExtractionError("No text was recognised in this image.")
    b = _Builder()
    for para in re.split(r"\n\s*\n", text):
        b.add(para)
    return b.build()


def _ocr_pdf(data: bytes) -> Optional[Extracted]:
    try:
        import pytesseract  # noqa: F401
        from pdf2image import convert_from_bytes
    except ImportError:
        return None
    try:
        images = convert_from_bytes(data)
    except Exception:
        return None
    b = _Builder()
    for n, img in enumerate(images, start=1):
        buf = io.BytesIO()
        img.save(buf, format="PNG")
        text = _ocr_image_bytes(buf.getvalue())
        if text.strip():
            b.mark_page(n)
            b.add(text)
    doc = b.build()
    return doc if doc.text.strip() else None


# Speech-to-text is pluggable so tests (and alternative backends) can supply
# segments without installing a model: set_transcriber(fn) where
# fn(data: bytes, filename: str) -> List[(t_start_ms, t_end_ms, text)].
_transcriber: Optional[Callable[[bytes, str], List[Tuple[int, int, str]]]] = None


def set_transcriber(fn) -> None:
    global _transcriber
    _transcriber = fn


def _default_transcriber(data: bytes, filename: str):
    import os
    import tempfile

    provider = os.getenv("STT_PROVIDER", "faster-whisper")
    if provider == "openai":
        from app.core.stt import openai_transcribe

        return openai_transcribe(data, filename)
    try:
        from faster_whisper import WhisperModel
    except ImportError as exc:
        raise ExtractionError(
            "Audio transcription requires 'faster-whisper' (pip install faster-whisper) "
            "or STT_PROVIDER=openai with STT_BASE_URL/STT_API_KEY."
        ) from exc
    model = WhisperModel(os.getenv("WHISPER_MODEL", "base"), compute_type="int8")
    suffix = Path(filename).suffix or ".wav"
    with tempfile.NamedTemporaryFile(suffix=suffix) as fh:
        fh.write(data)
        fh.flush()
        segments, _info = model.transcribe(fh.name)
        return [(int(s.start * 1000), int(s.end * 1000), s.text) for s in segments]


def segments_to_doc(segments: List[Tuple[int, int, str]]) -> Extracted:
    """Group transcript segments into ~paragraphs, keeping per-segment timing."""
    b = _Builder()
    for t0, t1, text in segments:
        b.add(" ".join(text.split()), t_ms=(int(t0), int(t1)))
    return b.build()


def _extract_audio_doc(data: bytes, filename: str = "audio.wav") -> Extracted:
    fn = _transcriber or _default_transcriber
    segs = fn(data, filename)
    doc = segments_to_doc(segs)
    if not doc.text.strip():
        raise ExtractionError("No speech was recognised in this audio file.")
    return doc


def _plain_doc(fn: Callable[[bytes], str]) -> Callable[[bytes], Extracted]:
    return lambda data: Extracted(text=fn(data))


IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp", ".webp", ".gif"}
AUDIO_EXTENSIONS = {".mp3", ".wav", ".m4a", ".ogg", ".flac", ".webm", ".mp4"}

_DOC_EXTRACTORS: Dict[str, Callable[[bytes], Extracted]] = {
    ".txt": _plain_doc(_extract_text),
    ".md": _extract_markdown_doc,
    ".markdown": _extract_markdown_doc,
    ".pdf": _extract_pdf_doc,
    ".docx": _extract_docx_doc,
    ".csv": _plain_doc(lambda d: _extract_csv(d)),
    ".json": _plain_doc(lambda d: _extract_json(d)),
    ".pptx": _extract_pptx_doc,
    ".epub": _extract_epub_doc,
    ".html": _extract_html_doc,
    ".htm": _extract_html_doc,
}
for _ext in IMAGE_EXTENSIONS:
    _DOC_EXTRACTORS[_ext] = _extract_image_doc

SUPPORTED_EXTENSIONS = set(_DOC_EXTRACTORS) | AUDIO_EXTENSIONS


def kind_for(filename: str) -> str:
    ext = Path(filename).suffix.lower()
    if ext in IMAGE_EXTENSIONS:
        return "image"
    if ext in AUDIO_EXTENSIONS:
        return "audio"
    return "file"


def extract_doc(filename: str, data: bytes) -> Extracted:
    """Extract text plus position markers, dispatching on the extension."""
    if not data:
        raise ExtractionError("The file is empty.")
    suffix = Path(filename).suffix.lower()
    if suffix in AUDIO_EXTENSIONS:
        doc = _extract_audio_doc(data, filename)
    else:
        extractor = _DOC_EXTRACTORS.get(suffix)
        if extractor is None:
            raise ExtractionError(
                "Unsupported file type '{}'. Supported: {}".format(
                    suffix or "(none)", ", ".join(sorted(SUPPORTED_EXTENSIONS))
                )
            )
        doc = extractor(data)
    if not doc.text.strip():
        raise ExtractionError("No readable text could be extracted from this file.")
    return doc


def extract(filename: str, data: bytes) -> str:
    """Extract plain text (legacy interface)."""
    return extract_doc(filename, data).text
