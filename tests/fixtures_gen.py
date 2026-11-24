"""Build small binary fixtures in memory (PDF, PPTX, EPUB, DOCX) so the test
suite needs no checked-in binaries and no extra dependencies."""

from __future__ import annotations

import io
import zipfile
from typing import List, Tuple


def make_pdf(pages: List[str]) -> bytes:
    """A minimal valid PDF with one line of Helvetica text per page."""
    objs: List[bytes] = []
    n_pages = len(pages)
    # 1: catalog, 2: pages, 3: font, then (page, content) pairs
    kids = " ".join("{} 0 R".format(4 + 2 * i) for i in range(n_pages))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append("<< /Type /Pages /Kids [{}] /Count {} >>".format(kids, n_pages).encode())
    objs.append(b"<< /Type /Font /Subtype /Type1 /BaseFont /Helvetica >>")
    for i, text in enumerate(pages):
        safe = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
        stream = "BT /F1 12 Tf 72 720 Td ({}) Tj ET".format(safe).encode()
        objs.append("<< /Type /Page /Parent 2 0 R /MediaBox [0 0 612 792] "
                    "/Resources << /Font << /F1 3 0 R >> >> /Contents {} 0 R >>".format(5 + 2 * i).encode())
        objs.append(b"<< /Length " + str(len(stream)).encode() + b" >>\nstream\n" + stream + b"\nendstream")
    out = io.BytesIO()
    out.write(b"%PDF-1.4\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(out.tell())
        out.write("{} 0 obj\n".format(i).encode() + body + b"\nendobj\n")
    xref = out.tell()
    out.write("xref\n0 {}\n0000000000 65535 f \n".format(len(objs) + 1).encode())
    for off in offsets:
        out.write("{:010d} 00000 n \n".format(off).encode())
    out.write("trailer\n<< /Size {} /Root 1 0 R >>\nstartxref\n{}\n%%EOF\n".format(
        len(objs) + 1, xref).encode())
    return out.getvalue()


_A = 'xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main"'


def _slide_xml(paras: List[str]) -> str:
    body = "".join("<a:p><a:r><a:t>{}</a:t></a:r></a:p>".format(p) for p in paras)
    return ('<?xml version="1.0"?><p:sld xmlns:p="http://schemas.openxmlformats.org/'
            'presentationml/2006/main" {}><p:cSld><p:spTree><p:sp><p:txBody>{}</p:txBody>'
            '</p:sp></p:spTree></p:cSld></p:sld>').format(_A, body)


def make_pptx(slides: List[Tuple[List[str], List[str]]]) -> bytes:
    """slides: [(paragraphs, notes)]"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("[Content_Types].xml", "<Types/>")
        for i, (paras, notes) in enumerate(slides, start=1):
            z.writestr("ppt/slides/slide{}.xml".format(i), _slide_xml(paras))
            if notes:
                z.writestr("ppt/notesSlides/notesSlide{}.xml".format(i), _slide_xml(notes + [str(i)]))
    return buf.getvalue()


def make_epub(title: str, chapters: List[Tuple[str, str]]) -> bytes:
    """chapters: [(heading, paragraph)]"""
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        z.writestr("mimetype", "application/epub+zip")
        z.writestr("META-INF/container.xml",
                   '<?xml version="1.0"?><container xmlns="urn:oasis:names:tc:opendocument:xmlns:container">'
                   '<rootfiles><rootfile full-path="OEBPS/content.opf" '
                   'media-type="application/oebps-package+xml"/></rootfiles></container>')
        items = "".join('<item id="c{0}" href="c{0}.xhtml" media-type="application/xhtml+xml"/>'.format(i)
                        for i in range(len(chapters)))
        spine = "".join('<itemref idref="c{}"/>'.format(i) for i in range(len(chapters)))
        z.writestr("OEBPS/content.opf",
                   '<?xml version="1.0"?><package xmlns="http://www.idpf.org/2007/opf" version="3.0">'
                   '<metadata xmlns:dc="http://purl.org/dc/elements/1.1/"><dc:title>{}</dc:title></metadata>'
                   '<manifest>{}</manifest><spine>{}</spine></package>'.format(title, items, spine))
        for i, (h, p) in enumerate(chapters):
            z.writestr("OEBPS/c{}.xhtml".format(i),
                       "<html><body><h1>{}</h1><p>{}</p></body></html>".format(h, p))
    return buf.getvalue()


def make_docx(blocks: List[Tuple[str, str]]) -> bytes:
    """blocks: [(style, text)] where style is 'h1', 'h2' or 'p'."""
    import docx

    d = docx.Document()
    for style, text in blocks:
        if style.startswith("h"):
            d.add_heading(text, level=int(style[1]))
        else:
            d.add_paragraph(text)
    buf = io.BytesIO()
    d.save(buf)
    return buf.getvalue()
