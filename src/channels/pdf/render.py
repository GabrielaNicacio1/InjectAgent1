"""Render base documents to PDF, with or without a hidden payload.

Hiding methods (CH5):
  clean      no payload (false-positive control)
  visible    payload as a normal 11pt black paragraph (positive control)
  white      payload in 11pt white text on the white page (Gabriela's method 1)
  tiny_1pt   payload in 1pt black text (Gabriela's method 2)
  small_4pt  payload in 4pt black text: hidden-ish, but OCR at 300 dpi can read it,
             so it is the only hidden condition where both readers ingest the text

The payload is drawn between two body paragraphs and takes the vertical space
its own font size needs, so the layout is what a real attacker would produce.
Output is byte-for-byte reproducible (reportlab ``invariant=1``).
"""

import pymupdf
import numpy as np
from reportlab.lib.colors import black, white
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import simpleSplit
from reportlab.pdfgen import canvas

from src.channels.pdf.documents import INSERT_AFTER_PARAGRAPH

PAGE_W, PAGE_H = letter
MARGIN = 72
TEXT_W = PAGE_W - 2 * MARGIN
BODY_FONT, BODY_SIZE = "Helvetica", 11
TITLE_FONT, TITLE_SIZE = "Helvetica-Bold", 16
LEADING = 1.35  # line height as a multiple of font size
PARA_GAP = 8

HIDING_METHODS = {
    "clean": None,
    "visible": {"size": 11, "color": black},
    "white": {"size": 11, "color": white},
    "tiny_1pt": {"size": 1, "color": black},
    "small_4pt": {"size": 4, "color": black},
}

VISIBILITY_DPI = 150


def _draw_lines(c, lines, font, size, y):
    c.setFont(font, size)
    for line in lines:
        c.drawString(MARGIN, y - size, line)
        y -= size * LEADING
    return y


def render_pdf(path, doc, payload=None, hiding=None, draw_payload=True):
    """Write a one-page PDF. ``draw_payload=False`` keeps the payload's space but
    leaves it blank; that twin is the baseline for the visibility measurement."""
    c = canvas.Canvas(str(path), pagesize=letter, invariant=1)
    c.setTitle(doc["title"])
    y = PAGE_H - MARGIN
    y = _draw_lines(c, simpleSplit(doc["title"], TITLE_FONT, TITLE_SIZE, TEXT_W), TITLE_FONT, TITLE_SIZE, y)
    y -= PARA_GAP * 2
    for i, para in enumerate(doc["paragraphs"]):
        c.setFillColor(black)
        y = _draw_lines(c, simpleSplit(para, BODY_FONT, BODY_SIZE, TEXT_W), BODY_FONT, BODY_SIZE, y)
        y -= PARA_GAP
        if i == INSERT_AFTER_PARAGRAPH and payload is not None:
            spec = HIDING_METHODS[hiding]
            lines = simpleSplit(payload, BODY_FONT, spec["size"], TEXT_W)
            if draw_payload:
                c.setFillColor(spec["color"])
                y = _draw_lines(c, lines, BODY_FONT, spec["size"], y)
            else:
                y -= len(lines) * spec["size"] * LEADING
            c.setFillColor(black)
            y -= PARA_GAP
    if y < MARGIN:
        raise ValueError(f"{doc['doc_id']}: content overflows the page (y={y:.1f})")
    c.showPage()
    c.save()


def _gray_pixels(pdf_bytes):
    with pymupdf.open(stream=pdf_bytes, filetype="pdf") as d:
        pix = d[0].get_pixmap(dpi=VISIBILITY_DPI, colorspace=pymupdf.csGRAY, alpha=False)
        return np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.height, pix.width).astype(np.int16)


def visibility(pdf_path, doc, payload, hiding, tmp_path):
    """How much the payload changes what a human sees: render the PDF and its
    blank-payload twin at 150 dpi grayscale and count pixels that differ."""
    render_pdf(tmp_path, doc, payload, hiding, draw_payload=False)
    a = _gray_pixels(open(pdf_path, "rb").read())
    b = _gray_pixels(open(tmp_path, "rb").read())
    diff = np.abs(a - b)
    changed = int((diff > 0).sum())
    return {
        "Visible Pixels": changed,
        "Visible Fraction": round(changed / diff.size, 6),
        "Max Pixel Contrast": int(diff.max()),  # 0..255
    }
