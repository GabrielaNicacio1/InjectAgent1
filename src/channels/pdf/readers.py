"""PDF -> text pipelines for the mock read_pdf tool. The reader is a scaffold choice.

  pymupdf    embedded-text extraction with PyMuPDF (default for the main grid)
  pypdf      embedded-text extraction with pypdf (LangChain / LlamaIndex default loader)
  pdftotext  embedded-text extraction with poppler's pdftotext
  ocr        render each page with PyMuPDF at OCR_DPI, then Tesseract (no vision model)
"""

import shutil
import subprocess

import pymupdf
import pypdf

OCR_DPI = 300
TESSERACT_ARGS = ["--psm", "3", "-l", "eng", "--dpi", str(OCR_DPI)]


def read_pymupdf(path):
    with pymupdf.open(path) as d:
        return "".join(page.get_text("text") for page in d), d.page_count


def read_pypdf(path):
    r = pypdf.PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in r.pages), len(r.pages)


def read_pdftotext(path):
    out = subprocess.run(["pdftotext", "-enc", "UTF-8", str(path), "-"],
                         capture_output=True, text=True, check=True).stdout
    pages = [p for p in out.split("\f")]
    if pages and pages[-1].strip() == "":
        pages = pages[:-1]
    return "\n".join(pages), len(pages)


def read_ocr(path):
    texts = []
    with pymupdf.open(path) as d:
        for page in d:
            png = page.get_pixmap(dpi=OCR_DPI, colorspace=pymupdf.csGRAY, alpha=False).tobytes("png")
            res = subprocess.run(["tesseract", "stdin", "stdout", *TESSERACT_ARGS],
                                 input=png, capture_output=True, check=True)
            texts.append(res.stdout.decode("utf-8"))
        n = d.page_count
    return "\n".join(texts), n


READERS = {
    "pymupdf": read_pymupdf,
    "pypdf": read_pypdf,
    "pdftotext": read_pdftotext,
    "ocr": read_ocr,
}


def read_pdf(path, reader):
    """Return (text, num_pages). Trailing whitespace is stripped; nothing else is changed."""
    text, n = READERS[reader](str(path))
    return text.rstrip(), n


def tool_versions():
    v = {"pymupdf": pymupdf.VersionBind, "pypdf": pypdf.__version__}
    if shutil.which("pdftotext"):
        v["pdftotext"] = subprocess.run(["pdftotext", "-v"], capture_output=True, text=True).stderr.splitlines()[0]
    if shutil.which("tesseract"):
        v["tesseract"] = subprocess.run(["tesseract", "--version"], capture_output=True, text=True).stdout.splitlines()[0]
    return v
