"""Step 1: pull the text and the figures out of a PDF.

Figures are found through their captions, not through the images stored in the PDF. Some charts are drawn as
vector shapes and are not stored as images at all, so a plain image extract would miss them. Logos and QR codes
have no caption, so they are left out without any special rule.
"""
import logging
import re
from dataclasses import dataclass
from pathlib import Path

import fitz

log = logging.getLogger("extract")

CAPTION = re.compile(r"^(Figure|Fig\.)\s*(\d+)\s*\.", re.IGNORECASE)


@dataclass
class TextBlock:
    doc: str
    page: int
    text: str


@dataclass
class Figure:
    figure_id: str
    doc: str
    page: int
    number: int
    caption: str
    image_path: str


def _clean(text):
    return re.sub(r"\s+", " ", text).strip()


def _text_blocks(page):
    """[(rectangle, cleaned text)] in reading order."""
    blocks = []
    for x0, y0, x1, y1, text, _, block_type in page.get_text("blocks", sort=True):
        if block_type == 0 and text.strip():
            blocks.append((fitz.Rect(x0, y0, x1, y1), _clean(text)))
    return blocks


def _gap(a, b):
    """Shortest distance between two rectangles. 0 when they touch or overlap."""
    dx = max(a.x0 - b.x1, b.x0 - a.x1, 0)
    dy = max(a.y0 - b.y1, b.y0 - a.y1, 0)
    return (dx * dx + dy * dy) ** 0.5


def _figure_parts(page, settings):
    """Everything on the page that could belong to a figure: stored images and groups of vector shapes."""
    rects = [fitz.Rect(info["bbox"]) for info in page.get_image_info()]
    rects += [fitz.Rect(rect) for rect in page.cluster_drawings()]
    page_area = page.rect.get_area()
    size = settings.min_figure_part_size
    return [r for r in rects if r.width >= size and r.height >= size and r.get_area() < 0.9 * page_area]


def _figure_box(caption_rect, parts, settings):
    """Start at the part nearest the caption, then add the parts close to it. None when nothing is near."""
    # a figure sits straight above or below its caption
    nearby = [p for p in parts if min(p.x1, caption_rect.x1) > max(p.x0, caption_rect.x0)]
    nearby.sort(key=lambda p: _gap(p, caption_rect))
    if not nearby or _gap(nearby[0], caption_rect) > settings.figure_caption_gap:
        return None
    box = fitz.Rect(nearby[0])
    above = box.y0 < caption_rect.y0  # which side of the caption the figure is on
    changed = True
    while changed:
        changed = False
        for part in nearby:
            on_same_side = (part.y0 < caption_rect.y0) == above
            if on_same_side and not box.contains(part) and _gap(box, part) <= settings.figure_merge_gap:
                box |= part
                changed = True
    return box


def _save_figure(page, box, doc_name, number, settings):
    settings.figure_dir.mkdir(parents=True, exist_ok=True)
    path = settings.figure_dir / f"{doc_name}_fig{number}.png"
    page.get_pixmap(dpi=settings.figure_dpi, clip=box + (-2, -2, 2, 2)).save(path)
    return path


def extract_pdf(pdf_path, settings):
    """Returns (text blocks, figures) for one PDF."""
    pdf_path = Path(pdf_path)
    doc_name = pdf_path.stem
    try:
        pdf = fitz.open(pdf_path)
    except Exception as exc:
        raise RuntimeError(f"Could not open {pdf_path.name}: {exc}") from exc

    texts, figures = [], []
    stopped = False
    with pdf:
        for page in pdf:
            page_no = page.number + 1
            blocks = _text_blocks(page)
            parts = _figure_parts(page, settings)

            boxes, captions = [], []
            for rect, text in blocks:
                match = CAPTION.match(text)
                if not match:
                    continue
                captions.append(rect)
                box = _figure_box(rect, parts, settings)
                if box is None:
                    log.warning("%s page %d: no figure found for the caption %r", doc_name, page_no, text[:50])
                    continue
                number = int(match.group(2))
                path = _save_figure(page, box, doc_name, number, settings)
                boxes.append(box)
                figures.append(Figure(f"{doc_name}_fig{number}", doc_name, page_no, number, text, str(path)))

            for rect, text in blocks:
                if text.lower() in settings.stop_headings:
                    stopped = True
                if stopped:
                    break
                inside_figure = any(box.intersects(rect) and (box & rect).get_area() > 0.5 * rect.get_area() for box in boxes)
                if rect in captions or inside_figure or len(text) < settings.min_block_chars:
                    continue
                texts.append(TextBlock(doc_name, page_no, text))

    log.info("%s: %d text blocks, %d figures", doc_name, len(texts), len(figures))
    return texts, figures
