"""Contact sheets: a labelled grid of image thumbnails for comparing runs."""

import textwrap
from collections.abc import Callable
from pathlib import Path
from typing import Any

LABEL_WIDTH = 150  # left column holding the row labels
HEADER_HEIGHT = 64  # column labels, below the title
TITLE_HEIGHT = 30


def _font(size: int) -> Any:
    from PIL import ImageFont

    try:
        return ImageFont.load_default(size=size)
    except TypeError:  # Pillow without sized default fonts
        return ImageFont.load_default()


def make_sheet(
    path: Path,
    title: str,
    col_labels: list[str],
    row_labels: list[str],
    cell_path: Callable[[int, int], Path],
    thumb: int = 384,
) -> None:
    """
    Write a grid of thumbnails with a title, column labels and row labels.

    Args:
        path: Where to save the sheet (PNG); parent folders are created
        title: Text at the top left
        col_labels: One label per column (wrapped over up to 3 lines)
        row_labels: One label per row (may contain newlines)
        cell_path: cell_path(row, col) -> the image file for that cell. A cell
            whose file does not exist is drawn as a grey "missing" box.
        thumb: Thumbnail size in pixels (images are squashed to a square)

    Raises:
        ImportError: If Pillow is not installed (pip install prompt-runner[sheets])
    """
    try:
        from PIL import Image, ImageDraw
    except ImportError as e:
        raise ImportError(
            "Contact sheets need Pillow: pip install 'prompt-runner[sheets]'"
        ) from e

    font, small = _font(18), _font(14)
    width = LABEL_WIDTH + thumb * len(col_labels)
    height = TITLE_HEIGHT + HEADER_HEIGHT + thumb * len(row_labels)
    sheet = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(sheet)
    draw.text((6, 5), title, fill="black", font=font)

    for c, label in enumerate(col_labels):
        for k, line in enumerate(textwrap.wrap(label, 30)[:3]):
            x, y = LABEL_WIDTH + c * thumb + 6, TITLE_HEIGHT + 3 + k * 18
            draw.text((x, y), line, fill="black", font=small)

    for r, label in enumerate(row_labels):
        top = TITLE_HEIGHT + HEADER_HEIGHT + r * thumb
        for k, line in enumerate(label.split("\n")):
            draw.text((6, top + 6 + k * 18), line, fill="black", font=small)
        for c in range(len(col_labels)):
            x = LABEL_WIDTH + c * thumb
            image_file = cell_path(r, c)
            if image_file.exists():
                with Image.open(image_file) as img:
                    cell = img.convert("RGB").resize(
                        (thumb, thumb), Image.Resampling.LANCZOS
                    )
                sheet.paste(cell, (x, top))
            else:
                draw.rectangle(
                    [x, top, x + thumb - 1, top + thumb - 1], fill=(225, 225, 225)
                )
                draw.text(
                    (x + 10, top + thumb // 2), "missing", fill="black", font=font
                )

    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)
