"""Unit tests for src/prompt_runner/sheets.py"""

from PIL import Image

from prompt_runner.sheets import HEADER_HEIGHT, LABEL_WIDTH, TITLE_HEIGHT, make_sheet


def test_grid_size_and_missing_cell(tmp_path):
    red = tmp_path / "red.png"
    Image.new("RGB", (64, 32), "red").save(red)
    sheet = tmp_path / "out" / "sheet.png"

    make_sheet(
        sheet,
        "title",
        ["model a", "model b"],
        ["p1\nseed 1", "p2\nseed 1"],
        lambda r, c: red if (r, c) != (1, 1) else tmp_path / "missing.png",
        thumb=40,
    )

    with Image.open(sheet) as img:
        assert img.size == (LABEL_WIDTH + 2 * 40, TITLE_HEIGHT + HEADER_HEIGHT + 2 * 40)
        top = TITLE_HEIGHT + HEADER_HEIGHT
        assert img.getpixel((LABEL_WIDTH + 20, top + 20)) == (
            255,
            0,
            0,
        )  # an image cell
        assert img.getpixel((LABEL_WIDTH + 40 + 5, top + 40 + 5)) == (
            225,
            225,
            225,
        )  # the missing cell
