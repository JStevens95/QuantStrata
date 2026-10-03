"""Render the rade_xl interface mockups to PNG with headless Chromium.

Run with the project virtualenv, which carries Playwright:

    .venv/bin/python docs/mockups/rade_xl/shoot.py

Each page is captured full-height at 2x so the type stays crisp when the
image is scaled down for a document or a slide.
"""

from __future__ import annotations

import pathlib

from playwright.sync_api import sync_playwright

HERE = pathlib.Path(__file__).parent
SHOTS = HERE / "shots"
WIDTH = 1512  # a 16-inch MacBook Pro browser window, rounded
SCALE = 2

PAGES = [
    ("model_lab.html", "01_model_lab.png"),
    ("run_report.html", "02_run_report.png"),
    ("console.html", "03_console.png"),
]


def main() -> None:
    SHOTS.mkdir(exist_ok=True)
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(
            viewport={"width": WIDTH, "height": 950},
            device_scale_factor=SCALE,
        )
        for source, target in PAGES:
            path = HERE / source
            if not path.exists():
                print(f"skip {source} (not written yet)")
                continue
            page.goto(path.as_uri())
            # Let web fonts and SVG layout settle before the shutter.
            page.wait_for_timeout(250)
            page.screenshot(path=SHOTS / target, full_page=True)
            print(f"wrote shots/{target}")
        browser.close()


if __name__ == "__main__":
    main()
