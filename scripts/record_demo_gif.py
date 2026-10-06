"""Record assets/demo.gif: the live demo passing, blocking a broken prompt, then passing again.

Drives a running dashboard with Playwright and stitches the screenshots into a GIF with
Pillow. Every frame is the real app making real calls - nothing is mocked or edited
except the caption banner on top. Dev-only tooling, not a project dependency:

    pip install playwright pillow
    streamlit run dashboard/app.py --server.port 8599        # with AEG_API_KEY set
    python scripts/record_demo_gif.py --url http://localhost:8599 --chrome /usr/bin/google-chrome

(--chrome is optional; without it Playwright uses its own browser - `playwright install chromium`.)
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from playwright.sync_api import Locator, Page, sync_playwright

QUESTION_TYPE = "Trick questions (18)"
WIDTH, HEIGHT, BANNER = 904, 700, 52
FONT_PATHS = ("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", "/Library/Fonts/Arial Bold.ttf")


def _font(size: int) -> ImageFont.ImageFont:
    for path in FONT_PATHS:
        if Path(path).exists():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default()


def _frame(page: Page, caption: str, top: float) -> Image.Image:
    png = page.screenshot(clip={"x": 228, "y": top, "width": WIDTH, "height": HEIGHT - BANNER})
    shot = Image.open(io.BytesIO(png)).convert("RGB")
    frame = Image.new("RGB", (WIDTH, HEIGHT), "#16181D")
    frame.paste(shot, (0, BANNER))
    ImageDraw.Draw(frame).text((20, BANNER // 2), caption, font=_font(19), fill="#FFFFFF", anchor="lm")
    return frame


def _run(panel: Locator, expect: str) -> None:
    panel.get_by_role("button", name="Run this question").click()
    panel.get_by_text(expect).wait_for(timeout=120_000)
    panel.page.wait_for_timeout(1200)


def record(url: str, out: Path, chrome: str | None) -> None:
    with sync_playwright() as p:
        browser = p.chromium.launch(executable_path=chrome) if chrome else p.chromium.launch()
        page = browser.new_page(viewport={"width": 1360, "height": 2000})
        page.goto(url)
        page.get_by_role("tab", name="Try it live").wait_for(timeout=120_000)
        page.get_by_role("tab", name="Try it live").click()
        panel = page.get_by_role("tabpanel", name="Try it live")
        panel.get_by_text(QUESTION_TYPE, exact=True).click()
        page.wait_for_timeout(1500)
        top = panel.get_by_text("Question type", exact=True).bounding_box()["y"] - 12

        frames = []
        panel.get_by_text("Normal", exact=True).click()
        page.wait_for_timeout(800)
        _run(panel, "Would pass the gate")
        frames.append((_frame(page, "1 · Normal prompt: no document answers this, so it declines. Gate: pass", top), 3600))

        panel.get_by_text("Broken on purpose", exact=True).click()
        page.wait_for_timeout(1200)
        frames.append((_frame(page, "2 · A one-line prompt change: “never say you don't know”", top), 2200))

        _run(panel, "Would block the merge")
        frames.append((_frame(page, "3 · The model makes up an answer. Gate: merge blocked", top), 4200))

        panel.get_by_text("Normal", exact=True).click()
        page.wait_for_timeout(800)
        _run(panel, "Would pass the gate")
        frames.append((_frame(page, "4 · Revert the prompt: green again", top), 3600))
        browser.close()

    images = [frame.quantize(colors=128, dither=Image.Dither.NONE) for frame, _ in frames]
    images[0].save(
        out, save_all=True, append_images=images[1:], duration=[ms for _, ms in frames], loop=0, optimize=True
    )
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB, {len(images)} frames)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", default="http://localhost:8501")
    parser.add_argument("--out", type=Path, default=Path("assets/demo.gif"))
    parser.add_argument("--chrome", help="path to a Chrome/Chromium binary (optional)")
    args = parser.parse_args()
    record(args.url, args.out, args.chrome)
