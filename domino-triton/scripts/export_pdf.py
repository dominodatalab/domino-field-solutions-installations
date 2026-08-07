"""
Export presentation/index.html to a multi-page PDF.

Each slide (section.slide) becomes one page. Uses Chrome headless to render
each slide individually, then merges the pages with pypdf.
"""

import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path

try:
    from pypdf import PdfWriter, PdfReader
except ImportError:
    print("pypdf not found — run: pip3 install pypdf")
    sys.exit(1)

CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"
REPO_ROOT = Path(__file__).parent.parent
HTML_SRC = REPO_ROOT / "presentation" / "index.html"
OUTPUT_PDF = REPO_ROOT / "presentation.pdf"

# Slide dimensions: 1140px wide (matches --slide-w), 16:9 ratio
SLIDE_W_PX = 1140
SLIDE_H_PX = 760   # approx 3:2, close to a presentation slide


def extract_slides(html: str) -> list[tuple[str, str]]:
    """Return list of (slide_id, full_slide_html) for each section.slide."""
    pattern = re.compile(
        r'(<section\s+class="slide"\s+id="(s\d+)".*?</section>)',
        re.DOTALL,
    )
    return [(m.group(2), m.group(1)) for m in pattern.finditer(html)]


def make_slide_html(slide_html: str, original_html: str) -> str:
    """Wrap a single slide in a minimal page that reproduces the global styles."""
    # Extract <style> block from the original
    style_match = re.search(r"<style>(.*?)</style>", original_html, re.DOTALL)
    styles = style_match.group(1) if style_match else ""

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <style>
{styles}
    /* Override body layout for single-slide export */
    body {{
      padding: 0;
      gap: 0;
      background: var(--bg);
      display: block;
    }}
    .slide {{
      width: {SLIDE_W_PX}px;
      max-width: {SLIDE_W_PX}px;
      border-radius: 0;
      border: none;
      min-height: {SLIDE_H_PX}px;
    }}
    nav {{ display: none; }}
  </style>
</head>
<body>
{slide_html}
</body>
</html>"""


def render_slide_pdf(slide_html_content: str, out_path: str) -> None:
    """Use Chrome headless to render a single slide to PDF."""
    with tempfile.NamedTemporaryFile(suffix=".html", delete=False, mode="w") as f:
        f.write(slide_html_content)
        tmp_html = f.name

    try:
        subprocess.run(
            [
                CHROME,
                "--headless=new",
                "--disable-gpu",
                "--no-sandbox",
                "--disable-software-rasterizer",
                f"--print-to-pdf={out_path}",
                f"--window-size={SLIDE_W_PX},{SLIDE_H_PX}",
                "--print-to-pdf-no-header",
                "--no-pdf-header-footer",
                f"--force-device-scale-factor=2",
                f"file://{tmp_html}",
            ],
            check=True,
            capture_output=True,
        )
    finally:
        os.unlink(tmp_html)


def main():
    html = HTML_SRC.read_text()
    slides = extract_slides(html)

    if not slides:
        print("No slides found (expected <section class=\"slide\" id=\"sN\">)")
        sys.exit(1)

    print(f"Found {len(slides)} slides: {[s[0] for s in slides]}")

    writer = PdfWriter()

    with tempfile.TemporaryDirectory() as tmp_dir:
        for i, (slide_id, slide_html) in enumerate(slides):
            print(f"  Rendering {slide_id}...", end=" ", flush=True)
            page_html = make_slide_html(slide_html, html)
            pdf_path = os.path.join(tmp_dir, f"{slide_id}.pdf")
            render_slide_pdf(page_html, pdf_path)

            reader = PdfReader(pdf_path)
            for page in reader.pages:
                writer.add_page(page)
            print("done")

    with open(OUTPUT_PDF, "wb") as f:
        writer.write(f)

    print(f"\nSaved: {OUTPUT_PDF}")


if __name__ == "__main__":
    main()
