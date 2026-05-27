"""
export_report_pdf.py - Generate a clean PDF from an HTML backtest report.

Uses Playwright's headless Chromium to bypass the browser print dialog and
produce a PDF without browser-generated headers, footers, or page numbers.

Usage:
    python scripts/export_report_pdf.py <report.html>
    python scripts/export_report_pdf.py <report.html> --out <output.pdf>
    python scripts/export_report_pdf.py --latest
    python scripts/export_report_pdf.py --latest --out report.pdf

Requires (one-time setup):
    pip install playwright
    python -m playwright install chromium

Notes:
    - Honours the report's @page CSS (A4 portrait, 12mm margins).
    - Waits for charts to render before printing (default 1500ms after networkidle).
    - print_background=True preserves backgrounds, badges, and pill colours.
    - display_header_footer=False removes URL and page numbers.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_DIR = Path(__file__).resolve().parent.parent
REPORTS_DIR = PROJECT_DIR / "backtest" / "reports"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export an HTML backtest report to a clean PDF using Playwright.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "html_path", nargs="?", default=None,
        help="Path to the HTML report. Omit and pass --latest to use the most recent.",
    )
    parser.add_argument(
        "--out", type=str, default=None,
        help="Output PDF path. Defaults to <html_path>.pdf in the same directory.",
    )
    parser.add_argument(
        "--latest", action="store_true",
        help="Export the most recent report in backtest/reports/.",
    )
    parser.add_argument(
        "--render-wait-ms", type=int, default=1500, dest="render_wait_ms",
        help="Milliseconds to wait for charts to render after networkidle (default: 1500).",
    )
    parser.add_argument(
        "--landscape", action="store_true",
        help="Force landscape orientation (default uses the @page CSS rule).",
    )
    return parser.parse_args()


def _resolve_html(args: argparse.Namespace) -> Path:
    if args.latest:
        candidates = sorted(
            REPORTS_DIR.glob("report_*.html"),
            key=lambda p: p.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            sys.stderr.write(f"No reports found in {REPORTS_DIR}\n")
            sys.exit(1)
        return candidates[0]
    if not args.html_path:
        sys.stderr.write("Provide an html_path positional argument or pass --latest.\n")
        sys.exit(2)
    p = Path(args.html_path).resolve()
    if not p.exists():
        sys.stderr.write(f"Not found: {p}\n")
        sys.exit(1)
    return p


def _resolve_out(args: argparse.Namespace, html: Path) -> Path:
    if args.out:
        return Path(args.out).resolve()
    return html.with_suffix(".pdf")


def main() -> None:
    args = _parse_args()
    html = _resolve_html(args)
    out = _resolve_out(args, html)

    try:
        from playwright.sync_api import sync_playwright
    except ImportError:
        sys.stderr.write(
            "Playwright is not installed.\n"
            "  Install:  pip install playwright\n"
            "  Browser:  python -m playwright install chromium\n"
        )
        sys.exit(2)

    print(f"[INFO]  Source : {html}")
    print(f"[INFO]  Output : {out}")
    print(f"[INFO]  Wait   : {args.render_wait_ms} ms after networkidle")

    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        context = browser.new_context(viewport={"width": 1400, "height": 1800})
        page = context.new_page()
        page.goto(html.as_uri())
        page.wait_for_load_state("networkidle")
        page.wait_for_timeout(args.render_wait_ms)
        page.emulate_media(media="print")

        pdf_kwargs = {
            "path":                 str(out),
            "print_background":     True,
            "display_header_footer": False,
            "prefer_css_page_size":  True,
            "margin": {"top": "12mm", "bottom": "12mm", "left": "12mm", "right": "12mm"},
        }
        if args.landscape:
            pdf_kwargs["landscape"] = True
            pdf_kwargs["format"] = "A4"
            pdf_kwargs["prefer_css_page_size"] = False

        page.pdf(**pdf_kwargs)
        browser.close()

    size_kb = out.stat().st_size / 1024
    print(f"[DONE]  Wrote {size_kb:,.1f} KB to {out}")


if __name__ == "__main__":
    main()
