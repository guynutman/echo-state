"""Shared rendering helpers for the HTML outputs.

Both `report.py` (general reader) and `paper.py` (sprint submission) render the
same data into different documents. Everything they have in common lives here
so neither file has to restate it.
"""

from __future__ import annotations

import html
from pathlib import Path

# Colour encodes the experimental arm throughout, in every chart and table, so
# the comparison the study rests on is legible without reading a legend twice.
CONCEPT_COLOR = "#0E7C86"
RANDOM_COLOR = "#B4306E"

TEMPLATE_DIR = Path(__file__).parent / "templates"


def load_template(name: str) -> str:
    """Read a template from `echostate/templates`."""
    return (TEMPLATE_DIR / name).read_text(encoding="utf-8")


def render_standalone(page_name: str, values: dict[str, object]) -> str:
    """A complete HTML document, for PDF rendering rather than the artifact host.

    The artifact host supplies its own `<!doctype>` and `<head>`, so `render`
    deliberately emits a fragment. A PDF renderer supplies nothing, so this
    wraps the same content and appends the print stylesheet.
    """
    body = load_template(f"{page_name}.body.html")
    css = load_template(f"{page_name}.css")
    print_css = load_template("print.css")
    title = values["title"]
    return (
        "<!doctype html>\n"
        '<html lang="en">\n<head>\n<meta charset="utf-8">\n'
        f"<title>{title}</title>\n"
        f"<style>\n{css}\n{print_css}</style>\n"
        f"</head>\n<body>\n{body.format(**values)}\n</body>\n</html>\n"
    )


def render(page_name: str, values: dict[str, object]) -> str:
    """Assemble `<title>`, stylesheet, and body into one self-contained page.

    Markup lives in `templates/*.body.html` and styling in `templates/*.css`,
    rather than inside Python string literals. Keeping them apart means the CSS
    is real CSS — no doubled braces to appease f-string syntax — and an editor
    can highlight and validate both.
    """
    body = load_template(f"{page_name}.body.html")
    css = load_template(f"{page_name}.css")
    title = values["title"]
    return f"<title>{title}</title>\n<style>\n{css}</style>\n{body.format(**values)}"


def clean(text: str) -> str:
    """Strip characters that survive tokenizer decoding but break the page.

    Steering hard enough makes a model emit bytes that are not valid UTF-8; the
    tokenizer decodes those to U+FFFD, and control characters arrive the same
    way. Both are noise inside a quoted completion.
    """
    return "".join(c for c in text if c != "�" and (c.isprintable() or c in " \t"))


def escape_completion(text: str, limit: int = 180) -> str:
    """Prepare a model completion for display: cleaned, trimmed, escaped."""
    return html.escape(clean(str(text)).strip()[:limit])


def percent(value: float, places: int = 0) -> str:
    return f"{value * 100:.{places}f}%"


def fixed(value: float, places: int = 3) -> str:
    return f"{value:.{places}f}"


def rows_to_html(rows: list[str]) -> str:
    """Join pre-rendered table rows."""
    return "".join(rows)


class LineChart:
    """A small two-series line chart, rendered as inline SVG.

    Inline SVG rather than a plotting library because the page must be
    self-contained: the artifact CSP blocks every external request, so a CDN
    chart library would fail silently.
    """

    def __init__(
        self,
        width: int = 460,
        height: int = 240,
        pad_left: int = 54,
        pad_bottom: int = 36,
        pad_top: int = 16,
        pad_right: int = 12,
        tick_places: int = 2,
        tick_font: int = 10,
        dash_second_series: bool = False,
    ) -> None:
        self.width = width
        self.height = height
        self.pad_left = pad_left
        self.pad_bottom = pad_bottom
        self.pad_top = pad_top
        self.pad_right = pad_right
        self.tick_places = tick_places
        # Set as a presentation attribute, not only via CSS: some renderers
        # (weasyprint, for the PDF) do not cascade a stylesheet into inline
        # SVG, and fall back to a default size that dwarfs the body text.
        self.tick_font = tick_font
        self.dash_second_series = dash_second_series

    def render(self, series: dict[str, list[tuple[float, float]]], label: str) -> str:
        xs = [x for points in series.values() for x, _ in points]
        ys = [y for points in series.values() for _, y in points]
        if not xs or max(ys) == 0:
            return ""

        y_max = max(ys) * 1.15
        x_min, x_max = min(xs), max(xs)
        x_span = (x_max - x_min) or 1

        def to_x(value: float) -> float:
            usable = self.width - self.pad_left - self.pad_right
            return self.pad_left + (value - x_min) / x_span * usable

        def to_y(value: float) -> float:
            usable = self.height - self.pad_bottom - self.pad_top
            return self.height - self.pad_bottom - (value / y_max) * usable

        parts = [
            f'<svg viewBox="0 0 {self.width} {self.height}" role="img" '
            f'aria-label="{html.escape(label)} by steering strength">'
        ]
        parts += self._gridlines(y_max, to_y)
        parts += self._series(series, to_x, to_y)
        parts += self._x_axis(sorted(set(xs)), to_x)
        parts.append("</svg>")
        return "".join(parts)

    def _gridlines(self, y_max: float, to_y) -> list[str]:
        parts = []
        for step in range(4):
            value = y_max * step / 3
            y = to_y(value)
            parts.append(
                f'<line x1="{self.pad_left}" y1="{y:.1f}" '
                f'x2="{self.width - self.pad_right}" y2="{y:.1f}" class="grid"/>'
            )
            parts.append(
                f'<text x="{self.pad_left - 7}" y="{y + 3.8:.1f}" class="tick" '
                f'font-size="{self.tick_font}" '
                f'text-anchor="end">{value:.{self.tick_places}f}</text>'
            )
        return parts

    def _series(self, series, to_x, to_y) -> list[str]:
        parts = []
        for name, points in series.items():
            color = CONCEPT_COLOR if name == "concept" else RANDOM_COLOR
            ordered = sorted(points)
            path = " ".join(
                ("M" if i == 0 else "L") + f"{to_x(x):.1f} {to_y(y):.1f}"
                for i, (x, y) in enumerate(ordered)
            )
            dash = ""
            if self.dash_second_series and name != "concept":
                dash = ' stroke-dasharray="4 3"'
            parts.append(
                f'<path d="{path}" fill="none" stroke="{color}" stroke-width="1.8"{dash}/>'
            )
            for index, (x, y) in enumerate(ordered):
                radius = 4 if index == len(ordered) - 1 else 2.6
                parts.append(
                    f'<circle cx="{to_x(x):.1f}" cy="{to_y(y):.1f}" '
                    f'r="{radius}" fill="{color}"/>'
                )
        return parts

    def _x_axis(self, strengths: list[float], to_x) -> list[str]:
        parts = [
            f'<text x="{to_x(x):.1f}" y="{self.height - self.pad_bottom + 17:.0f}" '
            f'class="tick" font-size="{self.tick_font}" '
            f'text-anchor="middle">{x:g}</text>'
            for x in strengths
        ]
        parts.append(
            f'<text x="{self.width / 2:.0f}" y="{self.height - 4}" class="axis" '
            f'font-size="{self.tick_font}" '
            f'text-anchor="middle">steering strength</text>'
        )
        return parts
