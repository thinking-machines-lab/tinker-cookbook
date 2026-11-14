"""Headless rendering of p5.brush sketches to PNG.

A sketch is a JavaScript program that defines ``setup()`` and paints on a
WEBGL canvas with p5.brush. It is executed in a sandboxed headless Chromium
page (Playwright, SwiftShader WebGL) with the pinned p5 + p5.brush sources
inlined; the canvas is read back as PNG. The result carries the three
"compile gate" facts the reward uses: did the sketch run without an error,
did it actually draw with the brush library, and is the canvas non-blank.
"""

from __future__ import annotations

import asyncio
import base64
import contextlib
import io
import logging
import re
import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from PIL import Image, ImageStat

from tinker_cookbook.recipes.paint_rl.assets import get_library_sources

if TYPE_CHECKING:
    from playwright.async_api import Browser, Playwright

logger = logging.getLogger(__name__)

_CODE_FENCE = re.compile(r"```(?:javascript|js)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)
_BRUSH_CALL = re.compile(r"\bbrush\.([A-Za-z_]\w*)\s*\(")
# Calls that configure the library rather than draw; they do not count as
# "using the brush".
_NON_DRAWING_CALLS = frozenset(
    {"load", "seed", "instance", "preload", "colorCache", "scaleBrushes", "noLoop"}
)

# Runs after the sketch: wraps setup() so a thrown error is captured instead
# of killing the page silently, draws one frame if the sketch defined draw(),
# then freezes the loop and signals completion.
_HARNESS_JS = """
(function () {
  var userSetup = window.setup;
  var userDraw = window.draw;
  if (typeof userSetup !== 'function') {
    window.__error = window.__error || 'no setup() function defined';
    window.__done = true;
    return;
  }
  window.draw = undefined;
  window.setup = function () {
    try {
      userSetup();
      if (typeof userDraw === 'function') { userDraw(); }
      if (typeof noLoop === 'function') { noLoop(); }
    } catch (e) {
      window.__error = String((e && e.stack) || e);
    }
    window.__done = true;
  };
})();
"""

_CHROMIUM_ARGS = [
    "--use-gl=angle",
    "--use-angle=swiftshader",
    "--enable-unsafe-swiftshader",
    "--ignore-gpu-blocklist",
    "--disable-dev-shm-usage",
]

_PAGE_TEMPLATE = """<!doctype html>
<html><head><meta charset="utf-8"><style>html,body{{margin:0;background:#fff}}</style></head>
<body>
<script>{p5}</script>
<script>{brush}</script>
<script>
window.__done = false; window.__error = null;
// The harness freezes the loop after setup() itself. Models trained on p5
// habitually end setup() with noLoop() and sometimes write brush.noLoop(),
// which p5.brush does not define: make it the harmless no-op it means to be
// instead of a TypeError that loses the painting.
if (window.brush && typeof window.brush.noLoop !== 'function') {{
  try {{ window.brush.noLoop = function () {{}}; }} catch (e) {{}}
}}
window.addEventListener('error', function (e) {{
  if (!window.__error) {{ window.__error = String((e && (e.message || e.error)) || 'script error'); }}
}});
</script>
<script>
{code}
</script>
<script>{harness}</script>
</body></html>
"""


def extract_code(response_text: str) -> str | None:
    """Pull the sketch out of a model response.

    Prefers the longest fenced ``javascript``/``js`` block; falls back to
    the raw text when it already looks like a sketch (defines ``setup``).
    Returns ``None`` when no code can be found.
    """
    blocks = [m.group(1).strip() for m in _CODE_FENCE.finditer(response_text)]
    blocks = [b for b in blocks if b]
    if blocks:
        return max(blocks, key=len)
    if "function setup" in response_text or "setup =" in response_text:
        return response_text.strip()
    return None


def count_brush_calls(code: str) -> int:
    """Number of distinct p5.brush drawing/config calls in the code."""
    names = {m.group(1) for m in _BRUSH_CALL.finditer(code)}
    return len(names - _NON_DRAWING_CALLS)


def is_blank(image: Image.Image, threshold: float = 6.0) -> bool:
    """True when the canvas is (near) uniform, i.e. nothing was painted."""
    stat = ImageStat.Stat(image.convert("L"))
    return stat.stddev[0] < threshold


@dataclass(frozen=True)
class RenderResult:
    """Outcome of rendering one sketch."""

    ok: bool
    """The sketch ran without error, used the brush, and painted something."""
    png: bytes | None
    """PNG bytes of the canvas (``None`` when nothing could be captured)."""
    error: str | None
    """Error text when ``ok`` is False (JS error, timeout, gate failure)."""
    n_brush_calls: int
    blank: bool
    duration_s: float

    def image(self) -> Image.Image | None:
        if self.png is None:
            return None
        return Image.open(io.BytesIO(self.png)).convert("RGB")


class SketchRenderer:
    """Renders sketches in a pool of headless Chromium browsers.

    Software WebGL (SwiftShader) runs in one GPU process per browser, so
    pages in the same browser contend for it; ``max_concurrency`` renders run
    at once, spread round-robin over ``n_browsers`` browsers. Create one
    renderer per process and reuse it; ``start()`` is called lazily. The
    browsers run on the current asyncio loop, so the renderer must be used
    from the loop it was started on.
    """

    def __init__(
        self,
        canvas_size: int = 512,
        max_concurrency: int = 16,
        timeout_s: float = 120.0,
        min_brush_calls: int = 3,
        n_browsers: int | None = None,
    ):
        self.canvas_size = canvas_size
        self.timeout_s = timeout_s
        self.min_brush_calls = min_brush_calls
        self.max_concurrency = max_concurrency
        # About four pages per browser keeps each SwiftShader GPU process busy
        # without pages starving one another.
        if n_browsers is None:
            n_browsers = -(-max_concurrency // 4)
        self.n_browsers = max(1, min(n_browsers, max_concurrency))
        self._semaphore = asyncio.Semaphore(max_concurrency)
        self._lock = asyncio.Lock()
        self._playwright: Playwright | None = None
        self._browsers: list[Browser] = []
        self._next_browser = 0

    async def start(self) -> None:
        async with self._lock:
            if self._browsers:
                return
            from playwright.async_api import async_playwright

            self._playwright = await async_playwright().start()
            for _ in range(self.n_browsers):
                self._browsers.append(await self._playwright.chromium.launch(args=_CHROMIUM_ARGS))
            logger.info(
                "Started %d headless Chromium browsers for sketch rendering",
                self.n_browsers,
            )

    async def close(self) -> None:
        async with self._lock:
            for browser in self._browsers:
                with contextlib.suppress(Exception):
                    await browser.close()
            self._browsers = []
            if self._playwright is not None:
                await self._playwright.stop()
                self._playwright = None

    def _page_html(self, code: str) -> str:
        p5_src, brush_src = get_library_sources()
        # A stray "</script>" inside the sketch would end the script tag early.
        safe_code = code.replace("</script", "<\\/script")
        return _PAGE_TEMPLATE.format(
            p5=p5_src, brush=brush_src, code=safe_code, harness=_HARNESS_JS
        )

    async def render(self, code: str) -> RenderResult:
        """Render ``code`` and evaluate the compile/brush/blank gates."""
        t0 = time.monotonic()
        n_brush = count_brush_calls(code)
        if n_brush < self.min_brush_calls:
            return RenderResult(
                ok=False,
                png=None,
                error=f"only {n_brush} distinct brush.* drawing calls (need {self.min_brush_calls})",
                n_brush_calls=n_brush,
                blank=True,
                duration_s=time.monotonic() - t0,
            )
        page_html = self._page_html(code)
        async with self._semaphore:
            data_url, error = await self._render_local(page_html)

        png: bytes | None = None
        blank = True
        if data_url and "," in data_url:
            png = base64.b64decode(data_url.split(",", 1)[1])
            try:
                blank = is_blank(Image.open(io.BytesIO(png)))
            except Exception as e:
                error = error or f"bad png: {e}"
                png = None
        if error is None and png is None:
            error = "no canvas was created"
        if error is None and blank:
            error = "canvas is blank"
        return RenderResult(
            ok=error is None,
            png=png,
            error=error,
            n_brush_calls=n_brush,
            blank=blank,
            duration_s=time.monotonic() - t0,
        )

    async def _live_browser(self, index: int) -> Any:
        """The browser in slot ``index``, relaunched first if its process died.

        A crashed Chromium otherwise stays in the pool, and every sketch
        routed to it would fail.
        """
        browser = self._browsers[index]
        if browser.is_connected():
            return browser
        async with self._lock:
            browser = self._browsers[index]
            if not browser.is_connected():
                assert self._playwright is not None
                logger.warning("Headless browser %d died; relaunching it", index)
                browser = await self._playwright.chromium.launch(args=_CHROMIUM_ARGS)
                self._browsers[index] = browser
        return browser

    async def _render_local(self, page_html: str) -> tuple[str | None, str | None]:
        if not self._browsers:
            await self.start()
        index = self._next_browser % len(self._browsers)
        self._next_browser += 1
        # One retry on a relaunched browser if the first one died under us.
        for attempt in range(2):
            browser = await self._live_browser(index)
            context = None
            error: str | None = None
            data_url: str | None = None
            try:
                context = await browser.new_context(
                    viewport={"width": self.canvas_size, "height": self.canvas_size},
                    offline=True,  # no network from inside the sketch
                )
                page = await context.new_page()
                await page.set_content(page_html, timeout=self.timeout_s * 1000)
                await page.wait_for_function(
                    "window.__done === true", timeout=self.timeout_s * 1000
                )
                error = await page.evaluate("window.__error")
                data_url = await page.evaluate(
                    "(() => { const c = document.querySelector('canvas'); "
                    "return c ? c.toDataURL('image/png') : null; })()"
                )
            except Exception as e:  # playwright timeouts, crashed pages or browsers, ...
                error = f"render failed: {type(e).__name__}: {str(e).splitlines()[0][:300]}"
                if attempt == 0 and not browser.is_connected():
                    continue  # the browser died: relaunch it and render again
            finally:
                if context is not None:
                    with contextlib.suppress(Exception):
                        await context.close()
            return data_url, error
        raise AssertionError("unreachable")


_SHARED: SketchRenderer | None = None


def get_shared_renderer(
    canvas_size: int = 512,
    max_concurrency: int = 16,
    timeout_s: float = 120.0,
) -> SketchRenderer:
    """Process-wide renderer (one browser pool), created on first use."""
    global _SHARED
    if (
        _SHARED is None
        or _SHARED.canvas_size != canvas_size
        or _SHARED.timeout_s != timeout_s
        or _SHARED.max_concurrency != max_concurrency
    ):
        _SHARED = SketchRenderer(
            canvas_size=canvas_size,
            max_concurrency=max_concurrency,
            timeout_s=timeout_s,
        )
    return _SHARED
