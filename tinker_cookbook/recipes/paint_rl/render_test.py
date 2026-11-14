from PIL import Image

from tinker_cookbook.recipes.paint_rl.render import (
    count_brush_calls,
    extract_code,
    is_blank,
)


def test_extract_code_prefers_longest_fenced_block():
    text = "intro\n```js\nlet a = 1;\n```\nmore\n```javascript\nfunction setup() {\n  createCanvas(1, 1, WEBGL);\n}\n```\n"
    code = extract_code(text)
    assert code is not None and code.startswith("function setup()")


def test_extract_code_falls_back_to_raw_sketch():
    assert extract_code("function setup() { noLoop(); }") == "function setup() { noLoop(); }"
    assert extract_code("I cannot paint.") is None


def test_count_brush_calls_ignores_setup_calls():
    code = "brush.load(); brush.seed(1); brush.fill('red', 80); brush.circle(0,0,10); brush.fill('blue', 9);"
    assert count_brush_calls(code) == 2
    assert count_brush_calls(code + " brush.noLoop();") == 2


def test_is_blank():
    assert is_blank(Image.new("RGB", (32, 32), "white"))
    im = Image.new("RGB", (32, 32), "white")
    im.paste((10, 10, 200), (0, 0, 16, 32))
    assert not is_blank(im)
