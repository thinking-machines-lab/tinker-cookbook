"""
Logtree: Scope-based logging library for creating nested HTML reports.

This module provides a context-based API for generating structured HTML logs
that reflect the call tree of your code. Ideal for logging RL rollouts,
model evaluations, and other hierarchical computations.

Example usage:
    import logtree

    async def train_iteration():
        with logtree.init_trace("Training Iteration 1", path="output.html"):
            with logtree.scope_header("Sampling"):
                logtree.log_text("Generated 100 samples")
                await sample_data()

            with logtree.scope_header("Training"):
                logtree.log_text("Loss: 0.42")
                await train_model()
"""

import functools
import html as html_module
import inspect
import os
import traceback
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Iterator, Mapping, Protocol, Sequence, TypeVar, overload

# Context variables for task-local state
_current_trace: ContextVar["Trace | None"] = ContextVar("lt_current_trace", default=None)
_container_stack: ContextVar["tuple[Node, ...]"] = ContextVar("lt_container_stack", default=())
_header_depth: ContextVar["tuple[int, ...]"] = ContextVar("lt_header_depth", default=())
_logging_disabled: ContextVar[bool] = ContextVar("lt_logging_disabled", default=False)


class Formatter(Protocol):
    """Protocol for objects that can format themselves as HTML with CSS."""

    def to_html(self) -> str:
        """Generate HTML representation of this object."""
        ...

    def get_css(self) -> str:
        """Get CSS needed to style this object's HTML."""
        ...


@dataclass
class Node:
    """Represents an HTML element in the tree."""

    tag: str
    attrs: dict[str, str] = field(default_factory=dict)
    children: list["Node | str"] = field(default_factory=list)

    def to_html(self, indent: int = 0) -> str:
        """Convert node to HTML string."""
        ind = "  " * indent
        attrs_str = "".join(
            f' {k}="{html_module.escape(v, quote=True)}"' for k, v in self.attrs.items()
        )

        if not self.children:
            return f"{ind}<{self.tag}{attrs_str}></{self.tag}>\n"

        lines = [f"{ind}<{self.tag}{attrs_str}>\n"]
        for child in self.children:
            if isinstance(child, str):
                lines.append(child)
            else:
                lines.append(child.to_html(indent + 1))
        lines.append(f"{ind}</{self.tag}>\n")
        return "".join(lines)


@dataclass
class Theme:
    """Theme configuration for HTML output."""

    css_text: str | None = None  # Custom CSS; if None, use built-in
    css_urls: list[str] = field(default_factory=list)
    css_vars: dict[str, str] = field(default_factory=dict)  # CSS custom properties


class Trace:
    """Root trace object representing an HTML document."""

    def __init__(self, title: str, path: str | os.PathLike | None, write_on_error: bool):
        self.title = title
        self.path = Path(path) if path is not None else None
        self.write_on_error = write_on_error
        self.started_at = datetime.now()
        self.root = Node("body", {"class": "lt-root"})
        self._formatter_css: set[str] = set()  # Deduplicated CSS from formatters

    def _register_formatter_css(self, css: str) -> None:
        """Register CSS from a formatter (deduplicated per trace)."""
        if css:
            self._formatter_css.add(css)

    def body_html(self, wrap_body: bool = True) -> str:
        """Get the body HTML."""
        inner = self.root.to_html(indent=0)
        if wrap_body:
            return inner
        else:
            # Return just the inner content
            return "\n".join(
                line for line in inner.split("\n") if "<body" not in line and "</body>" not in line
            )

    def get_html(self) -> str:
        """Alias for body_html()."""
        return self.body_html(wrap_body=True)

    def head_html(
        self, theme: Theme | None = None, title: str | None = None, extra_head: str | None = None
    ) -> str:
        """Generate the <head> section of the HTML document."""
        if theme is None:
            theme = Theme()

        parts = []
        parts.append(f"<title>{html_module.escape(title or self.title)}</title>")
        parts.append('<meta charset="UTF-8">')
        parts.append('<meta name="viewport" content="width=device-width, initial-scale=1.0">')

        # External CSS
        for url in theme.css_urls:
            parts.append(f'<link rel="stylesheet" href="{html_module.escape(url, quote=True)}">')

        # Inline CSS
        css = theme.css_text if theme.css_text is not None else _DEFAULT_CSS
        if css or self._formatter_css:
            parts.append("<style>")
            if css:
                parts.append(css)
            # CSS custom properties
            if theme.css_vars:
                parts.append(":root {")
                for key, value in theme.css_vars.items():
                    parts.append(f"  {key}: {value};")
                parts.append("}")
            # Formatter CSS
            if self._formatter_css:
                parts.append("\n/* Formatter CSS */")
                for formatter_css in self._formatter_css:
                    parts.append(formatter_css)
            parts.append("</style>")

        if extra_head:
            parts.append(extra_head)

        return "\n".join(parts)


# Default CSS styling
_DEFAULT_CSS = """
body {
    font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
    line-height: 1.6;
    max-width: 1200px;
    margin: 0 auto;
    padding: 20px;
    background: var(--lt-bg, #f5f5f5);
    color: var(--lt-text, #333);
}

.lt-root {
    background: var(--lt-card, white);
    padding: 2rem;
    border-radius: 8px;
    box-shadow: 0 2px 4px rgba(0,0,0,0.1);
}

.lt-title {
    margin: 0 0 0.5rem 0;
    color: var(--lt-accent, #2563eb);
    border-bottom: 2px solid var(--lt-border, #e5e7eb);
    padding-bottom: 0.5rem;
}

.lt-subtitle {
    color: var(--lt-sub, #666);
    font-size: 0.875rem;
    margin-bottom: 2rem;
}

.lt-section {
    margin: 1.5rem 0;
    padding-left: 1rem;
    border-left: 2px solid var(--lt-border, #e5e7eb);
}

.lt-section-body {
    margin-top: 0.5rem;
}

.lt-section h2, .lt-section h3, .lt-section h4, .lt-section h5, .lt-section h6 {
    margin: 0.5rem 0;
    color: var(--lt-accent, #2563eb);
}

.lt-p {
    margin: 0.5rem 0;
    white-space: pre-wrap;
}

.lt-details {
    margin: 0.5rem 0;
    border: 1px solid var(--lt-border, #e5e7eb);
    border-radius: 4px;
    padding: 0.5rem;
}

.lt-details summary {
    cursor: pointer;
    font-weight: 600;
    user-select: none;
}

.lt-details-body {
    margin-top: 0.5rem;
    padding: 0.5rem;
    background: var(--lt-bg, #f5f5f5);
    border-radius: 4px;
    overflow-x: auto;
}

.lt-details-body pre {
    margin: 0;
    font-family: var(--lt-mono, "Courier New", monospace);
    font-size: 0.875rem;
    white-space: pre-wrap;
}

.lt-table {
    border-collapse: collapse;
    width: 100%;
    margin: 1rem 0;
    font-size: 0.875rem;
}

.lt-table th {
    background: var(--lt-accent, #2563eb);
    color: white;
    padding: 0.5rem;
    text-align: left;
    font-weight: 600;
}

.lt-table td {
    padding: 0.5rem;
    border-bottom: 1px solid var(--lt-border, #e5e7eb);
}

.lt-table tr:nth-child(even) {
    background: var(--lt-bg, #f5f5f5);
}

.lt-table-caption {
    font-weight: 600;
    margin-bottom: 0.5rem;
    color: var(--lt-text, #333);
}

.lt-exc {
    background: #fee;
    border: 2px solid #c00;
    border-radius: 4px;
    padding: 1rem;
    margin: 1rem 0;
}

.lt-exc summary {
    color: #c00;
    font-weight: 700;
    cursor: pointer;
}

.lt-exc pre {
    margin-top: 0.5rem;
    font-family: var(--lt-mono, "Courier New", monospace);
    font-size: 0.875rem;
    overflow-x: auto;
}

.answer, .reward {
    font-weight: 600;
    padding: 0.25rem 0.5rem;
    border-radius: 4px;
    display: inline-block;
    margin: 0.25rem 0;
}

.answer {
    background: #dbeafe;
    color: #1e40af;
}

.reward {
    background: #dcfce7;
    color: #166534;
}
"""


# Helper functions


def _normalize_attrs(**attrs: Any) -> dict[str, str]:
    """Normalize attribute names (class_ -> class, data__foo -> data-foo)."""
    result = {}
    for key, value in attrs.items():
        if key == "class_":
            key = "class"
        elif key.startswith("data__"):
            key = key.replace("__", "-", 1)
        result[key] = str(value)
    return result


def _append(node: Node) -> None:
    """Append a node to the current container."""
    stack = _container_stack.get()
    if not stack:
        raise RuntimeError("No active container to append to")
    stack[-1].children.append(node)


def _next_header_level() -> int:
    """Get the next header level based on current depth."""
    depth = _header_depth.get()
    current = depth[-1] if depth else 1
    return min(6, current + 1)


def _is_logging_enabled() -> bool:
    """Check if logging is currently enabled."""
    return _current_trace.get() is not None and not _logging_disabled.get()


@contextmanager
def _in_container(node: Node) -> Iterator[None]:
    """Context manager to push/pop a container."""
    token = _container_stack.set(_container_stack.get() + (node,))
    try:
        yield
    finally:
        _container_stack.reset(token)


def _exception_block(exc: BaseException) -> Node:
    """Create a details block for an exception."""
    tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
    details_node = Node("details", {"class": "lt-exc", "open": "open"})
    details_node.children.append(Node("summary", {}, [f"Exception: {type(exc).__name__}: {exc}"]))
    pre_node = Node("pre", {})
    pre_node.children.append(html_module.escape(tb_str))
    details_node.children.append(pre_node)
    return details_node


def _write_trace(trace: Trace, theme: Theme | None = None) -> None:
    """Write the trace to disk."""
    if trace.path is None:
        return

    trace.path.parent.mkdir(parents=True, exist_ok=True)

    with open(trace.path, "w") as f:
        f.write("<!doctype html>\n")
        f.write('<html lang="en">\n')
        f.write("<head>\n")
        f.write(trace.head_html(theme=theme))
        f.write("</head>\n")
        f.write(trace.body_html(wrap_body=True))
        f.write("</html>\n")


# Public API: Trace lifecycle


@contextmanager
def init_trace(
    title: str, path: str | os.PathLike | None = None, *, write_on_error: bool = True
) -> Iterator[Trace]:
    """
    Initialize a new trace context.

    Args:
        title: Title for the HTML document (becomes <h1>)
        path: Path to write HTML file (None = don't write automatically)
        write_on_error: If True, write partial HTML even if exception occurs

    Example:
        with logtree.init_trace("My Report", path="output.html"):
            logtree.log_text("Hello world")
    """
    trace = Trace(title, path, write_on_error=write_on_error)

    tok_t = _current_trace.set(trace)
    tok_s = _container_stack.set((trace.root,))
    tok_h = _header_depth.set((1,))

    # Emit title and subtitle
    _append(Node("h1", {"class": "lt-title"}, [html_module.escape(title)]))
    _append(
        Node(
            "div",
            {"class": "lt-subtitle"},
            [f"Generated {trace.started_at.isoformat(timespec='seconds')}"],
        )
    )

    try:
        yield trace
    except BaseException as e:
        _append(_exception_block(e))
        if write_on_error and trace.path is not None:
            _write_trace(trace)
        raise
    else:
        if trace.path is not None:
            _write_trace(trace)
    finally:
        # Always reset context even on exception
        _header_depth.reset(tok_h)
        _container_stack.reset(tok_s)
        _current_trace.reset(tok_t)


@contextmanager
def scope_header(title: str, **attrs: Any) -> Iterator[None]:
    """
    Open a section with an auto-leveled header.

    Args:
        title: Text for the header
        **attrs: HTML attributes (use class_="foo" for class, data__x="y" for data-x)

    Example:
        with logtree.scope_header("Results", class_="important"):
            logtree.log_text("Success rate: 95%")
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        yield
        return

    section = Node("section", {"class": "lt-section", **_normalize_attrs(**attrs)})
    _append(section)

    tok_h = None
    try:
        with _in_container(section):
            h = _next_header_level()
            _append(Node(f"h{h}", {"class": f"lt-h{h}"}, [html_module.escape(title)]))

            # Push header level for nested scopes
            tok_h = _header_depth.set(_header_depth.get() + (h,))
            body = Node("div", {"class": "lt-section-body"})
            _append(body)
            with _in_container(body):
                yield
    finally:
        # Always reset context even on exception
        if tok_h is not None:
            _header_depth.reset(tok_h)


F = TypeVar("F", bound=Callable[..., Any])


# Overloads the parameterized usage
@overload
def scope_header_decorator(title: str) -> Callable[[F], F]: ...  # String title


# Overloads the bare usage
@overload
def scope_header_decorator(title: F) -> F: ...  # Bare: @scope_header_decorator


def scope_header_decorator(
    title: str | F,
) -> F | Callable[[F], F]:
    """
    Decorator to wrap function in a scope_header.

    Args:
        title: String or function returning string

    Examples:
        @logtree.scope_header_decorator
        async def process_batch():
            ...

        @logtree.scope_header_decorator("Handling item")
        def handle_item():
            ...
    """
    title_str = title if isinstance(title, str) else title.__name__

    def _wrap(fn: F) -> F:
        if inspect.iscoroutinefunction(fn):

            @functools.wraps(fn)
            async def aw(*args: Any, **kwargs: Any) -> Any:
                # Graceful degradation: if logging is disabled, just run the function
                if not _is_logging_enabled():
                    return await fn(*args, **kwargs)

                with scope_header(title_str):
                    return await fn(*args, **kwargs)

            return aw  # type: ignore
        else:

            @functools.wraps(fn)
            def w(*args: Any, **kwargs: Any) -> Any:
                # Graceful degradation: if logging is disabled, just run the function
                if not _is_logging_enabled():
                    return fn(*args, **kwargs)

                with scope_header(title_str):
                    return fn(*args, **kwargs)

            return w  # type: ignore

    if isinstance(title, str):
        return _wrap
    else:
        fn = title
        return _wrap(fn)


@contextmanager
def scope_div(**attrs: Any) -> Iterator[None]:
    """
    Open a <div> scope (does not change header level).

    Args:
        **attrs: HTML attributes

    Example:
        with logtree.scope_div(class_="grading"):
            logtree.log_text("Grade: A")
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        yield
        return

    div = Node("div", _normalize_attrs(**attrs))
    _append(div)
    with _in_container(div):
        yield


@contextmanager
def scope_disable() -> Iterator[None]:
    """
    Disable all logging within this scope.

    Example:
        with scope_header("Group A") if should_log else scope_disable():
            logtree.log_text("Data here")
    """
    token = _logging_disabled.set(True)
    try:
        yield
    finally:
        _logging_disabled.reset(token)


@contextmanager
def optional_enable_logging(enable: bool) -> Iterator[None]:
    """Context manager to optionally enable logging."""
    if enable:
        yield
    else:
        with scope_disable():
            yield


@contextmanager
def scope_details(summary: str) -> Iterator[None]:
    """
    Open a collapsible <details> scope.

    Args:
        summary: Summary text shown when collapsed

    Example:
        with logtree.scope_details("Click to expand"):
            logtree.log_text("Hidden content")
            logtree.log_text("More hidden content")
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        yield
        return

    details_node = Node("details", {"class": "lt-details"})
    details_node.children.append(Node("summary", {}, [html_module.escape(summary)]))

    body_div = Node("div", {"class": "lt-details-body"})
    details_node.children.append(body_div)

    _append(details_node)
    with _in_container(body_div):
        yield


# Public API: Content


def log_text(text: str, *, div_class: str | None = None) -> None:
    """
    Log a text paragraph.

    Args:
        text: Text to log (will be HTML-escaped)
        div_class: If set, wrap in <div class="{div_class}"> instead of <p>

    Example:
        logtree.log_text("Processing complete")
        logtree.log_text("Score: 0.95", div_class="score")
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    escaped = html_module.escape(text)
    if div_class:
        _append(Node("div", {"class": div_class}, [escaped]))
    else:
        _append(Node("p", {"class": "lt-p"}, [escaped]))


def log_html(html: str, *, div_class: str | None = None) -> None:
    """
    Log raw HTML (not escaped).

    Args:
        html: HTML string to insert verbatim
        div_class: If set, wrap in <div class="{div_class}">

    Example:
        logtree.log_html("<strong>Important</strong>")
        logtree.log_html(conversation_html, div_class="conversation")
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    if div_class:
        div = Node("div", {"class": div_class})
        div.children.append(html)
        _append(div)
    else:
        # Create a container node that holds raw HTML
        container = Node("div", {})
        container.children.append(html)
        _append(container)


def log_formatter(formatter: Formatter) -> None:
    """
    Log an object that knows how to format itself as HTML.

    The formatter's CSS will be automatically included in the trace output.
    CSS is deduplicated per trace, so logging multiple objects of the same
    type only includes the CSS once.

    Args:
        formatter: Object implementing the Formatter protocol (to_html() and get_css())

    Example:
        from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

        logtree.log_formatter(ConversationFormatter(messages=[...]))
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    # Register CSS from the formatter (deduplicated)
    trace = _current_trace.get()
    assert trace is not None  # _is_logging_enabled() ensures this
    css = formatter.get_css()
    trace._register_formatter_css(css)

    # Log the HTML
    html = formatter.to_html()
    log_html(html)


def details(text: str, *, summary: str = "Details", pre: bool = True) -> None:
    """
    Log collapsible details block.

    Args:
        text: Content text
        summary: Summary text (what you see when collapsed)
        pre: If True, use <pre> (preserves whitespace), else <div>

    Example:
        logtree.details(long_chain_of_thought, summary="CoT Reasoning", pre=True)
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    details_node = Node("details", {"class": "lt-details"})
    details_node.children.append(Node("summary", {}, [html_module.escape(summary)]))

    body_node = Node("pre" if pre else "div", {"class": "lt-details-body"})
    body_node.children.append(html_module.escape(text))
    details_node.children.append(body_node)

    _append(details_node)


def header(text: str, *, level: int | None = None) -> None:
    """
    Log an inline header.

    Args:
        text: Header text
        level: Header level (1-6), or None to auto-compute from scope depth

    Example:
        logtree.header("Results")
        logtree.header("Subsection", level=4)
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    h = level if level is not None else _next_header_level()
    h = max(1, min(6, h))
    _append(Node(f"h{h}", {"class": f"lt-h{h}"}, [html_module.escape(text)]))


# Public API: Tables


def table(obj: Any, *, caption: str | None = None) -> None:
    """
    Log a table from various data types.

    Supports:
    - pandas.DataFrame
    - list[dict]
    - list[list]

    Does NOT support raw dict (use table_from_dict or table_from_dict_of_lists).

    Args:
        obj: Data object
        caption: Optional caption text

    Example:
        logtree.table(df, caption="Results")
        logtree.table([{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}])
    """
    # Graceful degradation: if logging is disabled, do nothing
    if not _is_logging_enabled():
        return

    if isinstance(obj, dict):
        raise TypeError(
            "table() does not accept dict directly. Use table_from_dict() or table_from_dict_of_lists()."
        )

    # Try DataFrame
    try:
        import pandas as pd

        if isinstance(obj, pd.DataFrame):
            html_str = obj.to_html(classes="lt-table", border=0, escape=True, index=False)
            if caption:
                _append(Node("div", {"class": "lt-table-caption"}, [html_module.escape(caption)]))
            _append(Node("div", {}, [html_str]))
            return
    except ImportError:
        pass

    # list[dict]
    if isinstance(obj, list) and obj and isinstance(obj[0], dict):
        _table_from_list_of_dicts(obj, caption=caption)
        return

    # list[list]
    if isinstance(obj, list) and obj and isinstance(obj[0], (list, tuple)):
        _table_from_list_of_lists(obj, caption=caption)
        return

    raise TypeError(f"table() does not support type {type(obj)}")


def table_from_dict(
    data: Mapping[Any, Any],
    *,
    caption: str | None = None,
    key_header: str = "key",
    value_header: str = "value",
    sort_by: str | None = None,
) -> None:
    """
    Log a two-column key-value table from a dict.

    Args:
        data: Dictionary to display
        caption: Optional caption
        key_header: Column header for keys
        value_header: Column header for values
        sort_by: "key", "value", or None

    Example:
        logtree.table_from_dict({"lr": 0.001, "batch_size": 32}, caption="Hyperparams")
    """
    items = list(data.items())
    if sort_by == "key":
        items.sort(key=lambda x: x[0])
    elif sort_by == "value":
        items.sort(key=lambda x: x[1])

    rows = [[key_header, value_header]] + [[str(k), str(v)] for k, v in items]
    _table_from_list_of_lists(rows, caption=caption, has_header=True)


def table_from_dict_of_lists(
    columns: Mapping[str, Sequence[Any]],
    *,
    caption: str | None = None,
    order: Sequence[str] | None = None,
) -> None:
    """
    Log a columnar table from dict of lists.

    Args:
        columns: Dict where keys are column names, values are column data
        caption: Optional caption
        order: Column order (if None, use insertion order)

    Example:
        logtree.table_from_dict_of_lists({
            "name": ["Alice", "Bob"],
            "score": [95, 87]
        })
    """
    if not columns:
        return

    # Validate equal lengths
    lengths = [len(v) for v in columns.values()]
    if len(set(lengths)) > 1:
        raise ValueError("All columns must have equal length")

    col_names = list(order) if order else list(columns.keys())
    rows = [col_names]
    for i in range(lengths[0]):
        rows.append([str(columns[name][i]) for name in col_names])

    _table_from_list_of_lists(rows, caption=caption, has_header=True)


def _table_from_list_of_dicts(data: list[dict], *, caption: str | None = None) -> None:
    """Helper: create table from list of dicts."""
    if not data:
        return

    keys = list(data[0].keys())
    rows = [keys]
    for item in data:
        rows.append([str(item.get(k, "")) for k in keys])

    _table_from_list_of_lists(rows, caption=caption, has_header=True)


def _table_from_list_of_lists(
    rows: list[list[Any]], *, caption: str | None = None, has_header: bool = False
) -> None:
    """Helper: create HTML table from list of lists."""
    if not rows:
        return

    if caption:
        _append(Node("div", {"class": "lt-table-caption"}, [html_module.escape(caption)]))

    table_node = Node("table", {"class": "lt-table"})

    if has_header:
        thead = Node("thead")
        tr = Node("tr")
        for cell in rows[0]:
            tr.children.append(Node("th", {}, [html_module.escape(str(cell))]))
        thead.children.append(tr)
        table_node.children.append(thead)
        rows = rows[1:]

    tbody = Node("tbody")
    for row in rows:
        tr = Node("tr")
        for cell in row:
            tr.children.append(Node("td", {}, [html_module.escape(str(cell))]))
        tbody.children.append(tr)
    table_node.children.append(tbody)

    _append(table_node)


# Public API: Export & theming


def write_html_with_default_style(
    body_html: str,
    path: str | os.PathLike,
    *,
    title: str = "Trace",
    theme: Theme | None = None,
    lang: str = "en",
    extra_head: str | None = None,
) -> None:
    """
    Write a complete HTML document with default styling.

    Args:
        body_html: Body HTML (with or without <body> tags)
        path: Output file path
        title: Document title
        theme: Optional theme
        lang: HTML lang attribute
        extra_head: Extra content for <head>
    """
    if theme is None:
        theme = Theme()

    # Create a temporary trace just for head generation
    trace = Trace(title, None, False)

    path_obj = Path(path)
    path_obj.parent.mkdir(parents=True, exist_ok=True)

    with open(path_obj, "w") as f:
        f.write(f'<!doctype html>\n<html lang="{html_module.escape(lang)}">\n')
        f.write("<head>\n")
        f.write(trace.head_html(theme=theme, title=title, extra_head=extra_head))
        f.write("</head>\n")
        # Ensure body tags are present
        if "<body" not in body_html:
            f.write("<body>\n")
            f.write(body_html)
            f.write("</body>\n")
        else:
            f.write(body_html)
        f.write("</html>\n")


def jinja_context(trace: Trace, **extra: Any) -> dict[str, Any]:
    """
    Create a context dict for Jinja2 templates.

    Args:
        trace: Trace object
        **extra: Additional context variables

    Returns:
        Dict with standard keys: title, generated_at, started_at, body_html, head_html
    """
    return {
        "title": trace.title,
        "generated_at": datetime.now().isoformat(),
        "started_at": trace.started_at.isoformat(),
        "body_html": trace.body_html(),
        "head_html": trace.head_html(),
        **extra,
    }


def render_with_jinja(
    env: Any,
    template_name: str,
    *,
    context: dict[str, Any],
    write_to: str | os.PathLike | None = None,
) -> str:
    """
    Render using Jinja2 (requires jinja2 to be installed).

    Args:
        env: jinja2.Environment instance
        template_name: Template file name
        context: Template context
        write_to: Optional path to write output

    Returns:
        Rendered HTML string
    """
    template = env.get_template(template_name)
    html = template.render(**context)

    if write_to is not None:
        path = Path(write_to)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(html)

    return html
