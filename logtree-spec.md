This RFC is for a scope-based logging library that lets you make a bunch of logging calls deep inside your code, and produces a tree-structured object at the end, which reflects the call tree, and which can be rendered to HTML. 

## Motivating example: rollout logging in RLHF

We’d like to instrument the RL code in tinker cookbook to log lots of info from each batch of RL rollouts. Let’s consider the sync RL setting, and let’s consider doing RLHF with a pairwise, CoT reward model. Further, let’s suppose we’d like to write a large HTML file each iteration, with the following contents:

* One section for each task, including  
  * The group of G trajectories sampled for that task  
  * The computation of final reward, which involves a bunch of pairwise comparisons between the G\*(G-1) ordered pairs. In each pairwise comparison,  
    * The raw prompt and completion given to the reward model  
    * The parsed completion, which is separated into a CoT and answer

Just for background, here’s an example of what the reward model’s prompt and completion might look like. The completion is shown in bold.

```
<prompt>
What is love?
</prompt>
<completion_A>
Love is a many-splendored thing.
</completion_A>
<completion_B>
Baby don't hurt me, don't hurt me, no more.
</completion_B>
<reasoning>
Both responses are perfect. It's a tie.
</reasoning>
<winner>Tie</winner>
```

This tie would result in a reward of zero for each of the two completions in the matchup.

## Code sketch

To do this logging, we shouldn’t have to pass a lot of data up and down the call stack – we should be able to log locally, but have everything aggregated nicely at a higher level of the stack.

Here are a few of the relevant snippets of code, based on tinker cookbook, with some logging calls added, using the imagined `logtree` library.

```py

async def do_sync_training(...):
    with logtree.init_trace(f"RL Iteration {iter_id}", path=out_path):
	# Here we initialize the tracing, and provide an output file, which
	# will be written with all the results when we exit the scope
        asyncio.gather(
          *[
   		# ... do all rollouts
	    ]
        )


async def do_group_rollout(
    env_group_builder: EnvGroupBuilder, policy: TokenCompleter, log_label: str|None
) -> TrajectoryGroup:
    with logtree.scope_header(f"Group {log_label}") if log_label else logtree.scope_disable():
        envs_G: Sequence[Env] = await env_group_builder.make_envs()
        trajectories_G = await asyncio.gather(*[do_single_rollout(policy, env) for env in envs_G])
        rewards_and_metrics_G = await env_group_builder.compute_group_rewards(trajectories_G)
        rewards_G, metrics_G = zip(*rewards_and_metrics_G, strict=True)
        logtree.log_text(f"Rewards: {rewards_G}")
    return TrajectoryGroup(trajectories_G, list(rewards_G), list(metrics_G))

# ...

   @logtree.scope_header_decorator
   async def compute_group_rewards(
        self, trajectory_group: list[Trajectory]
    ) -> list[tuple[float, Metrics]]:
        ...
        logtree.log_text(f"Got {len(trajectory_group)} trajectories, doing {len(j_comparisons)} pairwise match-ups.")
        j_rewards = await asyncio.gather(
            *[self.preference_model(comparison) for comparison in j_comparisons]
            )
        ...

...

class CotPairwisePreferenceModel:
    """
    Looks at a prompt and pair of responses, does chain of thought (CoT),
    then outputs a float from -1 (first response better) to 
    1 (second response better)
    """
    @logtree.scope_header_decorator
    def __call__(self, comparison: Comparison) -> float:
        prompt_convo = self.make_prompt_convo(comparison)
        completion_message, is_valid = self.call_model(prompt_convo)
        cot, answer = = self.parse_completion_message(completion_message)
        with logtree.scope_header("pref model prompt"):
           logtree.log_html(convo_to_html(prompt_convo)) # convo_to_html produces an html string wrapped in a <div>
        with logtree.scope_header("pref model response"):
            logtree.log_html(message_to_html(completion_message))
            logtree.log_text(f"Answer: {answer}", div_class="answer") # adds text in a div
        reward= {"A": -1, "Tie": 0, "B": 1}[answer]
        logtree.log_text(f"Reward: {reward:.2f}", div_class="reward")
        return reward
```

This will write a nested HTML.

* Each `@scoped_header_decorator` or `with_scoped_header` creates a header at the next level.  
* `Log_html` and `log_text` each add divs to the html


The html at the end should look like this. (I wasn’t that careful to match it to the code above…but you get the idea).

```html
<html>
<head>
	<title>RL iteration 0</title>
</head>
<body>
<h1>RL Iteration 0<h1>
<h2>Group 0</h2>
<h3>Trajectory 0</h3>
<h4>Prompt</h4>
<div>...</div>
<h4>Completion<h4>
<div>...</div>
...other groups...
<h2>Grading</h2>
Got 4 completions, 12 matchups
<h3>Matchup 0: (0,1)</h3>
<h4>Reward model prompt</h4>
...
<h4>Reward model response</h4>
<div>...</div>
<div>Answer: Tie</div>
...
</body>
</html>

```

---

## Auto header levels (no manual `level=` required)

To keep logging ergonomic, **every `scope_header(...)` (and `@scope_header_decorator`) automatically emits a heading one level deeper than the *current* section header**. The page title (inserted by `init_trace`) is `h1`. The first `scope_header(...)` you open is `h2`, nested scopes are `h3`, and so on (clamped at `h6`). Inline `header(...)` calls—if you use them—also default to “next level down” unless you explicitly override.

> In short: you never need to pass `level=`. It’s inferred from scope nesting.

---

## Full API Reference

### 1) Trace lifecycle

```python
@contextmanager
def init_trace(
    title: str,
    path: str | os.PathLike | None = None,
    *,
    write_on_error: bool = True,
) -> Iterator[Trace]
```

**Semantics**

* If `path` is provided, a **complete HTML document** is written on normal exit. If an exception escapes and `write_on_error=True` (default), a traceback block is appended and the file is still written before re‑raising.
* If `path=None`, **nothing is written automatically**. Grab the body with `trace.get_html()` and wrap it yourself (see Export).
* Async‑safe: active trace and scope stacks are **task‑local** and persist across `await`. Concurrency is supported without mis‑nesting.

```python
class Trace:
    # Export helpers
    def body_html(self, *, wrap_body: bool = True) -> str
    def get_html(self) -> str        # alias: returns <body>…</body>
    def head_html(
        self, *, theme: Theme | None = None,
        title: str | None = None,
        extra_head: str | None = None,
    ) -> str

    # Metadata
    title: str
    started_at: datetime
```

---

### 2) Structure (scopes)

```python
@contextmanager
def scope_header(title: str, **attrs) -> Iterator[None]
```

* Opens a `<section class="lt-section" …>` with an automatically chosen heading `<hN>`:

  * If the enclosing context is the page, this is `h2`; if you’re inside a section, it becomes `h3`, etc. (clamped at `h6`).
* Adds a `<div class="lt-section-body">` inside the section; your nested logs go there.
* `attrs` become HTML attributes on the `<section>` (use `class_="…"`, `data__foo="bar"` etc.).

```python
def scope_header_decorator(
    title: str | Callable[..., str] | None = None,
    **attrs,
) -> Callable[[F], F]
```

* Decorates a function (sync or async) so each call runs inside a `scope_header`.
* `title` rules:

  * **No args**: `@scope_header_decorator` → use the function’s name as the header text.
  * String: `@scope_header_decorator("Pairwise comparisons")`
  * Callable: `@scope_header_decorator(lambda self, c: f"Compare {c.i},{c.j}")`
* Inherits auto header level from where it’s called. Attributes are applied to the generated `<section>`.

```python
@contextmanager
def scope_div(**attrs) -> Iterator[None]
```

* Opens a `<div …>` as a nesting scope (does **not** change the inferred header level). Useful for stylable panes (e.g., grading blocks: `with scope_div(class_="grading"):`).

```python
@contextmanager
def scope_disable() -> Iterator[None]
```

* No‑op scope for conditional scoping, e.g.:

  ```python
  with scope_header(f"Group {label}") if label else scope_disable():
      ...
  ```

---

### 3) Content

```python
def log_text(text: str, *, div_class: str | None = None) -> None
```

* Adds a human‑readable paragraph (`<p class="lt-p">…</p>`). If `div_class` is set, the text is wrapped in `<div class="{div_class}">…</div>` instead—handy for styling specific callouts like answers/rewards.

```python
def log_html(html: str, *, div_class: str | None = None) -> None
```

* Inserts **verbatim** HTML (no escaping). If `div_class` is set, wraps it in `<div class="…">…</div>` first. Ideal for helpers like `convo_to_html(...)` / `message_to_html(...)` that already emit safe markup. (If content may be untrusted, sanitize upstream.)

```python
def details(text: str, *, summary: str = "Details", pre: bool = True) -> None
```

* Collapsible block using native `<details>`:

  * Renders `<summary>summary</summary>` plus a body set to `<pre>` (preserves whitespace) by default, or `<div>` if `pre=False`.
  * Great for long CoT dumps.

```python
def header(text: str, *, level: int | None = None) -> None
```

* Adds an inline header `<hN>text</hN>`. If `level=None`, picks the **next deeper level** based on the current scope (same auto rule as `scope_header`). You can override with an explicit `level` if you need fine control.

---

### 4) Tables

```python
def table(obj: Any, *, caption: str | None = None) -> None
```

* For **DataFrames**: uses `to_html(classes="lt-table", border=0, escape=True)`.
* For **`list[dict]`**: keys become columns (missing keys render empty cells).
* For **`list[list]`**: auto‑numbered columns (`col0`, `col1`, …).
* **Dicts are explicit**—use one of these:

```python
def table_from_dict(
    data: Mapping[Any, Any],
    *,
    caption: str | None = None,
    key_header: str = "key",
    value_header: str = "value",
    sort_by: str | None = None,       # "key" | "value" | None
) -> None
```

* Two‑column K/V table (hyperparams, per‑rollout signals). Preserves insertion order unless `sort_by` is given.

```python
def table_from_dict_of_lists(
    columns: Mapping[str, Sequence[Any]],
    *,
    caption: str | None = None,
    order: Sequence[str] | None = None,
) -> None
```

* Columnar table (all lists must be equal length); choose column order with `order`.

---

### 5) Export & theming

```python
@dataclass(frozen=True)
class Theme:
    css_text: str | None = None          # if None, use built-in CSS
    css_urls: list[str] = field(default_factory=list)
    css_vars: dict[str, str] = field(default_factory=dict)  # appended to :root { … }
```

```python
def write_html_with_default_style(
    body_html: str,                      # <body>…</body> OR inner body HTML
    path: str | os.PathLike,
    *,
    title: str = "Trace",
    theme: Theme | None = None,
    lang: str = "en",
    extra_head: str | None = None,
) -> None
```

* Quick wrapper to turn a trace body into a complete HTML doc on disk.

```python
def jinja_context(trace: Trace, **extra) -> dict[str, object]
```

* Returns a dict: `{"title", "generated_at", "started_at", "body_html", "head_html", **extra}` for templating engines.

```python
def render_with_jinja(
    env, template_name: str, *,
    context: dict[str, object],
    write_to: str | os.PathLike | None = None,
) -> str
```

* Renders with your Jinja2 `Environment` (no built‑in dependency); optionally writes to file.

**Styling contract (stable classes & CSS vars)**

| Purpose         | Class name         | Element       |
| --------------- | ------------------ | ------------- |
| Root container  | `lt-root`          | `<main>`      |
| Title           | `lt-title`         | `<h1>`        |
| Subtitle        | `lt-subtitle`      | `<div>`       |
| Section         | `lt-section`       | `<section>`   |
| Section body    | `lt-section-body`  | `<div>`       |
| Paragraph       | `lt-p`             | `<p>`         |
| Details wrapper | `lt-details`       | `<details>`   |
| Details body    | `lt-details-body`  | `<pre>/<div>` |
| Table           | `lt-table`         | `<table>`     |
| Table caption   | `lt-table-caption` | `<div>`       |
| Exception block | `lt-exc`           | `<details>`   |

CSS variables (overridable via `Theme.css_vars`):
`--lt-bg`, `--lt-card`, `--lt-text`, `--lt-sub`, `--lt-accent`, `--lt-border`, `--lt-mono`.

---

## Possible implementation ideas with context vars

This section sketches how to implement the behavior above—especially **auto header levels** and asyncio safety—without committing to a particular code layout.

### A. Task‑local context

Use `contextvars` to isolate per‑Task state:

```python
_current_trace: ContextVar[Trace | None] = ContextVar("lt_current_trace", default=None)
_container_stack: ContextVar[tuple[Node, ...]] = ContextVar("lt_container_stack", default=())
_header_depth: ContextVar[tuple[int, ...]] = ContextVar("lt_header_depth", default=())  # NEW
```

* `_current_trace` — which `Trace` we’re writing to in this Task.
* `_container_stack` — where to append the next node (push/pop on scopes).
* `_header_depth` — a **stack of header levels** so auto‑leveling is deterministic.

### B. Entering a trace

```python
@contextmanager
def init_trace(title, path=None, *, write_on_error=True):
    trace = Trace(title, path, write_on_error=write_on_error)
    tok_t = _current_trace.set(trace)
    tok_s = _container_stack.set((trace.root,))
    tok_h = _header_depth.set((1,))              # page title is h1

    # Emit title/subtitle
    _append(Node("h1", {"class": "lt-title"}, [title]))
    _append(Node("div", {"class": "lt-subtitle"}, [f"Generated {trace.started_at.isoformat(timespec='seconds')}"]))

    try:
        yield trace
    except BaseException as e:
        _append(_exception_block(e))
        if write_on_error and trace.path is not None:
            _write_trace(trace)
        _header_depth.reset(tok_h)
        _container_stack.reset(tok_s)
        _current_trace.reset(tok_t)
        raise
    else:
        if trace.path is not None:
            _write_trace(trace)
        _header_depth.reset(tok_h)
        _container_stack.reset(tok_s)
        _current_trace.reset(tok_t)
```

### C. Auto header level computation

Helper:

```python
def _next_header_level() -> int:
    depth = _header_depth.get()
    current = depth[-1] if depth else 1
    return min(6, current + 1)
```

`scope_header`:

```python
@contextmanager
def scope_header(title: str, **attrs):
    section = Node("section", {"class": "lt-section", **_normalize(attrs)})
    _append(section)

    with _in_container(section):
        h = _next_header_level()
        _append(Node(f"h{h}", {"class": f"lt-h{h}"}, [title]))

        # push header level for nested scopes
        tok_h = _header_depth.set(_header_depth.get() + (h,))
        try:
            body = Node("div", {"class": "lt-section-body"})
            _append(body)
            with _in_container(body):
                yield
        finally:
            # pop header level
            _header_depth.reset(tok_h)
```

Notes:

* `scope_div` **does not** change `_header_depth`; it’s purely a container.
* `header(text, level=None)` should default to `level=_next_header_level()` and **not** modify `_header_depth` (inline headings don’t create a new structural level).

### D. Decorators

Support both `@scope_header_decorator` and `@scope_header_decorator("Title")`:

```python
def scope_header_decorator(title=None, **attrs):
    def _wrap(fn, title_fn):
        if asyncio.iscoroutinefunction(fn):
            @functools.wraps(fn)
            async def aw(*a, **k):
                with scope_header(title_fn(*a, **k), **attrs):
                    return await fn(*a, **k)
            return aw
        else:
            @functools.wraps(fn)
            def w(*a, **k):
                with scope_header(title_fn(*a, **k), **attrs):
                    return fn(*a, **k)
            return w

    # Bare decorator: @scope_header_decorator
    if callable(title) or title is None:
        # If "title" is actually a function, we’re being used bare.
        # Set title_fn to use the wrapped function's __name__.
        def deco(fn):
            title_fn = (title if callable(title) else (lambda *_a, **_k: fn.__name__))
            return _wrap(fn, title_fn)
        return deco

    # Parameterized: @scope_header_decorator("Title") or with lambda
    def deco_param(fn):
        title_fn = (title if callable(title) else (lambda *_a, **_k: title))
        return _wrap(fn, title_fn)
    return deco_param
```

This matches your examples and ensures title text is chosen sensibly at call time. 

### E. Content helpers & escaping

* `log_text` → `<p class="lt-p">…</p>` or `<div class="{div_class}">…</div>`; escape text.
* `log_html` → inject `Raw(html)` verbatim; wrap in `<div class="…">` when requested; **do not** escape.
* `details` → native `<details>`; put the body in `<pre>` to preserve CoT whitespace.

### F. Tables

* DataFrame path: `obj.to_html(classes="lt-table", border=0, escape=True)`.
* `list[dict]` / `list[list]`: render from headers/rows via a small `_table_html` utility with proper escaping.
* Explicit dict helpers:

  * `table_from_dict` (2‑column K/V, optional `sort_by`).
  * `table_from_dict_of_lists` (verify equal lengths; optional column `order`).
* Make `table(obj)` **reject** `Mapping` with a clear `TypeError` that points callers to the explicit helpers.

### G. Export & theming

* `Trace.head_html(theme=None, title=None, extra_head=None)` builds meta/title and styles:

  * External CSS links from `theme.css_urls`.
  * Inline CSS from `theme.css_text` or the built‑in stylesheet.
  * Append `:root { … }` overrides from `theme.css_vars`.
* `_write_trace(trace)` writes a full doc:

  ```html
  <!doctype html>
  <html lang="en">
    <head>{head_html}</head>
    {body_html}
  </html>
  ```
* `write_html_with_default_style(body_html, path, ...)` wraps `<body>…</body>` or body inner HTML.
* `jinja_context(trace, **extra)` and `render_with_jinja(env, ...)` keep templating integration simple without hard dependencies.

### H. Concurrency notes

* When you `asyncio.create_task(...)`, Python **copies the current context** into the new Task. That means the new Task sees the same active trace and current scope/header depth at the moment of creation.
* Because `_container_stack` and `_header_depth` are **task‑local**, independent Tasks won’t pop each other’s frames; structure stays valid under concurrency.
* Inter‑Task **ordering is nondeterministic** (scheduler‑dependent), which is typically acceptable for logs. If you need strict ordering, capture and merge with explicit sequencing.

