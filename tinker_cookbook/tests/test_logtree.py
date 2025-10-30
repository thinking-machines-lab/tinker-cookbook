"""Tests for the logtree module."""

import asyncio
import os
import tempfile
from pathlib import Path

from tinker_cookbook.utils import logtree


def test_basic_trace():
    """Test basic trace creation and HTML generation."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "test.html"

        with logtree.init_trace("Test Report", path=output_path):
            logtree.log_text("Hello world")
            with logtree.scope_header("Section 1"):
                logtree.log_text("Content in section 1")

        assert output_path.exists()
        content = output_path.read_text()

        # Check for expected elements
        assert "<title>Test Report</title>" in content
        assert "<h1" in content and "Test Report" in content
        assert "Hello world" in content
        assert "Section 1" in content
        assert "Content in section 1" in content


def test_nested_scopes():
    """Test nested scopes and auto header levels."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "nested.html"

        with logtree.init_trace("Nested Test", path=output_path):
            with logtree.scope_header("Level 1"):
                logtree.log_text("At level 1")
                with logtree.scope_header("Level 2"):
                    logtree.log_text("At level 2")
                    with logtree.scope_header("Level 3"):
                        logtree.log_text("At level 3")

        content = output_path.read_text()

        # Check that we have h1 (title), h2, h3, h4
        assert "<h1" in content
        assert "<h2" in content
        assert "<h3" in content
        assert "<h4" in content


def test_conditional_logging():
    """Test scope_disable for conditional logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "conditional.html"

        with logtree.init_trace("Conditional Test", path=output_path):
            for i in range(5):
                # Only log groups 0 and 2
                with logtree.scope_header(f"Group {i}") if i in {0, 2} else logtree.scope_disable():
                    logtree.log_text(f"Content for group {i}")

        content = output_path.read_text()

        # Check that groups 0 and 2 are present
        assert "Group 0" in content
        assert "Content for group 0" in content
        assert "Group 2" in content
        assert "Content for group 2" in content

        # Check that groups 1, 3, 4 are not present
        assert "Group 1" not in content
        assert "Group 3" not in content
        assert "Group 4" not in content


def test_table_rendering():
    """Test various table rendering functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "tables.html"

        with logtree.init_trace("Table Test", path=output_path):
            # Test table_from_dict
            logtree.table_from_dict({"lr": 0.001, "batch_size": 32}, caption="Hyperparams")

            # Test table from list of dicts
            logtree.table([{"name": "Alice", "score": 95}, {"name": "Bob", "score": 87}])

            # Test table_from_dict_of_lists
            logtree.table_from_dict_of_lists(
                {"name": ["Charlie", "Diana"], "score": [92, 88]}, caption="Results"
            )

        content = output_path.read_text()

        assert "Hyperparams" in content
        assert "0.001" in content
        assert "batch_size" in content
        assert "Alice" in content
        assert "Bob" in content
        assert "Charlie" in content
        assert "Results" in content


def test_html_content():
    """Test log_html for raw HTML insertion."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "html.html"

        with logtree.init_trace("HTML Test", path=output_path):
            logtree.log_html("<strong>Bold text</strong>")
            logtree.log_html("<em>Italic</em>", div_class="emphasis")

        content = output_path.read_text()

        assert "<strong>Bold text</strong>" in content
        assert "<em>Italic</em>" in content
        assert 'class="emphasis"' in content


def test_details():
    """Test collapsible details blocks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "details.html"

        with logtree.init_trace("Details Test", path=output_path):
            logtree.details("This is a long\nmultiline\ntext", summary="Click to expand")

        content = output_path.read_text()

        assert "<details" in content
        assert "<summary" in content
        assert "Click to expand" in content
        assert "long" in content and "multiline" in content


async def async_test_async_safety():
    """Test that logtree is async-safe with concurrent tasks."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "async.html"

        async def worker(task_id: int):
            with logtree.scope_header(f"Task {task_id}"):
                logtree.log_text(f"Started task {task_id}")
                await asyncio.sleep(0.01)
                logtree.log_text(f"Finished task {task_id}")

        with logtree.init_trace("Async Test", path=output_path):
            await asyncio.gather(*[worker(i) for i in range(5)])

        content = output_path.read_text()

        # Check that all tasks are logged
        for i in range(5):
            assert f"Task {i}" in content
            assert f"Started task {i}" in content
            assert f"Finished task {i}" in content


def test_async_safety():
    """Wrapper to run async test."""
    asyncio.run(async_test_async_safety())


def test_scope_header_decorator():
    """Test the scope_header_decorator."""

    @logtree.scope_header_decorator
    def simple_function():
        logtree.log_text("Inside simple function")

    @logtree.scope_header_decorator("Custom Title")
    def custom_title_function():
        logtree.log_text("Inside custom title function")

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "decorator.html"

        with logtree.init_trace("Decorator Test", path=output_path):
            simple_function()
            custom_title_function()

        content = output_path.read_text()

        assert "simple_function" in content
        assert "Inside simple function" in content
        assert "Custom Title" in content
        assert "Inside custom title function" in content


async def async_test_scope_header_decorator():
    """Test scope_header_decorator with async functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "async_decorator.html"

        @logtree.scope_header_decorator("Async Work")
        async def async_work(value: int):
            logtree.log_text(f"Working on {value}")
            await asyncio.sleep(0.01)
            logtree.log_text(f"Done with {value}")

        with logtree.init_trace("Async Decorator Test", path=output_path):
            await async_work(123)

        content = output_path.read_text()

        assert "Async Work" in content
        assert "Working on 123" in content
        assert "Done with 123" in content


def test_async_decorator():
    """Wrapper to run async decorator test."""
    asyncio.run(async_test_scope_header_decorator())


def test_error_handling():
    """Test that traces are written even on error when write_on_error=True."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "error.html"

        try:
            with logtree.init_trace("Error Test", path=output_path, write_on_error=True):
                logtree.log_text("Before error")
                raise ValueError("Test error")
        except ValueError:
            pass

        assert output_path.exists()
        content = output_path.read_text()

        assert "Before error" in content
        assert "Exception" in content
        assert "ValueError" in content
        assert "Test error" in content


def test_no_write_without_path():
    """Test that no file is written when path=None."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Change to tmpdir to ensure no files are created
        original_cwd = os.getcwd()
        try:
            os.chdir(tmpdir)

            with logtree.init_trace("No Write Test", path=None) as trace:
                logtree.log_text("This should not be written to disk")
                body_html = trace.body_html()

            # Check that body_html contains the content
            assert "This should not be written to disk" in body_html

            # Check that no HTML files were created
            html_files = list(Path(tmpdir).glob("*.html"))
            assert len(html_files) == 0

        finally:
            os.chdir(original_cwd)


def test_scope_div():
    """Test scope_div for wrapping content without changing header level."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "div.html"

        with logtree.init_trace("Div Test", path=output_path):
            with logtree.scope_header("Section"):
                with logtree.scope_div(class_="custom-div"):
                    logtree.log_text("Inside custom div")
                    logtree.header("Inline header")

        content = output_path.read_text()

        assert 'class="custom-div"' in content
        assert "Inside custom div" in content


def test_inline_header():
    """Test inline header function."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "inline_header.html"

        with logtree.init_trace("Inline Header Test", path=output_path):
            logtree.header("First Header")
            logtree.log_text("Some content")
            logtree.header("Second Header", level=3)

        content = output_path.read_text()

        assert "First Header" in content
        assert "Second Header" in content
        assert "Some content" in content


def test_div_class_parameter():
    """Test div_class parameter for log_text and log_html."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "div_class.html"

        with logtree.init_trace("Div Class Test", path=output_path):
            logtree.log_text("Answer: A", div_class="answer")
            logtree.log_text("Reward: 0.95", div_class="reward")

        content = output_path.read_text()

        assert 'class="answer"' in content
        assert 'class="reward"' in content
        assert "Answer: A" in content
        assert "Reward: 0.95" in content


def test_export_helpers():
    """Test export helper functions."""
    with tempfile.TemporaryDirectory() as tmpdir:
        # Test write_html_with_default_style
        output_path = Path(tmpdir) / "export.html"
        body = "<p>Test content</p>"

        logtree.write_html_with_default_style(body, output_path, title="Export Test")

        assert output_path.exists()
        content = output_path.read_text()

        assert "<title>Export Test</title>" in content
        assert "Test content" in content
        assert "<!doctype html>" in content.lower()


def test_graceful_degradation():
    """Test that logtree functions work gracefully when no trace is active."""

    # All these should work without error when no trace is active
    logtree.log_text("This should not crash")
    logtree.log_html("<p>This should not crash</p>")
    logtree.header("This should not crash")
    logtree.details("This should not crash")
    logtree.table([{"a": 1}])

    with logtree.scope_header("This should not crash"):
        logtree.log_text("Nested content")

    with logtree.scope_div():
        logtree.log_text("Div content")

    @logtree.scope_header_decorator
    def decorated_func():
        logtree.log_text("Should not crash")

    decorated_func()


async def async_test_graceful_degradation_decorator():
    """Test that decorated async functions work without trace."""

    @logtree.scope_header_decorator("Async Work")
    async def async_work():
        logtree.log_text("Should not crash")
        await asyncio.sleep(0.001)

    # Should work without error
    await async_work()


def test_graceful_degradation_async():
    """Wrapper for async graceful degradation test."""
    asyncio.run(async_test_graceful_degradation_decorator())


def test_formatter():
    """Test the formatter object API."""
    from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "formatter.html"

        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
            {"role": "user", "content": "How are you?"},
        ]

        with logtree.init_trace("Formatter Test", path=output_path):
            logtree.log_formatter(ConversationFormatter(messages=messages))

        content = output_path.read_text()

        # Check that messages are present
        assert "Hello" in content
        assert "Hi there!" in content
        assert "How are you?" in content

        # Check that CSS is included
        assert "lt-conversation" in content
        assert "lt-message" in content
        assert "lt-message-role" in content


def test_formatter_css_deduplication():
    """Test that formatter CSS is deduplicated per trace."""
    from tinker_cookbook.utils.logtree_formatters import ConversationFormatter

    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "dedup.html"

        messages1 = [{"role": "user", "content": "Message 1"}]
        messages2 = [{"role": "assistant", "content": "Message 2"}]
        messages3 = [{"role": "user", "content": "Message 3"}]

        with logtree.init_trace("Dedup Test", path=output_path):
            # Log three conversation formatters
            logtree.log_formatter(ConversationFormatter(messages=messages1))
            logtree.log_formatter(ConversationFormatter(messages=messages2))
            logtree.log_formatter(ConversationFormatter(messages=messages3))

        content = output_path.read_text()

        # CSS should appear only once
        css_count = content.count(".lt-conversation {")
        assert css_count == 1, f"Expected CSS to appear once, but appeared {css_count} times"

        # All messages should be present
        assert "Message 1" in content
        assert "Message 2" in content
        assert "Message 3" in content


def test_scope_details():
    """Test collapsible details scope."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "scope_details.html"

        with logtree.init_trace("Scope Details Test", path=output_path):
            logtree.log_text("Before details")
            with logtree.scope_details("Click to expand"):
                logtree.log_text("Hidden content 1")
                logtree.log_text("Hidden content 2")
            logtree.log_text("After details")

        content = output_path.read_text()

        assert "<details" in content
        assert "<summary" in content
        assert "Click to expand" in content
        assert "Hidden content 1" in content
        assert "Hidden content 2" in content
        assert "Before details" in content
        assert "After details" in content


def test_scope_disable_nested():
    """Test that scope_disable actually disables nested logging."""
    with tempfile.TemporaryDirectory() as tmpdir:
        output_path = Path(tmpdir) / "scope_disable.html"

        with logtree.init_trace("Scope Disable Test", path=output_path):
            logtree.log_text("Before disabled scope")

            # This entire block should not be logged
            with logtree.scope_disable():
                logtree.log_text("This should NOT appear")
                with logtree.scope_header("Nested Header"):
                    logtree.log_text("This should also NOT appear")
                logtree.log_text("Still should NOT appear")

            logtree.log_text("After disabled scope")

        content = output_path.read_text()

        # Check that logged content is present
        assert "Before disabled scope" in content
        assert "After disabled scope" in content

        # Check that disabled content is NOT present
        assert "This should NOT appear" not in content
        assert "Nested Header" not in content
        assert "This should also NOT appear" not in content
        assert "Still should NOT appear" not in content


if __name__ == "__main__":
    # Run tests
    test_basic_trace()
    test_nested_scopes()
    test_conditional_logging()
    test_table_rendering()
    test_html_content()
    test_details()
    test_async_safety()
    test_scope_header_decorator()
    test_async_decorator()
    test_error_handling()
    test_no_write_without_path()
    test_scope_div()
    test_inline_header()
    test_div_class_parameter()
    test_export_helpers()
    test_graceful_degradation()
    test_graceful_degradation_async()
    test_formatter()
    test_formatter_css_deduplication()
    test_scope_details()
    test_scope_disable_nested()

    print("All tests passed!")
