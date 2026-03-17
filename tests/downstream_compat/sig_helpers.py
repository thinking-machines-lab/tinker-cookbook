"""Helpers for checking function/method signatures in downstream compat tests."""

import inspect


def get_param_names(func) -> list[str]:
    """Return parameter names (excluding 'self') for a function or method."""
    sig = inspect.signature(func)
    return [name for name in sig.parameters if name != "self"]


def assert_params(func, expected_params: list[str]) -> None:
    """Assert that a function has exactly the expected parameter names (excluding 'self')."""
    actual = get_param_names(func)
    assert actual == expected_params, (
        f"{func.__qualname__}: expected params {expected_params}, got {actual}"
    )


def assert_params_subset(func, required_params: list[str]) -> None:
    """Assert that a function has at least the required parameter names (in order)."""
    actual = get_param_names(func)
    for param in required_params:
        assert param in actual, (
            f"{func.__qualname__}: missing required param '{param}', has {actual}"
        )
