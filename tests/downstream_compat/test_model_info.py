"""Downstream compatibility tests for tinker_cookbook.model_info.

Validates that model metadata functions and ModelAttributes remain stable.
"""

from dataclasses import fields

from tinker_cookbook.model_info import (
    ModelAttributes,
    get_model_attributes,
    get_recommended_renderer_name,
    get_recommended_renderer_names,
)


class TestModelAttributes:
    def test_fields(self):
        names = {f.name for f in fields(ModelAttributes)}
        expected = {
            "organization",
            "version_str",
            "size_str",
            "is_chat",
            "recommended_renderers",
            "is_vl",
        }
        assert expected.issubset(names)

    def test_constructable(self):
        attrs = ModelAttributes(
            organization="test-org",
            version_str="1.0",
            size_str="8B",
            is_chat=True,
            recommended_renderers=("qwen3",),
        )
        assert attrs.organization == "test-org"
        assert attrs.is_vl is False  # default


class TestModelInfoFunctions:
    def test_get_model_attributes_returns_model_attributes(self):
        # Use a well-known model name
        attrs = get_model_attributes("Qwen/Qwen3-8B")
        assert isinstance(attrs, ModelAttributes)
        assert attrs.organization == "Qwen"

    def test_get_recommended_renderer_name_returns_string(self):
        name = get_recommended_renderer_name("Qwen/Qwen3-8B")
        assert isinstance(name, str)
        assert len(name) > 0

    def test_get_recommended_renderer_names_returns_list(self):
        names = get_recommended_renderer_names("Qwen/Qwen3-8B")
        assert isinstance(names, list)
        assert all(isinstance(n, str) for n in names)
        assert len(names) > 0

    def test_recommended_renderer_name_is_first_of_names(self):
        name = get_recommended_renderer_name("Qwen/Qwen3-8B")
        names = get_recommended_renderer_names("Qwen/Qwen3-8B")
        assert name == names[0]
