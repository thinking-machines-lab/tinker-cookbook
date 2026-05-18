"""Tests for picklability of Renderers.

Renderers created via get_renderer() must survive pickle roundtrips so that
EnvGroupBuilder instances (which often hold Renderer references) can be
serialized for distributed rollout execution.
"""

import pickle

import pytest

from tinker_cookbook.renderers import get_renderer, register_renderer, unregister_renderer
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.tokenizer_utils import Tokenizer, get_tokenizer

# Models that don't require special access / are commonly available in CI.
# Each entry is (renderer_name, model_name).
_TEXT_RENDERERS = [
    ("role_colon", "Qwen/Qwen3.5-9B-Base"),
    ("llama3", "meta-llama/Llama-3.2-1B-Instruct"),
    ("qwen3", "Qwen/Qwen3-8B"),
    ("qwen3_disable_thinking", "Qwen/Qwen3-8B"),
    ("qwen3_instruct", "Qwen/Qwen3-8B"),
    ("deepseekv3", "deepseek-ai/DeepSeek-V3-0324"),
    ("deepseekv3_disable_thinking", "deepseek-ai/DeepSeek-V3-0324"),
    ("deepseekv3_thinking", "deepseek-ai/DeepSeek-V3-0324"),
]


@pytest.fixture(params=_TEXT_RENDERERS, ids=[r[0] for r in _TEXT_RENDERERS])
def renderer_and_model(request: pytest.FixtureRequest) -> tuple[str, str]:
    return request.param


class TestRendererPickle:
    def test_pickle_roundtrip(self, renderer_and_model: tuple[str, str]) -> None:
        """Renderers created via get_renderer() survive pickle roundtrip."""
        renderer_name, model_name = renderer_and_model
        tokenizer = get_tokenizer(model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        restored = pickle.loads(pickle.dumps(renderer))

        assert restored._renderer_name == renderer_name
        assert restored._model_name == renderer._model_name
        assert type(restored) is type(renderer)
        assert restored.get_stop_sequences() == renderer.get_stop_sequences()

    def test_pickle_metadata_set(self, renderer_and_model: tuple[str, str]) -> None:
        """get_renderer() stamps _renderer_name and _model_name."""
        renderer_name, model_name = renderer_and_model
        tokenizer = get_tokenizer(model_name)
        renderer = get_renderer(renderer_name, tokenizer)

        assert renderer._renderer_name == renderer_name
        # _model_name may differ from model_name due to tokenizer remapping (e.g., Llama 3)
        assert renderer._model_name == tokenizer.name_or_path

    def test_pickle_without_metadata_raises(self) -> None:
        """Renderers created directly (not via get_renderer()) raise on pickle."""
        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

        from tinker_cookbook.renderers.llama3 import Llama3Renderer

        renderer = Llama3Renderer(tokenizer)
        # _renderer_name and _model_name are None
        with pytest.raises(pickle.PicklingError, match="not set"):
            pickle.dumps(renderer)

    def test_pickle_with_manual_metadata(self) -> None:
        """Manually setting pickle metadata works for direct-constructed renderers."""
        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")

        from tinker_cookbook.renderers.llama3 import Llama3Renderer

        renderer = Llama3Renderer(tokenizer)
        renderer._renderer_name = "llama3"
        renderer._model_name = "meta-llama/Llama-3.2-1B-Instruct"
        renderer._has_image_processor = False

        restored = pickle.loads(pickle.dumps(renderer))
        assert type(restored) is Llama3Renderer
        assert restored.get_stop_sequences() == renderer.get_stop_sequences()

    def test_pickle_without_metadata_vl_renderer(self) -> None:
        """VL renderers that bypass super().__init__() still raise clean PicklingError."""
        tokenizer = get_tokenizer("Qwen/Qwen3-8B")

        from tinker_cookbook.renderers.qwen3 import Qwen3VLRenderer

        # Qwen3VLRenderer bypasses super().__init__(), so _renderer_name is never
        # set via __init__. Class-level defaults + getattr in __reduce__ handle this.
        renderer = Qwen3VLRenderer(tokenizer, image_processor=None)
        with pytest.raises(pickle.PicklingError, match="not set"):
            pickle.dumps(renderer)

    def test_pickle_with_explicit_model_name(self) -> None:
        """The model_name param in get_renderer() overrides tokenizer.name_or_path."""
        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        # tokenizer.name_or_path is remapped, but we can override it
        renderer = get_renderer("llama3", tokenizer, model_name="meta-llama/Llama-3.2-1B-Instruct")

        assert renderer._model_name == "meta-llama/Llama-3.2-1B-Instruct"

        restored = pickle.loads(pickle.dumps(renderer))
        assert restored._model_name == "meta-llama/Llama-3.2-1B-Instruct"
        assert type(restored) is type(renderer)

    def test_pickle_custom_registered_renderer(self) -> None:
        """Custom renderers registered via register_renderer() are pickle-safe."""
        from tinker_cookbook.renderers.role_colon import RoleColonRenderer

        def my_factory(tokenizer: Tokenizer, image_processor: object = None) -> Renderer:
            return RoleColonRenderer(tokenizer)

        register_renderer("test_custom_pickle", my_factory)
        try:
            tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
            renderer = get_renderer("test_custom_pickle", tokenizer)

            assert renderer._renderer_name == "test_custom_pickle"

            restored = pickle.loads(pickle.dumps(renderer))
            assert type(restored) is RoleColonRenderer
            assert restored._renderer_name == "test_custom_pickle"
        finally:
            unregister_renderer("test_custom_pickle")


class TestMessageCompleterPickle:
    def test_tinker_message_completer_pickle_structure(self) -> None:
        """TinkerMessageCompleter fields are individually pickleable (Renderer + SamplingClient).

        We test the Renderer part here; SamplingClient has its own __reduce__ in the SDK.
        """
        tokenizer = get_tokenizer("meta-llama/Llama-3.2-1B-Instruct")
        renderer = get_renderer("llama3", tokenizer)

        # Just verify the renderer component pickles fine when it would be inside a completer
        restored_renderer = pickle.loads(pickle.dumps(renderer))
        assert type(restored_renderer) is type(renderer)
        assert restored_renderer.get_stop_sequences() == renderer.get_stop_sequences()
