import pytest

pytest.importorskip("modal")

from tinker_cookbook.inference.modal import common


def test_registry_resolves_and_hybrids_force_merge():
    for cfg in common.MODEL_REGISTRY.values():
        mode = common.resolve_mode(cfg, None)
        assert mode == ("adapter" if cfg.lora_serving else "merge")
        if not cfg.lora_serving:
            # asking for adapter on a model SGLang can't LoRA-serve must fail fast
            with pytest.raises(ValueError):
                common.resolve_mode(cfg, "adapter")
