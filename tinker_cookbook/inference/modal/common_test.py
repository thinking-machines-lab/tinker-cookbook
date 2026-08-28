import pytest

pytest.importorskip("modal")

from tinker_cookbook.inference.modal import common


def test_sglang_command_builds_for_registry_models():
    for cfg in common.MODEL_REGISTRY.values():
        argv = common.sglang_command(
            model_path=common.artifact_dir("ft"), served_name="ft", tp=cfg.tp, port=30000
        )
        assert argv[:3] == ("python", "-m", "sglang.launch_server")
        assert "--model-path" in argv and "/artifacts/ft" in argv
        assert argv[argv.index("--tp") + 1] == str(cfg.tp)


def test_unknown_model_raises():
    with pytest.raises(KeyError):
        common.model_config("nope/not-a-model")
