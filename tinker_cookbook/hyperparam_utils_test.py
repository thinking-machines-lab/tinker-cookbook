import warnings

import pytest

from tinker_cookbook import hyperparam_utils
from tinker_cookbook.exceptions import ConfigurationError

TABLE = hyperparam_utils._lr_table()
MODELS = sorted(TABLE["multipliers"])
REFERENCE_MODEL = "Qwen/Qwen3-8B"


def test_runtime_table_holds_only_the_reference_lr_and_multipliers():
    assert set(TABLE) == {"reference_lr", "multipliers"}
    assert [m for m in MODELS if TABLE["multipliers"][m] == 1.0] == [REFERENCE_MODEL]
    for name in MODELS:
        assert hyperparam_utils.get_lora_lr_multiplier(name) == TABLE["multipliers"][name]


def test_get_lr_is_reference_lr_times_multiplier():
    assert hyperparam_utils.get_lr(REFERENCE_MODEL) == TABLE["reference_lr"]
    for name in MODELS:
        expected = TABLE["reference_lr"] * hyperparam_utils.get_lora_lr_multiplier(name)
        assert hyperparam_utils.get_lr(name) == pytest.approx(expected)


def test_transfer_lr_scales_by_the_multiplier_ratio():
    ref = REFERENCE_MODEL
    other = next(m for m in MODELS if hyperparam_utils.get_lora_lr_multiplier(m) != 1.0)
    mult = hyperparam_utils.get_lora_lr_multiplier(other)
    assert hyperparam_utils.transfer_lr(1e-3, ref, other) == pytest.approx(1e-3 * mult)
    assert hyperparam_utils.transfer_lr(1e-3, other, ref) == pytest.approx(1e-3 / mult)


def test_transfer_lr_round_trip():
    a, b = MODELS[0], MODELS[-1]
    lr = hyperparam_utils.transfer_lr(3e-4, a, b)
    assert hyperparam_utils.transfer_lr(lr, b, a) == pytest.approx(3e-4)
    assert hyperparam_utils.transfer_lr(3e-4, a, a) == pytest.approx(3e-4)


def test_model_variants():
    name = MODELS[0]
    assert hyperparam_utils.get_lr(name + ":peft:262144") == hyperparam_utils.get_lr(name)
    assert hyperparam_utils.get_lr(name + ":peft:262144:inference") == hyperparam_utils.get_lr(name)
    with pytest.raises(ConfigurationError, match="only LoRA"):
        hyperparam_utils.get_lr(name + ":full:65536")
    with pytest.raises(ConfigurationError, match="No calibrated learning rate"):
        hyperparam_utils.get_lr("example-org/not-a-model")


def test_is_lora_is_deprecated():
    name = MODELS[0]
    with pytest.warns(DeprecationWarning):
        assert hyperparam_utils.get_lr(name, is_lora=True) == hyperparam_utils.get_lr(name)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        with pytest.raises(ConfigurationError, match="Full fine-tuning"):
            hyperparam_utils.get_lr(name, is_lora=False)
