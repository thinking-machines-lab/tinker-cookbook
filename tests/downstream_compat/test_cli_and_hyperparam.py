"""Downstream compatibility tests for tinker_cookbook.cli_utils and hyperparam_utils.

Validates that CLI utilities and hyperparameter functions remain stable.
"""

from tinker_cookbook.cli_utils import check_log_dir
from tinker_cookbook.hyperparam_utils import (
    get_lora_lr_over_full_finetune_lr,
    get_lora_param_count,
    get_lr,
)


class TestCliUtils:
    def test_check_log_dir_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(check_log_dir, ["log_dir", "behavior_if_exists"])


class TestHyperparamUtils:
    def test_get_lr_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(get_lr, ["model_name", "is_lora"])

    def test_get_lora_lr_over_full_finetune_lr_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(get_lora_lr_over_full_finetune_lr, ["model_name", "lora_alpha"])

    def test_get_lora_param_count_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params_subset

        assert_params_subset(get_lora_param_count, ["model_name", "lora_rank"])

    def test_get_lr_returns_float(self):
        lr = get_lr("Qwen/Qwen3-8B", is_lora=True)
        assert isinstance(lr, float)
        assert lr > 0
