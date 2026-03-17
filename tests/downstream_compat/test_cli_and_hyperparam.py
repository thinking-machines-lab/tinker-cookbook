"""Downstream compatibility tests for tinker_cookbook.cli_utils and hyperparam_utils.

Validates that CLI utilities and hyperparameter functions remain stable.
"""

import inspect

from tinker_cookbook.cli_utils import check_log_dir
from tinker_cookbook.hyperparam_utils import (
    get_lora_lr_over_full_finetune_lr,
    get_lora_param_count,
    get_lr,
)


class TestCliUtils:
    def test_check_log_dir_callable(self):
        assert callable(check_log_dir)

    def test_check_log_dir_signature(self):
        sig = inspect.signature(check_log_dir)
        params = list(sig.parameters.keys())
        assert "log_dir" in params


class TestHyperparamUtils:
    def test_get_lr_callable(self):
        assert callable(get_lr)

    def test_get_lora_lr_over_full_finetune_lr_callable(self):
        assert callable(get_lora_lr_over_full_finetune_lr)

    def test_get_lora_param_count_callable(self):
        assert callable(get_lora_param_count)

    def test_get_lr_returns_float(self):
        lr = get_lr("Qwen/Qwen3-8B", is_lora=True)
        assert isinstance(lr, float)
        assert lr > 0
