"""Downstream compatibility tests for tinker_cookbook.cli_utils and hyperparam_utils.

Validates that CLI utilities and hyperparameter functions remain stable.
"""

import pytest

from tinker_cookbook.cli_utils import check_log_dir
from tinker_cookbook.hyperparam_utils import (
    get_lora_lr_over_full_finetune_lr,
    get_lora_param_count,
    get_lr,
)

# Independently-measured rank=1 LoRA parameter counts for every Tinker base
# model under every valid combination of train_mlp / train_attn / train_unembed.
# Keys are (train_mlp, train_attn, train_unembed). The all-False case is
# excluded because Tinker rejects it.
#
# To refresh these values (e.g., when a new model ships), ask a Tinker team
# member to re-run the measurement script.
_REFERENCE_PARAMS_PER_RANK: dict[str, dict[tuple[bool, bool, bool], int]] = {
    "Qwen/Qwen3-235B-A22B-Instruct-2507": {
        (True, True, True): 59_931_008,
        (True, True, False): 59_774_976,
        (True, False, True): 56_754_560,
        (True, False, False): 56_598_528,
        (False, True, True): 3_332_480,
        (False, True, False): 3_176_448,
        (False, False, True): 156_032,
    },
    "Qwen/Qwen3-30B-A3B": {
        (True, True, True): 15_440_256,
        (True, True, False): 15_286_272,
        (True, False, True): 14_604_672,
        (True, False, False): 14_450_688,
        (False, True, True): 989_568,
        (False, True, False): 835_584,
        (False, False, True): 153_984,
    },
    "Qwen/Qwen3-30B-A3B-Base": {
        (True, True, True): 15_440_256,
        (True, True, False): 15_286_272,
        (True, False, True): 14_604_672,
        (True, False, False): 14_450_688,
        (False, True, True): 989_568,
        (False, True, False): 835_584,
        (False, False, True): 153_984,
    },
    "Qwen/Qwen3-30B-A3B-Instruct-2507": {
        (True, True, True): 15_440_256,
        (True, True, False): 15_286_272,
        (True, False, True): 14_604_672,
        (True, False, False): 14_450_688,
        (False, True, True): 989_568,
        (False, True, False): 835_584,
        (False, False, True): 153_984,
    },
    "Qwen/Qwen3-32B": {
        (True, True, True): 8_545_664,
        (True, True, False): 8_388_608,
        (True, False, True): 6_055_296,
        (True, False, False): 5_898_240,
        (False, True, True): 2_647_424,
        (False, True, False): 2_490_368,
        (False, False, True): 157_056,
    },
    "Qwen/Qwen3-4B-Instruct-2507": {
        (True, True, True): 2_218_880,
        (True, True, False): 2_064_384,
        (True, False, True): 1_481_600,
        (True, False, False): 1_327_104,
        (False, True, True): 891_776,
        (False, True, False): 737_280,
        (False, False, True): 154_496,
    },
    "Qwen/Qwen3-8B": {
        (True, True, True): 2_883_968,
        (True, True, False): 2_727_936,
        (True, False, True): 1_925_504,
        (True, False, False): 1_769_472,
        (False, True, True): 1_114_496,
        (False, True, False): 958_464,
        (False, False, True): 156_032,
    },
    "Qwen/Qwen3-8B-Base": {
        (True, True, True): 2_883_968,
        (True, True, False): 2_727_936,
        (True, False, True): 1_925_504,
        (True, False, False): 1_769_472,
        (False, True, True): 1_114_496,
        (False, True, False): 958_464,
        (False, False, True): 156_032,
    },
    "Qwen/Qwen3-VL-235B-A22B-Instruct": {
        (True, True, True): 59_931_008,
        (True, True, False): 59_774_976,
        (True, False, True): 56_754_560,
        (True, False, False): 56_598_528,
        (False, True, True): 3_332_480,
        (False, True, False): 3_176_448,
        (False, False, True): 156_032,
    },
    "Qwen/Qwen3-VL-30B-A3B-Instruct": {
        (True, True, True): 15_440_256,
        (True, True, False): 15_286_272,
        (True, False, True): 14_604_672,
        (True, False, False): 14_450_688,
        (False, True, True): 989_568,
        (False, True, False): 835_584,
        (False, False, True): 153_984,
    },
    "Qwen/Qwen3.5-27B": {
        (True, True, True): 7_544_320,
        (True, True, False): 7_290_880,
        (True, False, True): 4_578_816,
        (True, False, False): 4_325_376,
        (False, True, True): 3_218_944,
        (False, True, False): 2_965_504,
        (False, False, True): 253_440,
    },
    "Qwen/Qwen3.5-35B-A3B": {
        (True, True, True): 17_545_728,
        (True, True, False): 17_295_360,
        (True, False, True): 16_531_968,
        (True, False, False): 16_281_600,
        (False, True, True): 1_264_128,
        (False, True, False): 1_013_760,
        (False, False, True): 250_368,
    },
    "Qwen/Qwen3.5-397B-A17B": {
        (True, True, True): 99_124_736,
        (True, True, False): 98_872_320,
        (True, False, True): 96_283_136,
        (True, False, False): 96_030_720,
        (False, True, True): 3_094_016,
        (False, True, False): 2_841_600,
        (False, False, True): 252_416,
    },
    "Qwen/Qwen3.5-4B": {
        (True, True, True): 2_278_400,
        (True, True, False): 2_027_520,
        (True, False, True): 1_381_376,
        (True, False, False): 1_130_496,
        (False, True, True): 1_147_904,
        (False, True, False): 897_024,
        (False, False, True): 250_880,
    },
    "Qwen/Qwen3.6-27B": {
        (True, True, True): 7_544_320,
        (True, True, False): 7_290_880,
        (True, False, True): 4_578_816,
        (True, False, False): 4_325_376,
        (False, True, True): 3_218_944,
        (False, True, False): 2_965_504,
        (False, False, True): 253_440,
    },
    "Qwen/Qwen3.6-35B-A3B": {
        (True, True, True): 17_545_728,
        (True, True, False): 17_295_360,
        (True, False, True): 16_531_968,
        (True, False, False): 16_281_600,
        (False, True, True): 1_264_128,
        (False, True, False): 1_013_760,
        (False, False, True): 250_368,
    },
    "deepseek-ai/DeepSeek-V3.1": {
        (True, True, True): 96_883_776,
        (True, True, False): 96_747_328,
        (True, False, True): 94_443_776,
        (True, False, False): 94_307_328,
        (False, True, True): 2_576_448,
        (False, True, False): 2_440_000,
        (False, False, True): 136_448,
    },
    "deepseek-ai/DeepSeek-V3.1-Base": {
        (True, True, True): 96_883_776,
        (True, True, False): 96_747_328,
        (True, False, True): 94_443_776,
        (True, False, False): 94_307_328,
        (False, True, True): 2_576_448,
        (False, True, False): 2_440_000,
        (False, False, True): 136_448,
    },
    "meta-llama/Llama-3.1-70B": {
        (True, True, True): 13_079_808,
        (True, True, False): 12_943_360,
        (True, False, True): 8_983_808,
        (True, False, False): 8_847_360,
        (False, True, True): 4_232_448,
        (False, True, False): 4_096_000,
        (False, False, True): 136_448,
    },
    "meta-llama/Llama-3.1-8B": {
        (True, True, True): 2_753_792,
        (True, True, False): 2_621_440,
        (True, False, True): 1_901_824,
        (True, False, False): 1_769_472,
        (False, True, True): 984_320,
        (False, True, False): 851_968,
        (False, False, True): 132_352,
    },
    "meta-llama/Llama-3.1-8B-Instruct": {
        (True, True, True): 2_753_792,
        (True, True, False): 2_621_440,
        (True, False, True): 1_901_824,
        (True, False, False): 1_769_472,
        (False, True, True): 984_320,
        (False, True, False): 851_968,
        (False, False, True): 132_352,
    },
    "meta-llama/Llama-3.2-1B": {
        (True, True, True): 834_816,
        (True, True, False): 704_512,
        (True, False, True): 621_824,
        (True, False, False): 491_520,
        (False, True, True): 343_296,
        (False, True, False): 212_992,
        (False, False, True): 130_304,
    },
    "meta-llama/Llama-3.2-3B": {
        (True, True, True): 1_650_944,
        (True, True, False): 1_519_616,
        (True, False, True): 1_077_504,
        (True, False, False): 946_176,
        (False, True, True): 704_768,
        (False, True, False): 573_440,
        (False, False, True): 131_328,
    },
    "meta-llama/Llama-3.3-70B-Instruct": {
        (True, True, True): 13_079_808,
        (True, True, False): 12_943_360,
        (True, False, True): 8_983_808,
        (True, False, False): 8_847_360,
        (False, True, True): 4_232_448,
        (False, True, False): 4_096_000,
        (False, False, True): 136_448,
    },
    "moonshotai/Kimi-K2-Thinking": {
        (True, True, True): 146_694_976,
        (True, True, False): 146_523_968,
        (True, False, True): 144_754_688,
        (True, False, False): 144_583_680,
        (False, True, True): 2_111_296,
        (False, True, False): 1_940_288,
        (False, False, True): 171_008,
    },
    "moonshotai/Kimi-K2.5": {
        (True, True, True): 146_694_976,
        (True, True, False): 146_523_968,
        (True, False, True): 144_754_688,
        (True, False, False): 144_583_680,
        (False, True, True): 2_111_296,
        (False, True, False): 1_940_288,
        (False, False, True): 171_008,
    },
    "moonshotai/Kimi-K2.6": {
        (True, True, True): 146_694_976,
        (True, True, False): 146_523_968,
        (True, False, True): 144_754_688,
        (True, False, False): 144_583_680,
        (False, True, True): 2_111_296,
        (False, True, False): 1_940_288,
        (False, False, True): 171_008,
    },
    "nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-BF16": {
        (True, True, True): 12_064_768,
        (True, True, False): 11_931_008,
        (True, False, True): 11_479_936,
        (True, False, False): 11_346_176,
        (False, True, True): 718_592,
        (False, True, False): 584_832,
        (False, False, True): 133_760,
    },
    "nvidia/NVIDIA-Nemotron-3-Super-120B-A12B-BF16": {
        (True, True, True): 113_160_192,
        (True, True, False): 113_025_024,
        (True, False, True): 111_484_928,
        (True, False, False): 111_349_760,
        (False, True, True): 1_810_432,
        (False, True, False): 1_675_264,
        (False, False, True): 135_168,
    },
    "openai/gpt-oss-120b": {
        (True, True, True): 41_074_624,
        (True, True, False): 40_870_656,
        (True, False, True): 40_328_128,
        (True, False, False): 40_124_160,
        (False, True, True): 950_464,
        (False, True, False): 746_496,
        (False, False, True): 203_968,
    },
    "openai/gpt-oss-20b": {
        (True, True, True): 7_544_512,
        (True, True, False): 7_340_544,
        (True, False, True): 7_046_848,
        (True, False, False): 6_842_880,
        (False, True, True): 701_632,
        (False, True, False): 497_664,
        (False, False, True): 203_968,
    },
}

_TEST_RANKS = (1, 2, 4, 8, 16, 32, 64)


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
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(
            get_lora_param_count,
            ["model_name", "lora_rank", "train_mlp", "train_attn", "train_unembed"],
        )

    def test_get_lr_returns_float(self):
        lr = get_lr("Qwen/Qwen3.6-27B", is_lora=True)
        assert isinstance(lr, float)
        assert lr > 0

    def test_get_lora_param_count_rejects_all_false(self):
        with pytest.raises(ValueError):
            get_lora_param_count(
                "Qwen/Qwen3.6-27B",
                lora_rank=32,
                train_mlp=False,
                train_attn=False,
                train_unembed=False,
            )

    @pytest.mark.parametrize(
        "flag_combo",
        sorted(
            {combo for params in _REFERENCE_PARAMS_PER_RANK.values() for combo in params},
            reverse=True,
        ),
    )
    @pytest.mark.parametrize("model_name", sorted(_REFERENCE_PARAMS_PER_RANK.keys()))
    def test_get_lora_param_count_matches_measurements(
        self, model_name: str, flag_combo: tuple[bool, bool, bool]
    ) -> None:
        """Function output must equal the measured rank=1 value times the rank,
        for every (model, train_mlp, train_attn, train_unembed) combination.
        """
        train_mlp, train_attn, train_unembed = flag_combo
        params_at_rank_1 = _REFERENCE_PARAMS_PER_RANK[model_name][flag_combo]
        for rank in _TEST_RANKS:
            expected = params_at_rank_1 * rank
            actual = get_lora_param_count(
                model_name,
                lora_rank=rank,
                train_mlp=train_mlp,
                train_attn=train_attn,
                train_unembed=train_unembed,
            )
            assert actual == expected, (
                f"{model_name} rank={rank} "
                f"(mlp={train_mlp}, attn={train_attn}, unembed={train_unembed}): "
                f"expected {expected:,}, got {actual:,} (off by {actual - expected:+,})"
            )
