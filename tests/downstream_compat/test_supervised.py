"""Downstream compatibility tests for tinker_cookbook.supervised.

Validates that supervised training types and data utilities remain stable.
"""

from tinker_cookbook.supervised.data import (
    FromConversationFileBuilder,
    StreamingSupervisedDatasetFromHFDataset,
    SupervisedDatasetFromHFDataset,
    conversation_to_datum,
)
from tinker_cookbook.supervised.types import (
    ChatDatasetBuilder,
    ChatDatasetBuilderCommonConfig,
    SupervisedDataset,
    SupervisedDatasetBuilder,
)


class TestSupervisedTypes:
    def test_supervised_dataset_has_get_batch(self):
        assert hasattr(SupervisedDataset, "get_batch")

    def test_supervised_dataset_has_len(self):
        assert hasattr(SupervisedDataset, "__len__")

    def test_supervised_dataset_builder_is_callable(self):
        assert callable(SupervisedDatasetBuilder)

    def test_chat_dataset_builder_is_subclass(self):
        assert issubclass(ChatDatasetBuilder, SupervisedDatasetBuilder)

    def test_chat_dataset_builder_common_config_exists(self):
        assert ChatDatasetBuilderCommonConfig is not None


class TestSupervisedData:
    def test_conversation_to_datum_signature(self):
        from tests.downstream_compat.sig_helpers import assert_params

        assert_params(
            conversation_to_datum,
            ["conversation", "renderer", "max_length", "train_on_what", "reduction"],
        )

    def test_from_conversation_file_builder_exists(self):
        assert FromConversationFileBuilder is not None

    def test_supervised_dataset_from_hf_exists(self):
        assert SupervisedDatasetFromHFDataset is not None

    def test_streaming_supervised_dataset_from_hf_exists(self):
        assert StreamingSupervisedDatasetFromHFDataset is not None
