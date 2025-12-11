"""
Datasets for supervised learning (SFT) that use chat-formatted data, which we
convert to tokens using a Renderer.
"""

import logging
from typing import Any, cast

import random
import torch
import math
import io
import chz
import datasets
import tinker
from PIL import Image
from collections import defaultdict
from tinker_cookbook.supervised.common import datum_from_model_input_weights
from tinker_cookbook.supervised.types import SupervisedDatasetBuilder, SupervisedDataset
from tinker_cookbook.tokenizer_utils import get_tokenizer
from tinker_cookbook.image_processing_utils import get_image_processor, resize_image
from tinker_cookbook.renderers import (
    Message,
    ContentPart,
    ImagePart,
    TextPart,
    TrainOnWhat,
    get_renderer,
)

logger = logging.getLogger(__name__)


@chz.chz
class ClassifierDatasetConfig:
    """
    Configuration for a classification dataset.
    """

    dataset: str
    dataset_split: str

    image_column_name: str = "image"
    label_column_name: str = "label"

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    train_on_what: TrainOnWhat = TrainOnWhat.LAST_ASSISTANT_MESSAGE

    # If set, sample only this many examples per class (for few-shot experiments)
    examples_per_class: int | None = None
    subset_seed: int = 0

    max_image_size: int = 480
    hflip_probability: float = 0.5


class ClassifierDataset(SupervisedDataset):
    def __init__(self, config: ClassifierDatasetConfig):
        """
        Construct a VLM classifier dataset with the provided data config.
        """

        self.config = config

        tokenizer = get_tokenizer(self.config.model_name_for_tokenizer)
        image_processor = get_image_processor(self.config.model_name_for_tokenizer)

        self.renderer = get_renderer(
            name=self.config.renderer_name, tokenizer=tokenizer, image_processor=image_processor
        )

        dataset = datasets.load_dataset(self.config.dataset)
        dataset = cast(datasets.DatasetDict, dataset)
        self.dataset = dataset[self.config.dataset_split]

        # If examples_per_class is set, sample N examples per class for few-shot setting
        if self.config.examples_per_class is not None:
            self.dataset = self._sample_per_class(self.dataset)

        self.class_labels = self.dataset.features[self.config.label_column_name]
        self.shuffled_indices = self.get_shuffled_indices()

    def get_shuffled_indices(self, seed: int = 0) -> list[int]:
        """
        Get a shuffled set of dataset indices with a target number of num_repeats.
        """

        max_repeat = int(math.ceil(self.config.num_repeats))
        max_examples = int(math.ceil(self.config.num_repeats * len(self.dataset)))

        random_gen = random.Random(seed)
        shuffled_indices: list[int] = []

        for _ in range(max_repeat):
            dataset_indices = list(range(len(self.dataset)))
            random_gen.shuffle(dataset_indices)
            shuffled_indices.extend(dataset_indices)

        return shuffled_indices[:max_examples]

    def _sample_per_class(self, dataset: datasets.Dataset) -> datasets.Dataset:
        """
        Sample up to N examples per class from the dataset for few-shot experiments.
        Uses self.config.examples_per_class, label_column_name, and subset_seed.
        """
        rng = random.Random(self.config.subset_seed)

        # Group indices by class label
        class_indices: dict[int, list[int]] = defaultdict(list)
        for idx, label in enumerate(dataset[self.config.label_column_name]):
            class_indices[label].append(idx)

        # Shuffle and sample up to examples_per_class from each class
        selected_indices: list[int] = []
        for label in sorted(class_indices.keys()):
            indices = class_indices[label]
            rng.shuffle(indices)

            selected_indices.extend(indices[: self.config.examples_per_class])

        logger.info(
            f"Sampled {len(selected_indices)} examples "
            f"({self.config.examples_per_class} per class, {len(class_indices)} classes)"
        )

        return dataset.select(selected_indices)

    def get_class_name(self, label: str) -> str:
        """
        Helper function to clean up the original class name.
        """

        return label.replace("_", " ").replace(".", " ").replace("-", " ").lower()

    def build_supervised_example(
        self,
        example: dict[str, Any],
    ) -> tuple[tinker.ModelInput, torch.Tensor]:
        """
        Generate an input to prompt the model.
        """

        class_label = example[self.config.label_column_name]
        class_label_name = self.get_class_name(self.class_labels.int2str(class_label))

        image = example[self.config.image_column_name]
        pil_image: Image.Image | None = None

        if isinstance(image, dict) and "bytes" in image:
            pil_image = Image.open(io.BytesIO(image["bytes"]))

        elif isinstance(image, Image.Image):
            pil_image = cast(Image.Image, image)

        # If the dataset cannot be loaded
        if pil_image is None:
            raise AssertionError(f"Unable to interpret {image} as an image")

        pil_image = resize_image(image=pil_image, max_size=self.config.max_image_size)

        # horizontal flip 50% of the time
        if random.random() < self.config.hflip_probability:
            pil_image = pil_image.transpose(Image.Transpose.FLIP_LEFT_RIGHT)

        user_parts: list[ContentPart] = [
            ImagePart(type="image", image=pil_image),
            TextPart(type="text", text="What is the name of the subject in this photo?"),
        ]

        assistant_parts: list[ContentPart] = [
            TextPart(type="text", text=f"The subject in this photo is: {class_label_name}\n"),
        ]

        messages = [
            Message(role="user", content=user_parts),
            Message(role="assistant", content=assistant_parts),
        ]

        return self.renderer.build_supervised_example(
            messages=messages,
            train_on_what=self.config.train_on_what,
        )

    def get_batch(self, index: int) -> list[tinker.Datum]:
        """
        Load a batch of training examples.
        """

        return [
            datum_from_model_input_weights(
                *self.build_supervised_example(self.dataset[self.shuffled_indices[idx]]),
                max_length=self.config.max_length,
            )
            for idx in range(
                self.config.batch_size * index,
                min(self.config.batch_size * (index + 1), len(self.shuffled_indices)),
            )
        ]

    def __len__(self) -> int:
        """
        Number of batches in the dataloader
        """

        return int(math.ceil(len(self.shuffled_indices) / self.config.batch_size))

    def set_epoch(self, seed: int = 0):
        """
        Set the epoch for shuffling the dataloader.
        """

        self.shuffled_indices = self.get_shuffled_indices(seed=seed)


@chz.chz
class Caltech101DatasetBuilder(SupervisedDatasetBuilder):
    """
    Configuration for a classification dataset.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    train_on_what: TrainOnWhat | None = None

    # If set, sample only this many examples per class (for few-shot experiments)
    examples_per_class: int | None = None
    subset_seed: int = 0

    max_image_size: int = 480

    run_nll_evaluator: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        default_train_on_what = self.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE

        train_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/caltech101",
            dataset_split="train",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            num_repeats=self.num_repeats,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            examples_per_class=self.examples_per_class,
            subset_seed=self.subset_seed,
            max_image_size=self.max_image_size,
            hflip_probability=0.5,
        )

        test_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/caltech101",
            dataset_split="test",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            max_image_size=self.max_image_size,
            hflip_probability=0.0,  # No augmentation for test set
            # Note: test set uses full data, no few-shot sampling
        )

        train_dataset = ClassifierDataset(train_config)

        if not self.run_nll_evaluator:
            return train_dataset, None

        return train_dataset, ClassifierDataset(test_config)


@chz.chz
class Flowers102DatasetBuilder(SupervisedDatasetBuilder):
    """
    Configuration for a classification dataset.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    train_on_what: TrainOnWhat | None = None

    # If set, sample only this many examples per class (for few-shot experiments)
    examples_per_class: int | None = None
    subset_seed: int = 0

    max_image_size: int = 480

    run_nll_evaluator: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        default_train_on_what = self.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE

        train_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/oxford_flowers102",
            dataset_split="train",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            num_repeats=self.num_repeats,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            examples_per_class=self.examples_per_class,
            subset_seed=self.subset_seed,
            max_image_size=self.max_image_size,
            hflip_probability=0.5,
        )

        test_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/oxford_flowers102",
            dataset_split="test",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            max_image_size=self.max_image_size,
            hflip_probability=0.0,
            # Note: test set uses full data, no few-shot sampling
        )

        train_dataset = ClassifierDataset(train_config)

        if not self.run_nll_evaluator:
            return train_dataset, None

        return train_dataset, ClassifierDataset(test_config)


@chz.chz
class OxfordPetsDatasetBuilder(SupervisedDatasetBuilder):
    """
    Configuration for a classification dataset.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    train_on_what: TrainOnWhat | None = None

    # If set, sample only this many examples per class (for few-shot experiments)
    examples_per_class: int | None = None
    subset_seed: int = 0

    max_image_size: int = 480

    run_nll_evaluator: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        default_train_on_what = self.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE

        train_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/oxford_iiit_pet",
            dataset_split="train",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            num_repeats=self.num_repeats,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            examples_per_class=self.examples_per_class,
            subset_seed=self.subset_seed,
            max_image_size=self.max_image_size,
            hflip_probability=0.5,
        )

        test_config = ClassifierDatasetConfig(
            dataset="dpdl-benchmark/oxford_iiit_pet",
            dataset_split="test",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            max_image_size=self.max_image_size,
            hflip_probability=0.0,
            # Note: test set uses full data, no few-shot sampling
        )

        train_dataset = ClassifierDataset(train_config)

        if not self.run_nll_evaluator:
            return train_dataset, None

        return train_dataset, ClassifierDataset(test_config)


@chz.chz
class StanfordCarsDatasetBuilder(SupervisedDatasetBuilder):
    """
    Configuration for a classification dataset.
    """

    model_name_for_tokenizer: str
    renderer_name: str

    num_repeats: float = 1
    batch_size: int = 32
    max_length: int = 8192

    train_on_what: TrainOnWhat | None = None

    # If set, sample only this many examples per class (for few-shot experiments)
    examples_per_class: int | None = None
    subset_seed: int = 0

    max_image_size: int = 480

    run_nll_evaluator: bool = False

    def __call__(self) -> tuple[SupervisedDataset, SupervisedDataset | None]:
        default_train_on_what = self.train_on_what or TrainOnWhat.LAST_ASSISTANT_MESSAGE

        train_config = ClassifierDatasetConfig(
            dataset="tanganke/stanford_cars",
            dataset_split="train",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            num_repeats=self.num_repeats,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            examples_per_class=self.examples_per_class,
            subset_seed=self.subset_seed,
            max_image_size=self.max_image_size,
            hflip_probability=0.5,
        )

        test_config = ClassifierDatasetConfig(
            dataset="tanganke/stanford_cars",
            dataset_split="test",
            image_column_name="image",
            label_column_name="label",
            renderer_name=self.renderer_name,
            model_name_for_tokenizer=self.model_name_for_tokenizer,
            batch_size=self.batch_size,
            max_length=self.max_length,
            train_on_what=default_train_on_what,
            max_image_size=self.max_image_size,
            hflip_probability=0.0,
            # Note: test set uses full data, no few-shot sampling
        )

        train_dataset = ClassifierDataset(train_config)

        if not self.run_nll_evaluator:
            return train_dataset, None

        return train_dataset, ClassifierDataset(test_config)


DATASETS = {
    "caltech101": Caltech101DatasetBuilder,
    "flowers102": Flowers102DatasetBuilder,
    "pets": OxfordPetsDatasetBuilder,
    "cars": StanfordCarsDatasetBuilder,
}


def get_dataset_builder(
    dataset: str,
    model_name_for_tokenizer: str,
    renderer_name: str,
    num_repeats: float = 1,
    batch_size: int = 32,
    max_length: int = 8192,
    train_on_what: TrainOnWhat | None = None,
    examples_per_class: int | None = None,
    subset_seed: int = 0,
    max_image_size: int = 480,
    run_nll_evaluator: bool = False,
) -> SupervisedDatasetBuilder:
    """
    Create a training and test dataset for a vlm classifier.

    Args:
        examples_per_class: If set, sample only this many examples per class
            from the training set (for few-shot experiments). Test set is
            unaffected.
        subset_seed: Seed for shuffling before selecting the few-shot subset.
        max_image_size: Maximum size for the longest side of images. Images
            larger than this will be resized while preserving aspect ratio.
    """

    return DATASETS[dataset](
        model_name_for_tokenizer=model_name_for_tokenizer,
        renderer_name=renderer_name,
        num_repeats=num_repeats,
        batch_size=batch_size,
        max_length=max_length,
        train_on_what=train_on_what,
        examples_per_class=examples_per_class,
        subset_seed=subset_seed,
        max_image_size=max_image_size,
        run_nll_evaluator=run_nll_evaluator,
    )
