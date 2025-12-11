# Supervised Learning

## VLM Image Classification

This recipe will teach you how to train an image classifier powered by vision-language models on `tinker`.

```bash
python -m tinker_cookbook.recipes.vlm_classifier.train \
    experiment_dir=./vlm_classifier \
    wandb_project=vlm-classifier \
    dataset=caltech101 \
    renderer_name=qwen3_vl \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct
```

Currently, the qwen series of VLMs are supported. Running the above script after installing tinker-cookbook will fine-tune `Qwen/Qwen3-VL-30B-A3B-Instruct` on the `caltech101` as an example.

### Evaluation

Once trained, you can evaluate the class predictions from your VLM as follows:

```bash
python -m tinker_cookbook.recipes.vlm_classifier.eval \
    dataset=caltech101 \
    model_path=$YOUR_MODEL_PATH \
    model_name=Qwen/Qwen3-VL-30B-A3B-Instruct \
    renderer_name=qwen3_vl
```

This will print the test accuracy of your model.

### Custom Datasets

You can add custom datasets by creating a custom `SupervisedDatasetBuilder` in `tinker_cookbook.recipes.vlm_classifier.data` if your dataset is available for download on Huggingface, and has a column with your image, and a column with the image labels (note, you must also define a `ClassLabel` for mapping integer labels to a human-readable class name).

For more general datasets, you can subclass the base `ClassifierDataset` to load arbitrary image classification datasets in the provided classifier tooling.

### Custom Evaluators

We provide a suite of evaluators in `tinker_cookbook.recipes.vlm_classifier.eval` for sampling from VLMs, parsing the predicted class name from the response, and computing evaluation metrics.

To define a custom evaluator for a new dataset, you can create a new `EvaluatorBuilder` if your dataset is available on Huggingface, or you can subclass `ClassifierEvaluator` to add an arbitrary custom dataset and parsing strategy.
