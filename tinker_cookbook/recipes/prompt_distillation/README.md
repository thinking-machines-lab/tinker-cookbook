# Prompt Distillation

Prompt Distillation -- also known as context distillation [1,2] -- is a training method that can "make an LLM internalize the prompt into its parameters".
In this method, the model is fine-tuned to behave as if it had been provided with a long and complex prompt, even without actually accessing it.

For example, we want to internalize the following target prompt $p$:

`Classify the language of the provided text into these labels: en, fr, zh, ja ...`

After prompt distillation, the LLM will respond with only the language label after receiving a query without seeing the prompt $p$, e.g.,
```
Query: 一生、バンドしてくれる？
Response: ja
```

At a high level, this method involves two stages:
1. **Creating data for distillation**: A teacher language model uses $p$ to generate responses $r$ on a set of queries $q$; i.e. $r \sim \text{teacher}(\cdot|p, q)$
2. **Training the student model**: A student model is fine-tuned to predict the responses $r$ to the query $q$ but without accessing $p$, hence learning to behave as if the target prompt is in its context; i.e. $\text{student}(\cdot | q)$ should predict $r$

## Example

The Tinker Cookbook provides a prompt distillation recipe tailored for a language classification task. The objective is straightforward: given a text query, the model should predict a two-character code corresponding to the language of the input. The set of possible labels is:
```
ar (Arabic), de (German), el (Greek), en (English), es (Spanish), fr (French), hi (Hindi), ru (Russian), tr (Turkish), ur (Urdu), vi (Vietnamese), zh (Chinese - Simplified), ot (Other/Unknown).
```

The recipe in [`create_data.py`](create_data.py) also includes handling strategies for inputs containing code, numerical content, or multiple languages.

In the example below, the same model (`Qwen/Qwen3-30B-A3B`) is used as both teacher and student, though in general they need not be identical.

---

### Step 1: Generate Training Data

Generate prompt distillation data using the teacher model with [`create_data.py`](create_data.py):

```bash
mkdir -p /tmp/tinker-datasets
python -m tinker_cookbook.recipes.prompt_distillation.create_data \
  output_file=/tmp/tinker-datasets/prompt_distillation_lang.jsonl
```

This command will:
- Use the configured teacher model to generate language classification examples
- Save the distilled dataset to the specified output file
- Create diverse training examples suitable for student model fine-tuning

### Step 2: Train the Student Model

Fine-tune a student model on the distillation data using [`train.py`](train.py):

```bash
python -m tinker_cookbook.recipes.prompt_distillation.train
```

The training script will:
- Load the generated distillation dataset
- Apply optimized training configurations
- Fine-tune the student model for language classification

### Step 3: Test Your Model

Once training is complete, you can test your distilled model by sampling from the trained model to verify its performance on language classification tasks.

## Advanced Configuration

The prompt distillation recipe can be customized for different scenarios:

- **Teacher model selection**: Choose different base models based on your requirements
- **Sampling strategies**: Adjust temperature and other generation parameters
- **Data volume**: Scale the number of generated examples based on your needs
- **Training hyperparameters**: Fine-tune learning rates and other training settings

[1] Askell, A., Bai, Y., Chen, A., Drain, D., Ganguli, D., Henighan, T., Jones, A., Joseph, N., Mann, B., DasSarma, N., Elhage, N., Hatfield-Dodds, Z., Hernandez, D., Kernion, J., Ndousse, K., Olsson, C., Amodei, D., Brown, T., Clark, J., McCandlish, S., Olah, C., & Kaplan, J. (2021). A general language assistant as a laboratory for alignment. arXiv preprint arXiv:2112.00861.

[2] Snell, C., Klein, D., & Zhong, R. (2022). Learning by distilling context. arXiv preprint arXiv:2209.15189.
