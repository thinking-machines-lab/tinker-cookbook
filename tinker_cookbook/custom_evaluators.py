import asyncio
import json
import logging
import os
from typing import Optional

import chz
import tinker
from tinker import types

from tinker_cookbook import renderers
from tinker_cookbook.evaluators import SamplingClientEvaluator
from tinker_cookbook.tokenizer_utils import get_tokenizer

# Set up logger
logger = logging.getLogger(__name__)


@chz.chz
class JsonlEvaluatorBuilder:
    """
    Configuration for JSONL-based evaluation.
    This class provides a structured way to configure JSONL evaluation
    parameters for evaluating model responses against expected answers
    using exact match scoring.
    """

    # Required parameters
    jsonl_file_path: str
    renderer_name: str
    model_name: str
    eval_name: str

    # Generation parameters
    temperature: float = 0.6
    top_p: float = 1.0
    max_tokens: int | None = None

    # Evaluation parameters
    limit: Optional[int] = None
    debug_errors: bool = False

    def __call__(self) -> SamplingClientEvaluator:
        return JsonlEvaluator(self)


class JsonlEvaluator(SamplingClientEvaluator):
    """
    A SamplingClientEvaluator that reads questions and answers from a JSONL file
    and evaluates model responses using exact match scoring.
    """

    def __init__(self, config: JsonlEvaluatorBuilder):
        """
        Initialize the JsonlEvaluator.
        Args:
            config: Configuration object containing all evaluation parameters
        """
        self.config = config
        tokenizer = get_tokenizer(self.config.model_name)
        self.renderer = renderers.get_renderer(self.config.renderer_name, tokenizer)
        self.qa_pairs, self.max_answer_tokens = self._load_qa_pairs()
        # Apply limit if specified
        if self.config.limit:
            self.qa_pairs = self.qa_pairs[: self.config.limit]
            logger.info(f"Limited evaluation to {len(self.qa_pairs)} questions")

        if self.config.max_tokens is None:
            # add a buffer of 32 to allow for reference in generation
            self.max_tokens = self.max_answer_tokens + 32
        else:
            self.max_tokens = self.config.max_tokens
        logger.info(f"Max tokens: {self.max_tokens}")

    def _load_qa_pairs(self) -> tuple[list[str], int]:
        """
        Load question-answer pairs from the JSONL file.
        Returns:
            List of (question, answer) tuples
        """
        qa_pairs = []
        jsonl_path = os.path.expanduser(self.config.jsonl_file_path)
        max_answer_tokens = 0

        if not os.path.exists(jsonl_path):
            raise FileNotFoundError(f"JSONL file not found: {jsonl_path}")

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                question = data.get("question", "").strip()
                answer = data.get("answer", "").strip()

                answer_tokens = self.renderer.tokenizer.encode(answer, add_special_tokens=False)
                max_answer_tokens = max(len(answer_tokens), max_answer_tokens)
                qa_pairs.append((question, answer))

        logger.info(f"Loaded {len(qa_pairs)} question-answer pairs from {jsonl_path}")
        logger.info(f"Max answer length: {max_answer_tokens}")
        return qa_pairs, max_answer_tokens

    def _exact_match(self, predicted: str, expected: str) -> bool:
        """
        Calculate exact match between predicted and expected answers.
        Args:
            predicted: The model's predicted answer
            expected: The expected correct answer
        Returns:
            True if exact match, False otherwise
        """
        # Strip whitespace and compare case-insensitively
        predicted_clean = predicted.strip().lower()
        expected_clean = expected.strip().lower()
        return predicted_clean == expected_clean

    def _reference_in_generation(self, predicted: str, reference: str) -> bool:
        """
        Check if the reference is in the predicted answer.
        """
        return reference.lower() in predicted.lower()

    async def _evaluate_single_question(
        self, sampling_client: tinker.SamplingClient, question: str, expected_answer: str
    ) -> dict[str, bool]:
        """
        Evaluate a single question asynchronously.
        """
        messages = [renderers.Message(role="user", content=question)]

        model_input = self.renderer.build_generation_prompt(messages)

        sampling_params = types.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            stop=self.renderer.get_stop_sequences(),
        )

        # Generate response
        response = await sampling_client.sample_async(
            prompt=model_input, num_samples=1, sampling_params=sampling_params
        )

        parsed_message, _ = self.renderer.parse_response(response.sequences[0].tokens)
        predicted_answer = parsed_message["content"].strip()

        # Check exact match
        is_correct = self._exact_match(predicted_answer, expected_answer)
        is_reference_in_generation = self._reference_in_generation(
            predicted_answer, expected_answer
        )

        # Log individual results for debugging if enabled
        if self.config.debug_errors:
            status = "~" if is_reference_in_generation else "✗"
            status = "✓" if is_correct else status
            logger.info(
                f"{status}: '{question}' -> Predicted: '{predicted_answer}', Expected: '{expected_answer}'"
            )

        return {
            "is_correct": is_correct,
            "is_reference_in_generation": is_reference_in_generation,
        }

    async def __call__(self, sampling_client: tinker.SamplingClient) -> dict[str, float]:
        """
        Run JSONL evaluation on the given sampling client and return metrics.
        Args:
            sampling_client: The sampling client to evaluate
        Returns:
            Dictionary of metrics from JSONL evaluation
        """
        # Load question-answer pairs
        qa_pairs = self.qa_pairs

        if not qa_pairs:
            logger.warning("No question-answer pairs found in JSONL file")
            return {"exact_match": 0.0, "total_questions": 0}

        # Evaluate each question
        num_correct = 0
        num_questions = len(qa_pairs)

        is_corrects = await asyncio.gather(
            *[
                self._evaluate_single_question(sampling_client, question, expected_answer)
                for question, expected_answer in qa_pairs
            ]
        )

        num_correct = sum([is_corrects[i]["is_correct"] for i in range(len(is_corrects))])
        num_reference_in_generation = sum(
            [is_corrects[i]["is_reference_in_generation"] for i in range(len(is_corrects))]
        )

        # Calculate final metrics
        exact_match_score = num_correct / num_questions
        reference_in_generation_score = num_reference_in_generation / num_questions

        metrics = {
            f"{self.config.eval_name}/exact_match": exact_match_score,
            f"{self.config.eval_name}/reference_in_generation": reference_in_generation_score,
        }

        logger.info(
            f"""
            Num correct: {num_correct}
            Num reference in generation: {num_reference_in_generation}
            Num questions: {num_questions}
            """
        )
        logger.info(f"{self.config.eval_name} evaluation completed. Metrics: {metrics}")
        return metrics
