from typing import Any, Callable, TypedDict

import tinker
from gepa.core.adapter import EvaluationBatch, GEPAAdapter

from tinker_cookbook import renderers
from tinker_cookbook.tokenizer_utils import Tokenizer

Scorer = Callable[[str, str, dict[str, Any] | None], float]


class TinkerDataInst(TypedDict):
    input: str
    answer: str
    metadata: dict[str, Any]


class TinkerTrajectory(TypedDict):
    data: TinkerDataInst
    response: str
    score: float
    error: str | None
    logprobs: list[float] | None
    tokens: list[int] | None


class TinkerRolloutOutput(TypedDict):
    response: str


TinkerReflectiveRecord = TypedDict(
    "TinkerReflectiveRecord",
    {
        "Inputs": str,
        "Generated Outputs": str,
        "Feedback": str,
    },
)


def default_scorer(response: str, answer: str, metadata: dict[str, Any] | None = None) -> float:
    return 1.0 if answer.lower().strip() in response.lower().strip() else 0.0


class TinkerReflectionLM:
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
        max_tokens: int = 4096,
        temperature: float = 0.3,
        system_prompt: str | None = None,
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.system_prompt = system_prompt or (
            "You are an expert prompt engineer. Analyze the execution traces and "
            "suggest improvements to the system prompt to improve task performance."
        )
        self.sampling_params = tinker.SamplingParams(
            max_tokens=max_tokens,
            temperature=temperature,
            stop=self.renderer.get_stop_sequences(),
        )

    def __call__(self, prompt: str) -> str:
        renderer_name = self.renderer.__class__.__name__
        supports_system = "DeepSeek" not in renderer_name

        if supports_system:
            messages: list[renderers.Message] = [
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": prompt},
            ]
        else:
            combined_content = f"{self.system_prompt}\n\n{prompt}"
            messages: list[renderers.Message] = [
                {"role": "user", "content": combined_content},
            ]

        model_input = self.renderer.build_generation_prompt(messages)

        future = self.sampling_client.sample(
            prompt=model_input,
            num_samples=1,
            sampling_params=self.sampling_params,
        )
        result = future.result()
        seq = result.sequences[0]
        parsed, _ = self.renderer.parse_response(seq.tokens)
        return parsed["content"]


class TinkerGEPAAdapter(GEPAAdapter[TinkerDataInst, TinkerTrajectory, TinkerRolloutOutput]):
    def __init__(
        self,
        sampling_client: tinker.SamplingClient,
        renderer: renderers.Renderer,
        tokenizer: Tokenizer,
        scorer: Scorer | None = None,
        max_tokens: int = 2048,
        temperature: float = 0.7,
        failure_score: float = 0.0,
        component_name: str = "system_prompt",
    ):
        self.sampling_client = sampling_client
        self.renderer = renderer
        self.tokenizer = tokenizer
        self.scorer = scorer or default_scorer
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.failure_score = failure_score
        self.component_name = component_name

        self.sampling_params = tinker.SamplingParams(
            max_tokens=self.max_tokens,
            temperature=self.temperature,
            stop=self.renderer.get_stop_sequences(),
        )

    def _get_system_prompt(self, candidate: dict[str, str]) -> str:
        if self.component_name not in candidate:
            raise ValueError(
                f"Candidate missing '{self.component_name}'. Got: {list(candidate.keys())}"
            )
        return candidate[self.component_name]

    def evaluate(
        self,
        batch: list[TinkerDataInst],
        candidate: dict[str, str],
        capture_traces: bool = False,
    ) -> EvaluationBatch[TinkerTrajectory, TinkerRolloutOutput]:
        system_prompt = self._get_system_prompt(candidate)

        futures = []
        for data in batch:
            messages: list[renderers.Message] = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": data["input"]},
            ]
            model_input = self.renderer.build_generation_prompt(messages)
            futures.append(
                self.sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=self.sampling_params,
                )
            )

        outputs: list[TinkerRolloutOutput] = []
        scores: list[float] = []
        trajectories: list[TinkerTrajectory] | None = [] if capture_traces else None

        for future, data in zip(futures, batch):
            result = future.result()
            seq = result.sequences[0]
            parsed, _ = self.renderer.parse_response(seq.tokens)
            response = parsed["content"]
            score = self.scorer(response, data["answer"], data.get("metadata"))

            outputs.append({"response": response})
            scores.append(score)

            if trajectories is not None:
                trajectories.append(
                    {
                        "data": data,
                        "response": response,
                        "score": score,
                        "error": None,
                        "logprobs": seq.logprobs,
                        "tokens": seq.tokens,
                    }
                )

        return EvaluationBatch(outputs=outputs, scores=scores, trajectories=trajectories)

    def make_reflective_dataset(
        self,
        candidate: dict[str, str],
        eval_batch: EvaluationBatch[TinkerTrajectory, TinkerRolloutOutput],
        components_to_update: list[str],
    ) -> dict[str, list[TinkerReflectiveRecord]]:
        trajectories = eval_batch.trajectories
        assert trajectories is not None

        result: dict[str, list[TinkerReflectiveRecord]] = {}

        for comp in components_to_update:
            items: list[TinkerReflectiveRecord] = []

            for traj in trajectories:
                data = traj["data"]
                response = traj["response"]
                score = traj["score"]
                error = traj["error"]

                if error:
                    feedback = f"Error: {error}"
                elif score >= 1.0:
                    feedback = f"Correct. Expected: '{data['answer']}'"
                else:
                    feedback = f"Incorrect. Expected: '{data['answer']}'"
                    if data.get("metadata"):
                        hints = ", ".join(f"{k}={v}" for k, v in data["metadata"].items())
                        feedback += f" (context: {hints})"

                items.append(
                    {
                        "Inputs": data["input"],
                        "Generated Outputs": response[:1000] if response else "(empty)",
                        "Feedback": feedback,
                    }
                )

            result[comp] = items

        return result
