from tinker_cookbook import model_info
from tinker_cookbook.recipes.rubric.env import RubricGradedEnv, RubricBasedDatapoint, Rubric
from tinker_cookbook.completers import TinkerMessageCompleter, TinkerTokenCompleter
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.tokenizer_utils import get_tokenizer
import tinker
from tinker_cookbook.rl.rollouts import do_single_rollout
import asyncio


def get_addition_datapoint() -> RubricBasedDatapoint:
    datapoint = RubricBasedDatapoint(
        convo=[
            {"role": "user", "content": "What is 4 + 5?"},
            {"role": "assistant", "content": "9"},
            {"role": "user", "content": "What is 125 + 311?"},
        ],
        rubric_items=[
            Rubric(rubric_str="Does the chatbot correctly get the answer 436?"),
            Rubric(rubric_str="Does the chatbot provide an answer without saying anything else?"),
        ],
    )

    return datapoint


def get_prometheus_datapoint() -> RubricBasedDatapoint:
    from tinker_cookbook.recipes.rubric.data import PrometheusDatapointListBuilder

    datapoint = PrometheusDatapointListBuilder()()
    datapoint = datapoint[0]
    return datapoint


async def main(datapoint: RubricBasedDatapoint):
    # Configuration parameters
    policy_name = "meta-llama/Llama-3.1-8B-Instruct"
    grader_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
    policy_max_tokens = 64
    grader_max_tokens = 64

    service_client = tinker.ServiceClient()
    policy = TinkerTokenCompleter(
        sampling_client=service_client.create_sampling_client(base_model=policy_name),
        max_tokens=policy_max_tokens,
    )
    policy_renderer = get_renderer(
        model_info.get_recommended_renderer_name(policy_name), get_tokenizer(policy_name)
    )
    grader = TinkerMessageCompleter(
        sampling_client=service_client.create_sampling_client(base_model=grader_name),
        renderer=get_renderer(
            model_info.get_recommended_renderer_name(grader_name), get_tokenizer(grader_name)
        ),
        max_tokens=grader_max_tokens,
    )

    env = RubricGradedEnv(
        renderer=policy_renderer,
        datapoint=datapoint,
        grader_llm=grader,
        debug=True,
    )

    await do_single_rollout(policy, env)


if __name__ == "__main__":
    dataset = "addition"

    if dataset == "addition":
        datapoint = get_addition_datapoint()
        asyncio.run(main(datapoint))
    elif dataset == "prometheus":
        datapoint = get_prometheus_datapoint()
        asyncio.run(main(datapoint))
    else:
        raise ValueError(f"Unknown dataset: {dataset}")
