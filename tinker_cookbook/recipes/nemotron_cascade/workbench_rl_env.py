"""
Workbench Tool-Calling RL environment for Nemotron-Cascade-2 replication.

Multi-turn environment where the model calls workplace tools (email, calendar,
CRM, etc.) to complete tasks. Uses the `build_agent_tool_env` multi-turn
infrastructure (same pattern as harbor_rl and swe_agentic).

The tools return mock results since we don't have the actual backend.
Reward is based on whether the model's tool calls match the ground_truth.
"""

import json
import logging
import math
from collections.abc import Sequence
from typing import cast

import chz
from datasets import Dataset

from tinker_cookbook import model_info, tokenizer_utils
from tinker_cookbook.renderers import get_renderer
from tinker_cookbook.renderers.base import Message, Renderer
from tinker_cookbook.rl.types import (
    Env,
    EnvGroupBuilder,
    Metrics,
    RLDataset,
    RLDatasetBuilder,
    Trajectory,
)
from tinker_cookbook.tool_use import ToolResult, build_agent_tool_env, simple_tool_result, tool
from tinker_cookbook.tool_use.agent_tool_message_env import RewardFn

logger = logging.getLogger(__name__)


def _normalize_args(args_str: str | dict) -> dict:
    if isinstance(args_str, dict):
        return args_str
    try:
        return json.loads(args_str)
    except (json.JSONDecodeError, TypeError):
        return {}


def check_tool_calls_in_messages(
    messages: list[Message],
    ground_truth: list[dict],
) -> float:
    """Check if the conversation contains the expected tool calls."""
    if not ground_truth:
        return 1.0

    # Extract all tool calls from assistant messages
    model_calls = []
    for msg in messages:
        if msg.get("role") == "assistant" and msg.get("tool_calls"):
            for tc in msg["tool_calls"]:
                if hasattr(tc, 'function'):
                    model_calls.append({"name": tc.function.name, "arguments": _normalize_args(tc.function.arguments)})
                elif isinstance(tc, dict):
                    func = tc.get("function", tc)
                    model_calls.append({"name": func.get("name", ""), "arguments": _normalize_args(func.get("arguments", "{}"))})

    if not model_calls:
        return 0.0

    # Check each ground truth call
    matched = 0
    for gt in ground_truth:
        gt_name = gt.get("name", "")
        gt_args = _normalize_args(gt.get("arguments", "{}"))
        for mc in model_calls:
            if mc["name"] == gt_name:
                args_match = all(str(mc["arguments"].get(k)) == str(v) for k, v in gt_args.items())
                if args_match:
                    matched += 1
                    break

    return matched / len(ground_truth)


class WorkbenchTools:
    """Mock tool implementations for workbench tasks.

    Returns plausible JSON results for workplace tools.
    The actual values don't matter much since we grade on tool call correctness,
    not on the final answer content.
    """

    @tool
    def company_directory_find_email_address(self, name: str = "") -> ToolResult:
        """Finds all email addresses containing the given name."""
        return simple_tool_result(json.dumps([
            {"email": f"{name.lower().replace(' ', '.')}@company.com", "name": name}
        ]))

    @tool
    def email_search_emails(self, query: str = "", date_min: str = "", date_max: str = "",
                           page: int = 1, page_size: int = 10) -> ToolResult:
        """Searches for emails matching the query."""
        return simple_tool_result(json.dumps({
            "emails": [{"email_id": f"0000025{i}", "subject": f"Re: {query}",
                        "sender": f"{query.lower()}@company.com", "sent_datetime": "2023-11-30T10:00:00"}
                       for i in range(min(3, page_size))],
            "total": 3, "page": page
        }))

    @tool
    def email_get_email_information_by_id(self, email_id: str = "", field: str = "") -> ToolResult:
        """Retrieves email details by ID."""
        info = {"email_id": email_id, "subject": "Meeting follow-up",
                "sender": "user@company.com", "body": "Please review the attached.", "sent_datetime": "2023-11-30T10:00:00"}
        return simple_tool_result(json.dumps(info.get(field, info) if field else info))

    @tool
    def email_send_email(self, to: str = "", subject: str = "", body: str = "") -> ToolResult:
        """Sends an email."""
        return simple_tool_result(json.dumps({"status": "sent", "email_id": "00000300"}))

    @tool
    def email_delete_email(self, email_id: str = "") -> ToolResult:
        """Deletes an email by ID."""
        return simple_tool_result(json.dumps({"status": "deleted", "email_id": email_id}))

    @tool
    def calendar_get_events(self, date: str = "", start_date: str = "", end_date: str = "") -> ToolResult:
        """Gets calendar events."""
        return simple_tool_result(json.dumps({"events": [
            {"event_id": "EVT001", "title": "Team standup", "start": f"{date or start_date}T09:00:00",
             "end": f"{date or start_date}T09:30:00", "attendees": ["user@company.com"]}
        ]}))

    @tool
    def calendar_create_event(self, title: str = "", start: str = "", end: str = "",
                             attendees: str = "") -> ToolResult:
        """Creates a calendar event."""
        return simple_tool_result(json.dumps({"status": "created", "event_id": "EVT100"}))

    @tool
    def calendar_delete_event(self, event_id: str = "") -> ToolResult:
        """Deletes a calendar event."""
        return simple_tool_result(json.dumps({"status": "deleted", "event_id": event_id}))

    @tool
    def calendar_update_event(self, event_id: str = "", **kwargs) -> ToolResult:
        """Updates a calendar event."""
        return simple_tool_result(json.dumps({"status": "updated", "event_id": event_id}))

    @tool
    def project_management_get_tasks(self, project: str = "", status: str = "") -> ToolResult:
        """Gets project tasks."""
        return simple_tool_result(json.dumps({"tasks": [
            {"task_id": "TASK001", "title": "Review PR", "status": "in_progress", "assignee": "user"}
        ]}))

    @tool
    def project_management_create_task(self, title: str = "", **kwargs) -> ToolResult:
        """Creates a project task."""
        return simple_tool_result(json.dumps({"status": "created", "task_id": "TASK100"}))

    @tool
    def project_management_update_task(self, task_id: str = "", **kwargs) -> ToolResult:
        """Updates a project task."""
        return simple_tool_result(json.dumps({"status": "updated", "task_id": task_id}))

    @tool
    def analytics_get_report(self, report_type: str = "", **kwargs) -> ToolResult:
        """Gets an analytics report."""
        return simple_tool_result(json.dumps({"report_type": report_type, "data": [
            {"metric": "revenue", "value": 125000}, {"metric": "users", "value": 5432}
        ]}))

    @tool
    def customer_relationship_manager_get_customer(self, customer_id: str = "") -> ToolResult:
        """Gets customer information."""
        return simple_tool_result(json.dumps({"customer_id": customer_id, "name": "Acme Corp",
                                              "email": "contact@acme.com", "status": "active"}))

    @tool
    def customer_relationship_manager_search_customers(self, query: str = "", **kwargs) -> ToolResult:
        """Searches for customers."""
        return simple_tool_result(json.dumps({"customers": [
            {"customer_id": "CUST001", "name": query, "email": f"{query.lower()}@example.com"}
        ]}))

    @tool
    def customer_relationship_manager_create_customer(self, **kwargs) -> ToolResult:
        """Creates a new customer."""
        return simple_tool_result(json.dumps({"status": "created", "customer_id": "CUST100"}))

    @tool
    def customer_relationship_manager_update_customer(self, customer_id: str = "", **kwargs) -> ToolResult:
        """Updates a customer record."""
        return simple_tool_result(json.dumps({"status": "updated", "customer_id": customer_id}))

    @tool
    def customer_relationship_manager_delete_customer(self, customer_id: str = "") -> ToolResult:
        """Deletes a customer record."""
        return simple_tool_result(json.dumps({"status": "deleted", "customer_id": customer_id}))


class WorkbenchReward:
    """Reward function that checks tool calls against ground_truth."""

    def __init__(self, ground_truth: list[dict]):
        self.ground_truth = ground_truth

    async def __call__(self, messages: list[Message]) -> tuple[float, Metrics]:
        reward = check_tool_calls_in_messages(messages, self.ground_truth)
        return reward, {"correct": reward, "n_expected": len(self.ground_truth)}


class WorkbenchEnvGroupBuilder(EnvGroupBuilder):
    """Builds multi-turn workbench environments."""

    def __init__(
        self,
        prompt_messages: list[dict],
        ground_truth: list[dict],
        renderer_name: str,
        tokenizer_name: str,
        num_envs: int,
        category: str = "workbench",
        max_turns: int = 10,
    ):
        self._prompt_messages = prompt_messages
        self._ground_truth = ground_truth
        self._renderer_name = renderer_name
        self._tokenizer_name = tokenizer_name
        self._num_envs = num_envs
        self._category = category
        self._max_turns = max_turns

    async def make_envs(self) -> Sequence[Env]:
        tokenizer = tokenizer_utils.get_tokenizer(self._tokenizer_name)
        renderer = get_renderer(self._renderer_name, tokenizer)

        from tinker_cookbook.tool_use import Tool
        tools_obj = WorkbenchTools()
        all_tools = [getattr(tools_obj, m) for m in dir(tools_obj)
                     if isinstance(getattr(tools_obj, m), Tool)]

        # Build initial messages (system + user from data)
        initial_messages: list[Message] = []
        for msg in self._prompt_messages:
            initial_messages.append({"role": msg["role"], "content": msg["content"]})

        reward_fn = WorkbenchReward(self._ground_truth)

        envs = []
        for _ in range(self._num_envs):
            env = build_agent_tool_env(
                renderer=renderer,
                tools=all_tools,
                initial_messages=initial_messages,
                reward_fn=reward_fn,
                max_turns=self._max_turns,
            )
            envs.append(env)
        return envs

    def logging_tags(self) -> list[str]:
        return [self._category or "workbench"]


class WorkbenchRLDataset(RLDataset):
    def __init__(self, batch_size: int, group_size: int, renderer_name: str,
                 tokenizer_name: str, max_turns: int = 10, seed: int = 0):
        logger.info("Loading workbench RL data...")
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="multi-domain-RL", split="train")
        ds = cast(Dataset, ds)
        ds = ds.filter(lambda x: x.get("environment_name") == "workbench" and x.get("ground_truth"))
        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.tokenizer_name = tokenizer_name
        self.max_turns = max_turns
        logger.info(f"Workbench dataset: {len(self.ds)} examples")

    def get_batch(self, index: int) -> Sequence[EnvGroupBuilder]:
        batch_start = index * self.batch_size
        batch_end = min((index + 1) * self.batch_size, len(self.ds))
        assert batch_start < batch_end
        return [
            b for row in self.ds.select(range(batch_start, batch_end))
            if (b := self._make_builder(row)) is not None
        ]

    def __len__(self) -> int:
        return math.ceil(len(self.ds) / self.batch_size)

    def _make_builder(self, row: dict) -> WorkbenchEnvGroupBuilder | None:
        try:
            return WorkbenchEnvGroupBuilder(
                prompt_messages=row["responses_create_params"]["input"],
                ground_truth=row["ground_truth"],
                renderer_name=self.renderer_name,
                tokenizer_name=self.tokenizer_name,
                num_envs=self.group_size,
                category=row.get("category", "workbench"),
                max_turns=self.max_turns,
            )
        except Exception as e:
            logger.warning(f"Failed to parse workbench row: {e}")
            return None


@chz.chz
class WorkbenchRLDatasetBuilder(RLDatasetBuilder):
    batch_size: int
    model_name_for_tokenizer: str
    renderer_name: str
    group_size: int = 16
    max_turns: int = 10
    seed: int = 0

    async def __call__(self) -> tuple[WorkbenchRLDataset, None]:
        return WorkbenchRLDataset(
            batch_size=self.batch_size,
            group_size=self.group_size,
            renderer_name=self.renderer_name,
            tokenizer_name=self.model_name_for_tokenizer,
            max_turns=self.max_turns,
            seed=self.seed,
        ), None
