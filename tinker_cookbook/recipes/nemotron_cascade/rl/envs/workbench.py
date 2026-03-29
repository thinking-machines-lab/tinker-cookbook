"""
Workbench Tool-Calling RL environment for Nemotron-Cascade-2 replication.

Single-turn environment where the model calls workplace tools (email, calendar,
CRM, etc.) to complete tasks. Uses ground-truth-seeded mock backends so that
info-gathering calls (e.g. `email_search_emails`) return IDs and values
consistent with the expected ground-truth tool calls. This lets the model
succeed on tasks that require a lookup step before the action call.

The reward function uses partial credit:
  - 0.0 if the model doesn't call any tools
  - 0.5 if the model calls the correct tool name(s) but with wrong arguments
  - 1.0 if the model calls the correct tool(s) with exact argument matches

Uses `build_agent_tool_env` with max_turns=3 so the model can do a short
info-gathering step before the action call. The reward function checks ALL
tool calls in the conversation, so the model gets credit as long as the
correct call appears anywhere in the trajectory.
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


def _extract_gt_ids(ground_truth: list[dict]) -> dict[str, list[str]]:
    """Extract IDs from ground-truth tool calls to seed mock backends.

    Scans ground-truth arguments for values that look like IDs (e.g. email_id,
    event_id, task_id, customer_id) and groups them by ID type. Also extracts
    email addresses and names that the model might need to discover via lookup.

    Returns a dict like:
        {"email_id": ["00000259"], "event_id": ["00000170"], "email": ["jinsoo.kim@atlas.com"]}
    """
    ids: dict[str, list[str]] = {}
    for gt_call in ground_truth:
        args = _normalize_args(gt_call.get("arguments", "{}"))
        for key, value in args.items():
            value_str = str(value)
            # Collect ID-like fields and useful lookup values
            if key.endswith("_id") or key.endswith("_email") or key == "email" or key == "assigned_to_email":
                ids.setdefault(key, []).append(value_str)
            # Also collect all argument values by key for general seeding
            ids.setdefault(f"_arg_{key}", []).append(value_str)
    return ids


def check_tool_calls_in_messages(
    messages: list[Message],
    ground_truth: list[dict],
) -> tuple[float, float, float]:
    """Check if the conversation contains the expected tool calls.

    Returns (reward, name_match_rate, exact_match_rate):
      - reward: 0.5 * name_match_rate + 0.5 * exact_match_rate
      - name_match_rate: fraction of GT calls matched by tool name
      - exact_match_rate: fraction of GT calls matched by name + all args
    """
    if not ground_truth:
        return 1.0, 1.0, 1.0

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
        return 0.0, 0.0, 0.0

    # Check each ground truth call for name match and exact match separately
    name_matched = 0
    exact_matched = 0
    used_indices: set[int] = set()  # Track which model calls we've used

    for gt in ground_truth:
        gt_name = gt.get("name", "")
        gt_args = _normalize_args(gt.get("arguments", "{}"))

        best_match = "none"
        best_idx = -1

        for i, mc in enumerate(model_calls):
            if i in used_indices:
                continue
            if mc["name"] == gt_name:
                args_match = all(str(mc["arguments"].get(k)) == str(v) for k, v in gt_args.items())
                if args_match:
                    best_match = "exact"
                    best_idx = i
                    break
                elif best_match == "none":
                    best_match = "name"
                    best_idx = i

        if best_match == "exact":
            exact_matched += 1
            name_matched += 1
            used_indices.add(best_idx)
        elif best_match == "name":
            name_matched += 1
            used_indices.add(best_idx)

    n = len(ground_truth)
    name_rate = name_matched / n
    exact_rate = exact_matched / n
    # Partial credit: 0.5 for getting the tool name right, 0.5 for exact args
    reward = 0.5 * name_rate + 0.5 * exact_rate
    return reward, name_rate, exact_rate


class WorkbenchTools:
    """Mock tool implementations seeded with ground-truth IDs.

    The ground_truth_ids dict maps ID field names to lists of expected values.
    When lookup tools are called (e.g. email_search_emails, calendar_get_events),
    the mock returns results containing these real IDs so the model can use them
    in subsequent action calls and match the ground truth.
    """

    def __init__(self, ground_truth_ids: dict[str, list[str]] | None = None):
        self._gt_ids = ground_truth_ids or {}

    def _get_gt_id(self, key: str, fallback: str) -> str:
        """Get the first ground-truth ID for a key, or return fallback."""
        vals = self._gt_ids.get(key, [])
        return vals[0] if vals else fallback

    def _get_gt_email(self) -> str:
        """Get a ground-truth email address if available."""
        for key in ("assigned_to_email", "email", "to"):
            vals = self._gt_ids.get(key, []) + self._gt_ids.get(f"_arg_{key}", [])
            if vals:
                return vals[0]
        return ""

    @tool
    def company_directory_find_email_address(self, name: str = "") -> ToolResult:
        """Finds all email addresses containing the given name."""
        gt_email = self._get_gt_email()
        if gt_email:
            email = gt_email
        else:
            email = f"{name.lower().replace(' ', '.')}@company.com"
        return simple_tool_result(json.dumps([
            {"email": email, "name": name}
        ]))

    @tool
    def email_search_emails(self, query: str = "", date_min: str = "", date_max: str = "",
                           page: int = 1, page_size: int = 10) -> ToolResult:
        """Searches for emails matching the query."""
        gt_email_id = self._get_gt_id("email_id", "")
        if not gt_email_id:
            gt_email_id = self._get_gt_id("_arg_email_id", "00000250")
        return simple_tool_result(json.dumps({
            "emails": [{"email_id": gt_email_id, "subject": f"Re: {query}",
                        "sender": f"{query.lower()}@company.com", "sent_datetime": "2023-11-30T10:00:00"}],
            "total": 1, "page": page
        }))

    @tool
    def email_get_email_information_by_id(self, email_id: str = "", field: str = "") -> ToolResult:
        """Retrieves email details by ID."""
        gt_email = self._get_gt_email()
        info = {"email_id": email_id, "subject": "Meeting follow-up",
                "sender": gt_email or "user@company.com",
                "body": "Please review the attached.", "sent_datetime": "2023-11-30T10:00:00"}
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
        gt_event_id = self._get_gt_id("event_id", "")
        if not gt_event_id:
            gt_event_id = self._get_gt_id("_arg_event_id", "EVT001")
        gt_event_name = self._get_gt_id("_arg_field", "")
        # If the ground truth has an event name update, include it in the results
        title = gt_event_name if gt_event_name else "Team standup"
        d = date or start_date or "2023-11-30"
        return simple_tool_result(json.dumps({"events": [
            {"event_id": gt_event_id, "title": title, "start": f"{d}T09:00:00",
             "end": f"{d}T09:30:00", "attendees": ["user@company.com"]}
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
        gt_task_id = self._get_gt_id("task_id", "")
        if not gt_task_id:
            gt_task_id = self._get_gt_id("_arg_task_id", "TASK001")
        return simple_tool_result(json.dumps({"tasks": [
            {"task_id": gt_task_id, "title": "Review PR", "status": "in_progress", "assignee": "user"}
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
    def analytics_create_plot(self, time_min: str = "", time_max: str = "",
                              value_to_plot: str = "", plot_type: str = "") -> ToolResult:
        """Creates an analytics plot."""
        return simple_tool_result(json.dumps({
            "status": "created", "plot_type": plot_type,
            "value_to_plot": value_to_plot, "time_range": f"{time_min} to {time_max}"
        }))

    @tool
    def customer_relationship_manager_get_customer(self, customer_id: str = "") -> ToolResult:
        """Gets customer information."""
        return simple_tool_result(json.dumps({"customer_id": customer_id, "name": "Acme Corp",
                                              "email": "contact@acme.com", "status": "active"}))

    @tool
    def customer_relationship_manager_search_customers(self, query: str = "", **kwargs) -> ToolResult:
        """Searches for customers."""
        gt_customer_id = self._get_gt_id("customer_id", "")
        if not gt_customer_id:
            gt_customer_id = self._get_gt_id("_arg_customer_id", "CUST001")
        return simple_tool_result(json.dumps({"customers": [
            {"customer_id": gt_customer_id, "name": query, "email": f"{query.lower()}@example.com"}
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
    """Reward function with partial credit for tool calls against ground_truth.

    Scoring:
      - 0.0 if model makes no tool calls
      - 0.5 * (fraction of GT calls matched by name) +
        0.5 * (fraction of GT calls matched by name + exact args)
      - So correct tool name with wrong args = 0.5, perfect match = 1.0
    """

    def __init__(self, ground_truth: list[dict]):
        self.ground_truth = ground_truth

    async def __call__(self, messages: list[Message]) -> tuple[float, Metrics]:
        reward, name_rate, exact_rate = check_tool_calls_in_messages(messages, self.ground_truth)
        return reward, {
            "correct": exact_rate,
            "name_match": name_rate,
            "partial_reward": reward,
            "n_expected": len(self.ground_truth),
        }


class WorkbenchEnvGroupBuilder(EnvGroupBuilder):
    """Builds workbench environments with ground-truth-seeded mock tools."""

    def __init__(
        self,
        prompt_messages: list[dict],
        ground_truth: list[dict],
        renderer_name: str,
        tokenizer_name: str,
        num_envs: int,
        category: str = "workbench",
        max_turns: int = 3,
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

        # Seed mock tools with ground-truth IDs so lookup calls return
        # values consistent with what the ground truth expects.
        gt_ids = _extract_gt_ids(self._ground_truth)
        tools_obj = WorkbenchTools(ground_truth_ids=gt_ids)
        all_tools = [getattr(tools_obj, m) for m in dir(tools_obj)
                     if isinstance(getattr(tools_obj, m), Tool)]

        # Extract user messages (skip system -- we'll rebuild it with tool schemas)
        user_messages: list[Message] = []
        system_prompt = ""
        for msg in self._prompt_messages:
            if msg["role"] == "system":
                system_prompt = msg["content"]
            else:
                user_messages.append({"role": msg["role"], "content": msg["content"]})

        # Build initial messages with tool schemas injected into the system message
        # so the model knows which tools are available and how to call them.
        tool_specs = [t.to_spec() for t in all_tools]
        initial_messages = renderer.create_conversation_prefix_with_tools(
            tools=tool_specs,
            system_prompt=system_prompt,
        ) + user_messages

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
                 tokenizer_name: str, max_turns: int = 3, seed: int = 0):
        logger.info("Loading workbench RL data...")
        from datasets import load_dataset
        ds = load_dataset("nvidia/Nemotron-Cascade-2-RL-data", name="multi-domain-RL", split="train")
        ds = cast(Dataset, ds)
        ds = ds.filter(lambda x: x.get("environment_name") == "workbench" and x.get("ground_truth"))
        # Include all workbench tasks (single-call and multi-call). Multi-call
        # tasks are mostly parallel calls (e.g. two analytics_create_plot) where
        # args come from the user message, not from chained tool results. The
        # ground-truth-seeded mock backend handles the few cases that need lookup
        # by returning IDs consistent with the expected ground truth.
        self.ds = ds.shuffle(seed=seed)
        self.batch_size = batch_size
        self.group_size = group_size
        self.renderer_name = renderer_name
        self.tokenizer_name = tokenizer_name
        self.max_turns = max_turns
        logger.info(f"Workbench dataset: {len(self.ds)} examples (single + multi-call)")

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
    max_turns: int = 3
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
