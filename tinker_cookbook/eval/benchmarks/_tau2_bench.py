"""tau2-Bench benchmark — multi-turn customer service agent evaluation.

**Status: Experimental** — functional, iterating towards matching public scores.

Architecture: Uses ``MessageEnv`` + ``EnvFromMessageEnv`` pattern for native
tool-call parsing via the renderer. The model interacts with a simulated
customer and calls tools against a backend database.

Each turn:
1. Model response parsed by renderer for tool calls (native format)
2. If tool calls → execute against backend DB → return tool results
3. If no tool calls (text to customer) → simulate customer response → continue
4. Episode ends when model stops or max turns reached

Known limitations:
- Tool backend uses heuristic DB matching, not official Python implementations
- NL assertion grading (``nl_assertions``) not implemented
- DB state grading (``db_check``) not implemented

Metric: Task completion rate (action matching).
Requires ``config.judge_sampling_client`` for the user simulator (ideally a
separate model from the one being evaluated).
"""

from __future__ import annotations

import copy
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.benchmarks._common import (
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import (
    Renderer,
    ToolCall,
    ToolSpec,
    format_content_as_string,
    get_text_content,
)
from tinker_cookbook.rl.message_env import EnvFromMessageEnv, MessageEnv, MessageStepResult
from tinker_cookbook.rl.types import Env
from tinker_cookbook.tool_use.tools import simple_tool_result
from tinker_cookbook.tool_use.types import ToolInput, ToolResult

logger = logging.getLogger(__name__)

MAX_TURNS = 30
"""Maximum number of agent turns before forced termination."""

_TAU2_REPO = "https://raw.githubusercontent.com/sierra-research/tau2-bench/main"
_TAU2_CACHE = Path.home() / ".cache" / "tau2_bench"

_TAU2_TOOLS_URL = (
    "https://raw.githubusercontent.com/sierra-research/tau2-bench/main"
    "/web/leaderboard/public/task-data/tools-data.json"
)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _convert_tau2_tools_to_openai(tau2_tools: list[dict]) -> list[dict]:
    """Convert tau2-bench tool format to OpenAI function-calling format.

    tau2 uses flat ``parameters: [{name, type, required, description}]``.
    OpenAI uses ``parameters: {type: object, properties: {...}, required: [...]}``.
    """
    openai_tools = []
    for tool in tau2_tools:
        properties: dict[str, dict] = {}
        required: list[str] = []
        for param in tool.get("parameters", []):
            pname = param["name"]
            ptype = param.get("type", "string")
            prop: dict[str, Any] = {"type": ptype}
            if param.get("description"):
                prop["description"] = param["description"]
            if param.get("enum"):
                prop["enum"] = param["enum"]
            properties[pname] = prop
            if param.get("required", False):
                required.append(pname)
        openai_tools.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description", ""),
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": required,
                    },
                },
            }
        )
    return openai_tools


def _load_tau2_data(domain: str = "airline") -> tuple[list[dict], str, dict, list[dict]]:
    """Load tau2-bench data from GitHub repo (cached locally).

    Returns:
        Tuple of (tasks, policy_text, db_dict, tool_definitions_openai).
    """
    import urllib.request

    cache_dir = _TAU2_CACHE / domain
    cache_dir.mkdir(parents=True, exist_ok=True)

    files = {
        "tasks.json": f"{_TAU2_REPO}/data/tau2/domains/{domain}/tasks.json",
        "policy.md": f"{_TAU2_REPO}/data/tau2/domains/{domain}/policy.md",
        "db.json": f"{_TAU2_REPO}/data/tau2/domains/{domain}/db.json",
    }

    for filename, url in files.items():
        local_path = cache_dir / filename
        if not local_path.exists():
            logger.info(f"Downloading tau2-bench {domain}/{filename}...")
            try:
                urllib.request.urlretrieve(url, local_path)
            except Exception as e:
                logger.warning(f"Failed to download {url}: {e}")
                return [], "", {}, []

    # Load tool definitions (separate file, shared across all domains)
    tools_path = cache_dir / "tools.json"
    if not tools_path.exists():
        logger.info("Downloading tau2-bench tool definitions...")
        try:
            urllib.request.urlretrieve(_TAU2_TOOLS_URL, tools_path)
        except Exception as e:
            logger.warning(f"Failed to download tool definitions: {e}")
            return [], "", {}, []

    tasks = json.loads((cache_dir / "tasks.json").read_text())
    policy = (cache_dir / "policy.md").read_text()
    db = json.loads((cache_dir / "db.json").read_text())

    # Extract domain-specific tools and convert to OpenAI format
    all_tools_data = json.loads(tools_path.read_text())
    domain_tools_raw = all_tools_data.get(domain, {}).get("tools", [])
    tool_definitions = _convert_tau2_tools_to_openai(domain_tools_raw)

    logger.info(
        f"Loaded tau2-bench {domain}: {len(tasks)} tasks, {len(tool_definitions)} tools"
    )
    return tasks, policy, db, tool_definitions


# ---------------------------------------------------------------------------
# Tool execution engine — simulates the backend DB
# ---------------------------------------------------------------------------


class ToolBackend:
    """Simulated tool execution backend backed by an in-memory DB.

    Executes tool calls by exact name match against the dataset's tool
    definitions.  The tool schemas (OpenAI function-calling format) are
    used to understand parameter semantics — which arguments are IDs
    (for lookup) vs values (for mutation).

    For tools whose parameters reference a DB collection, the backend
    performs the appropriate read or write.  For tools that don't map to
    any collection, it returns a generic success response — the grading
    only checks that the right tools were called with the right arguments.
    """

    def __init__(self, db: dict[str, Any], tool_definitions: list[dict]):
        self.db = copy.deepcopy(db)
        self.tool_definitions = tool_definitions

        # Build a lookup: tool_name -> schema dict
        self._tools_by_name: dict[str, dict] = {}
        for t in tool_definitions:
            fn = t.get("function", t)
            name = fn.get("name", "")
            if name:
                self._tools_by_name[name] = fn

        self.call_log: list[dict] = []

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool call and return the result as a string."""
        self.call_log.append({"name": tool_name, "arguments": arguments})

        if tool_name not in self._tools_by_name:
            logger.warning(
                "Tool %r not found in schema. Available: %s",
                tool_name,
                sorted(self._tools_by_name.keys()),
            )
            return json.dumps(
                {
                    "error": f"Unknown tool: {tool_name}. Available: {sorted(self._tools_by_name.keys())}"
                }
            )

        try:
            result = self._execute_tool(tool_name, arguments)
            return json.dumps(result, default=str)
        except Exception as e:
            return json.dumps({"error": f"Tool execution failed: {e}"})

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a single tool against the DB using schema info."""
        schema = self._tools_by_name[tool_name]
        params = schema.get("parameters", {})
        param_props = params.get("properties", {})
        id_args = self._identify_id_args(param_props, args)
        value_args = {k: v for k, v in args.items() if k not in id_args}

        collection_name, collection = self._find_collection(tool_name, args)
        if collection is None:
            return {"status": "success", "tool": tool_name, "arguments": args}

        is_write = bool(value_args) and self._schema_looks_like_write(
            tool_name, param_props, value_args
        )

        if is_write:
            return self._handle_write(collection_name, collection, id_args, value_args)
        else:
            return self._handle_read(collection, args)

    @staticmethod
    def _identify_id_args(param_props: dict, args: dict) -> dict[str, Any]:
        """Pick out arguments that look like identifiers (for record lookup)."""
        id_keys: dict[str, Any] = {}
        for key, val in args.items():
            prop = param_props.get(key, {})
            desc = str(prop.get("description", "")).lower()
            key_lower = key.lower()
            if (
                key_lower.endswith("_id")
                or key_lower == "id"
                or key_lower.endswith("_number")
                or key_lower in ("email", "username", "phone")
                or "identifier" in desc
                or "unique" in desc
                or "id of" in desc
                or "id for" in desc
            ):
                id_keys[key] = val
        return id_keys

    @staticmethod
    def _schema_looks_like_write(tool_name: str, param_props: dict, value_args: dict) -> bool:
        """Decide if a tool call is a write (mutation) based on schema cues."""
        mutation_words = {
            "update", "set", "change", "modify", "new", "replace",
            "cancel", "create", "delete", "remove", "add",
        }
        for key in value_args:
            prop = param_props.get(key, {})
            desc = str(prop.get("description", "")).lower()
            if any(w in desc for w in mutation_words):
                return True
            if key.lower().startswith("new_"):
                return True
        return False

    def _find_collection(self, tool_name: str, args: dict) -> tuple[str, Any]:
        """Find the best-matching DB collection for a tool."""
        if not self.db:
            return "", None

        for coll_name, coll in self.db.items():
            if isinstance(coll, dict):
                for val in args.values():
                    if str(val) in coll:
                        return coll_name, coll

        tool_lower = tool_name.lower()
        tool_tokens = set(tool_lower.split("_"))
        best_name, best_coll, best_score = "", None, 0
        for coll_name, coll in self.db.items():
            if not isinstance(coll, (list, dict)):
                continue
            coll_lower = coll_name.lower()
            coll_tokens = set(coll_lower.split("_"))
            overlap = len(tool_tokens & coll_tokens)
            if coll_lower in tool_lower or tool_lower in coll_lower:
                overlap += 2
            for ct in coll_tokens:
                for tt in tool_tokens:
                    if (ct.startswith(tt) or tt.startswith(ct)) and abs(len(ct) - len(tt)) <= 1:
                        overlap += 1
            if overlap > best_score:
                best_name, best_coll, best_score = coll_name, coll, overlap
        if best_score > 0:
            return best_name, best_coll

        return "", None

    def _handle_read(self, collection: list | dict, args: dict) -> Any:
        """Query a collection using arguments as filters."""
        if isinstance(collection, dict):
            for val in args.values():
                key = str(val)
                if key in collection:
                    return collection[key]
            results = []
            for _entry_key, entry in collection.items():
                if isinstance(entry, dict) and self._matches(entry, args):
                    results.append(entry)
            if len(results) == 1:
                return results[0]
            return results if results else {"message": "No matching records found"}

        if isinstance(collection, list):
            results = [r for r in collection if isinstance(r, dict) and self._matches(r, args)]
            if len(results) == 1:
                return results[0]
            return results if results else []

        return collection

    def _handle_write(
        self,
        collection_name: str,
        collection: list | dict,
        id_args: dict[str, Any],
        value_args: dict[str, Any],
    ) -> Any:
        """Mutate a record in the collection, looked up by id_args."""
        if isinstance(collection, dict):
            for val in id_args.values():
                key = str(val)
                if key in collection and isinstance(collection[key], dict):
                    collection[key].update(value_args)
                    return {"status": "success", "updated": collection[key]}
            for _entry_key, entry in collection.items():
                if isinstance(entry, dict) and self._matches(entry, id_args):
                    entry.update(value_args)
                    return {"status": "success", "updated": entry}

        elif isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict) and self._matches(record, id_args):
                    record.update(value_args)
                    return {"status": "success", "updated": record}

        return {
            "status": "success",
            "tool": f"write to {collection_name}",
            "id": id_args,
            "values": value_args,
        }

    @staticmethod
    def _matches(record: dict, filters: dict) -> bool:
        """Check if a record matches all non-empty filter values."""
        if not filters:
            return True
        for key, val in filters.items():
            if val is None:
                continue
            record_val = record.get(key)
            if record_val is None:
                continue
            if str(record_val).lower() != str(val).lower():
                return False
        return True


# ---------------------------------------------------------------------------
# Tool wrappers — bridge ToolBackend to the cookbook Tool protocol
# ---------------------------------------------------------------------------


class _Tau2Tool:
    """Wraps a single tau2 tool definition + backend into the Tool protocol."""

    def __init__(self, openai_spec: dict, backend: ToolBackend) -> None:
        fn = openai_spec.get("function", openai_spec)
        self._name = fn["name"]
        self._description = fn.get("description", "")
        self._parameters_schema = fn.get("parameters", {})
        self._backend = backend

    @property
    def name(self) -> str:
        return self._name

    @property
    def description(self) -> str:
        return self._description

    @property
    def parameters_schema(self) -> dict[str, Any]:
        return self._parameters_schema

    def to_spec(self) -> ToolSpec:
        return ToolSpec(  # type: ignore[misc]
            name=self._name,
            description=self._description,
            parameters_schema=self._parameters_schema,
        )

    async def run(self, input: ToolInput) -> ToolResult:
        result_str = self._backend.execute(self._name, input.arguments)
        return simple_tool_result(
            result_str,
            call_id=input.call_id or "",
            name=self._name,
        )


# ---------------------------------------------------------------------------
# User simulator
# ---------------------------------------------------------------------------


class _FallbackCompleter:
    """Simple fallback completer that returns synthetic user responses."""

    async def __call__(self, messages: list[Message]) -> dict:
        return {"content": "I see, thank you. Is there anything else you need from me?"}


async def _simulate_user(
    user_completer: TinkerMessageCompleter | _FallbackCompleter,
    user_scenario: dict,
    history: list[Message],
    is_opening: bool = False,
) -> str:
    """Generate a simulated customer message.

    Args:
        user_completer: LLM completer for user simulation.
        user_scenario: Scenario dict from the task.
        history: Conversation history so far (agent perspective).
        is_opening: If True, generate the opening customer message.
    """
    scenario_text = json.dumps(user_scenario, indent=2)[:3000]

    if is_opening:
        prompt_messages: list[Message] = [
            {
                "role": "system",
                "content": (
                    "You are simulating a customer calling for support. "
                    "Follow the scenario below exactly. Stay in character. "
                    "State your problem or request naturally as a customer would."
                ),
            },
            {
                "role": "user",
                "content": f"Scenario:\n{scenario_text}\n\nGenerate your opening message as the customer.",
            },
        ]
    else:
        prompt_messages = [
            {
                "role": "system",
                "content": (
                    "You are simulating a customer in a support conversation. "
                    f"Follow this scenario:\n{scenario_text}\n\n"
                    "Respond naturally as the customer. If the agent has resolved "
                    "your issue, say goodbye and end with USER_STOP. "
                    "If the agent asks for information, provide it from your scenario. "
                    "If you don't have the information, say you don't know."
                ),
            },
        ]
        # Add conversation history (condensed, last 10 messages).
        for msg in history[-10:]:
            role = msg["role"]
            content = format_content_as_string(msg.get("content", ""))[:500]
            if role == "system":
                continue
            elif role == "assistant":
                # Agent's message → what the customer sees
                prompt_messages.append({"role": "user", "content": content})
            elif role == "user" and not content.startswith("[Tool"):
                # Customer's prior reply
                prompt_messages.append({"role": "assistant", "content": content})

        prompt_messages.append(
            {"role": "user", "content": "Continue as the customer. What do you say next?"}
        )

    try:
        response = await user_completer(prompt_messages)
        return str(response.get("content", "I see, thank you."))
    except Exception as e:
        logger.warning(f"User simulator failed: {e}")
        if is_opening:
            instructions = user_scenario.get("instructions", user_scenario)
            if isinstance(instructions, dict):
                task_instructions = instructions.get("task_instructions", "")
                reason = instructions.get("reason_for_call", "")
            else:
                task_instructions = str(instructions)
                reason = ""
            return f"Hi, {reason} {task_instructions}".strip() or "Hi, I need help with my account."
        return "I see, thank you. Is there anything else you need from me?"


# ---------------------------------------------------------------------------
# Grading
# ---------------------------------------------------------------------------


def _check_actions(predicted_calls: list[dict], expected_actions: list[dict]) -> tuple[float, dict]:
    """Check if predicted tool calls cover all expected actions.

    Uses set-based matching (order doesn't matter, extra calls not penalized).
    Returns (score, metrics).
    """
    if not expected_actions:
        return 1.0, {"actions_matched": 0, "actions_expected": 0}

    consumed: set[int] = set()
    matched = 0

    for expected in expected_actions:
        exp_name = str(expected.get("name", expected.get("action", ""))).lower()
        exp_args = expected.get("arguments", expected.get("parameters", {}))

        for pred_idx, pred in enumerate(predicted_calls):
            if pred_idx in consumed:
                continue
            pred_name = str(pred.get("name", "")).lower()
            if pred_name != exp_name:
                continue
            pred_args = pred.get("arguments", {})
            if isinstance(exp_args, dict) and isinstance(pred_args, dict):
                args_match = all(
                    str(pred_args.get(k, "")).lower() == str(v).lower()
                    for k, v in exp_args.items()
                    if v is not None
                )
            else:
                args_match = True
            if args_match:
                matched += 1
                consumed.add(pred_idx)
                break

    score = matched / len(expected_actions)
    return score, {
        "actions_matched": matched,
        "actions_expected": len(expected_actions),
        "actions_predicted": len(predicted_calls),
    }


# ---------------------------------------------------------------------------
# MessageEnv — handles tool calls + user simulation
# ---------------------------------------------------------------------------


@dataclass
class Tau2MessageEnv(MessageEnv):
    """Message-level env for tau2-bench.

    Handles the three-way interaction: model ↔ tools + model ↔ customer.

    On each step:
    - If the model made tool calls → execute them, return tool results
    - If no tool calls → simulate customer response, return it
    - Episode ends when model stops calling tools AND conversation ends,
      or max_turns is reached
    """

    tools: list[_Tau2Tool]
    initial_messages: list[Message]
    max_turns: int
    backend: ToolBackend
    expected_actions: list[dict]
    user_completer: TinkerMessageCompleter | _FallbackCompleter
    user_scenario: dict
    example_id: str

    history: list[Message] = field(default_factory=list)
    _turn_count: int = 0
    _tool_dict: dict[str, _Tau2Tool] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self._tool_dict = {t.name: t for t in self.tools}

    async def initial_observation(self) -> list[Message]:
        if not self.history:
            self.history = list(self.initial_messages)
        return self.history

    async def step(self, message: Message) -> MessageStepResult:
        self._turn_count += 1
        logs: dict[str, Any] = {}

        self.history.append(message)

        # Extract tool calls from the parsed message (native format via renderer)
        tool_calls: list[ToolCall] = list(message.get("tool_calls") or [])

        if tool_calls:
            # Execute each tool call against the backend
            for tc in tool_calls:
                args = tc.function.arguments
                if isinstance(args, str):
                    try:
                        args = json.loads(args)
                    except json.JSONDecodeError:
                        args = {}
                result = await self._tool_dict[tc.function.name].run(
                    ToolInput(arguments=args, call_id=tc.id)
                )
                for msg in result.messages:
                    self.history.append(msg)

            # After tool calls, continue (model gets tool results)
            if self._turn_count >= self.max_turns:
                return self._finalize(message, logs)
            return MessageStepResult(
                reward=0.0,
                episode_done=False,
                next_messages=self.history,
                metrics={"turn": float(self._turn_count)},
                logs=logs,
            )

        # No tool calls — this is a message to the customer
        assistant_text = get_text_content(message)

        # Check if agent signals end
        if self._is_agent_done(assistant_text) or self._turn_count >= self.max_turns:
            return self._finalize(message, logs)

        # Simulate customer response
        user_response = await _simulate_user(
            self.user_completer, self.user_scenario, self.history,
        )
        self.history.append({"role": "user", "content": user_response})

        # Check if customer signals end
        if "user_stop" in user_response.lower():
            return self._finalize(message, logs)

        return MessageStepResult(
            reward=0.0,
            episode_done=False,
            next_messages=self.history,
            metrics={"turn": float(self._turn_count)},
            logs=logs,
        )

    @staticmethod
    def _is_agent_done(text: str) -> bool:
        """Check if the agent signals conversation end."""
        lower = text.lower().strip()
        if "agent_stop" in lower or "conversation_complete" in lower:
            return True
        # Only trigger on very clear end signals to avoid premature termination
        return lower.endswith("goodbye.") or lower.endswith("goodbye!")

    def _finalize(self, last_message: Message, logs: dict) -> MessageStepResult:
        """Grade the completed conversation and return final result."""
        all_predicted_calls = self.backend.call_log
        score, action_metrics = _check_actions(all_predicted_calls, self.expected_actions)

        logs["example_id"] = self.example_id
        logs["num_turns"] = self._turn_count
        logs["predicted_actions"] = json.dumps(all_predicted_calls)[:500]
        logs["expected_actions"] = json.dumps(self.expected_actions)[:500]

        return MessageStepResult(
            reward=score,
            episode_done=True,
            next_messages=self.history,
            metrics={
                "correct": float(score >= 1.0),
                "action_score": score,
                **{k: float(v) for k, v in action_metrics.items()},
                "num_turns": float(self._turn_count),
            },
            logs=logs,
        )


# ---------------------------------------------------------------------------
# Env factory — creates sandbox-free env with MessageEnv pattern
# ---------------------------------------------------------------------------


class _Tau2BenchEnvFactory(Env):
    """Wrapper that sets up the MessageEnv on first observation.

    Creates the tool backend, user simulator, and initial messages,
    then delegates to EnvFromMessageEnv for native tool-call handling.
    """

    def __init__(
        self,
        *,
        system_prompt: str,
        tool_definitions: list[dict],
        user_scenario: dict,
        expected_actions: list[dict],
        db: dict[str, Any],
        user_completer: TinkerMessageCompleter | _FallbackCompleter,
        renderer: Renderer,
        example_id: str,
        max_trajectory_tokens: int | None = None,
        max_generation_tokens: int | None = None,
    ):
        self.system_prompt = system_prompt
        self.tool_definitions = tool_definitions
        self.user_scenario = user_scenario
        self.expected_actions = expected_actions
        self.db = db
        self.user_completer = user_completer
        self.renderer = renderer
        self.example_id = example_id
        self.max_trajectory_tokens = max_trajectory_tokens
        self.max_generation_tokens = max_generation_tokens

        self._inner: EnvFromMessageEnv | None = None

    async def initial_observation(self):
        # Create backend and tool wrappers
        backend = ToolBackend(self.db, self.tool_definitions)
        tools = [_Tau2Tool(spec, backend) for spec in self.tool_definitions]
        tool_specs = [t.to_spec() for t in tools]

        # Generate initial user message (customer opening)
        initial_user_msg = await _simulate_user(
            self.user_completer, self.user_scenario, [], is_opening=True,
        )

        # Build initial messages with tool specs in renderer's native format
        initial_messages = self.renderer.create_conversation_prefix_with_tools(
            tools=tool_specs, system_prompt=self.system_prompt,
        )
        initial_messages.append({"role": "user", "content": initial_user_msg})

        # Create MessageEnv
        msg_env = Tau2MessageEnv(
            tools=tools,
            initial_messages=initial_messages,
            max_turns=MAX_TURNS,
            backend=backend,
            expected_actions=self.expected_actions,
            user_completer=self.user_completer,
            user_scenario=self.user_scenario,
            example_id=self.example_id,
        )

        self._inner = EnvFromMessageEnv(
            renderer=self.renderer,
            message_env=msg_env,
            failed_parse_reward=0.0,
            terminate_on_parse_error=False,
            max_trajectory_tokens=self.max_trajectory_tokens,
            max_generation_tokens=self.max_generation_tokens,
            context_overflow_reward=0.0,
        )

        return await self._inner.initial_observation()

    async def step(self, action, *, extra=None):
        assert self._inner is not None
        return await self._inner.step(action, extra=extra)


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class Tau2BenchBenchmarkBuilder(BenchmarkBuilder):
    """tau2-Bench: multi-turn customer service agent evaluation.

    The model acts as a customer service agent, interacting with simulated
    customers and calling tools against a backend database. Uses the
    ``MessageEnv`` / ``EnvFromMessageEnv`` pattern for native tool-call
    parsing via the renderer.

    Requires ``config.judge_sampling_client`` for the user simulator.
    """

    name = "tau2_bench"
    requires_judge = True
    multi_turn = True
    recommended_timeout = 600

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        tasks, policy, db, tool_definitions = _load_tau2_data("airline")
        if not tasks:
            logger.warning("Could not load tau2-bench data.")
            return []
        if not tool_definitions:
            logger.warning("tau2_bench: no tool definitions loaded.")
        if config.max_examples is not None:
            tasks = tasks[: config.max_examples]

        j_client = config.judge_sampling_client
        j_renderer = config.judge_renderer or renderer
        if j_client is None:
            logger.warning(
                "tau2_bench: no judge_sampling_client configured. "
                "User simulation will use fallback responses. "
                "Set config.judge_sampling_client for proper evaluation."
            )
            user_completer = _FallbackCompleter()
        else:
            user_completer = TinkerMessageCompleter(
                sampling_client=j_client,
                renderer=j_renderer,
                max_tokens=2048,
                temperature=0.7,
            )

        envs = []
        for task in tasks:
            task_id = task.get("id")
            user_scenario = task.get("user_scenario", {})
            if isinstance(user_scenario, str):
                user_scenario = {"task_instructions": user_scenario}
            eval_criteria = task.get("evaluation_criteria", {})
            expected_actions = (
                eval_criteria.get("actions", []) if isinstance(eval_criteria, dict) else []
            )

            task_system_prompt = config.system_prompt or policy[:8000]

            example_id = (
                f"tau2_bench_{task_id}"
                if task_id is not None
                else make_example_id("tau2_bench", str(task))
            )

            envs.append(
                _Tau2BenchEnvFactory(
                    system_prompt=task_system_prompt,
                    tool_definitions=tool_definitions,
                    user_scenario=user_scenario,
                    expected_actions=expected_actions,
                    db=db,
                    user_completer=user_completer,
                    renderer=renderer,
                    example_id=example_id,
                    max_trajectory_tokens=config.max_trajectory_tokens,
                    max_generation_tokens=config.max_generation_tokens,
                )
            )

        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(Tau2BenchBenchmarkBuilder())
