"""tau2-Bench benchmark — multi-turn customer service agent evaluation.

**Status: Experimental** — needs significant work before production use:

1. **Data loading**: Dataset is NOT on HuggingFace. It ships with the
   ``tau2-bench`` GitHub repo (https://github.com/sierra-research/tau2-bench).
   Current implementation tries to load from HF and fails.

2. **Tool definitions**: The official benchmark defines domain-specific tools
   (airline booking, retail orders) as Python functions in the repo's source
   code. Our implementation uses a simplified simulated backend that may not
   match the official tool behavior.

3. **User simulation**: Requires an LLM to simulate the customer. Quality
   depends heavily on the simulator model.

4. **NL grading**: The official benchmark uses ``nl_assertions`` (natural
   language checks like "Agent should refuse the cancellation") which require
   an LLM judge. Our implementation only checks action-based criteria.

5. **Architecture**: Should be migrated to ``MessageEnv`` + ``EnvFromMessageEnv``
   pattern (like terminal_bench and swe_bench) for proper renderer integration.

Metric: Task completion rate.
Requires ``config.judge_sampling_client`` for the user simulator.
"""

from __future__ import annotations

import copy
import json
import logging
import re
from collections.abc import Sequence
from typing import Any, cast

import tinker
from datasets import Dataset

from tinker_cookbook.completers import TinkerMessageCompleter
from tinker_cookbook.eval.benchmarks._common import (
    limit_dataset,
    load_benchmark_dataset,
    make_example_id,
)
from tinker_cookbook.eval.benchmarks._types import BenchmarkBuilder, BenchmarkConfig
from tinker_cookbook.renderers import Message
from tinker_cookbook.renderers.base import Renderer
from tinker_cookbook.rl.types import Env, StepResult

logger = logging.getLogger(__name__)

MAX_TURNS = 30
"""Maximum number of agent turns before forced termination."""


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

    # ------------------------------------------------------------------
    # Core dispatch — schema-driven, no prefix heuristics
    # ------------------------------------------------------------------

    def _execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute a single tool against the DB using schema info."""
        schema = self._tools_by_name[tool_name]
        params = schema.get("parameters", {})
        param_props = params.get("properties", {})
        id_args = self._identify_id_args(param_props, args)
        value_args = {k: v for k, v in args.items() if k not in id_args}

        # Try to find a matching DB collection for this tool.
        collection_name, collection = self._find_collection(tool_name, args)
        if collection is None:
            # No DB collection maps to this tool — return generic success
            # with the arguments echoed back (grading checks call names/args).
            return {"status": "success", "tool": tool_name, "arguments": args}

        # Decide read vs write based on whether there are non-ID value args
        # that would mutate a record.  Pure-ID calls are lookups.
        is_write = bool(value_args) and self._schema_looks_like_write(
            tool_name, param_props, value_args
        )

        if is_write:
            return self._handle_write(collection_name, collection, id_args, value_args)
        else:
            return self._handle_read(collection, args)

    # ------------------------------------------------------------------
    # Schema analysis helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _identify_id_args(param_props: dict, args: dict) -> dict[str, Any]:
        """Pick out arguments that look like identifiers (for record lookup).

        Uses the schema ``description`` and well-known naming conventions
        (``*_id``, ``id``, ``*_number``, ``email``).
        """
        id_keys: dict[str, Any] = {}
        for key, val in args.items():
            prop = param_props.get(key, {})
            desc = str(prop.get("description", "")).lower()
            key_lower = key.lower()
            # Heuristic: key name or description indicates an identifier
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
        """Decide if a tool call is a write (mutation) based on schema cues.

        Checks the parameter descriptions for mutation language and whether
        the value args go beyond pure lookup filters.
        """
        # Check value-arg descriptions for mutation language
        mutation_words = {
            "update",
            "set",
            "change",
            "modify",
            "new",
            "replace",
            "cancel",
            "create",
            "delete",
            "remove",
            "add",
        }
        for key in value_args:
            prop = param_props.get(key, {})
            desc = str(prop.get("description", "")).lower()
            if any(w in desc for w in mutation_words):
                return True
            # "new_*" parameter names strongly suggest writes
            if key.lower().startswith("new_"):
                return True
        return False

    def _find_collection(self, tool_name: str, args: dict) -> tuple[str, Any]:
        """Find the best-matching DB collection for a tool.

        Strategy:
        1. Check if any argument value directly matches a key in a dict collection.
        2. Check if the tool name overlaps with a collection name.
        Returns (collection_name, collection) or ("", None).
        """
        if not self.db:
            return "", None

        # Strategy 1: Check if an arg value is a key in a dict collection
        for coll_name, coll in self.db.items():
            if isinstance(coll, dict):
                for val in args.values():
                    if str(val) in coll:
                        return coll_name, coll

        # Strategy 2: Fuzzy match tool name to collection name
        tool_lower = tool_name.lower()
        # Tokenize the tool name (split on _ and lowercase)
        tool_tokens = set(tool_lower.split("_"))
        best_name, best_coll, best_score = "", None, 0
        for coll_name, coll in self.db.items():
            if not isinstance(coll, (list, dict)):
                continue
            coll_lower = coll_name.lower()
            coll_tokens = set(coll_lower.split("_"))
            # Score by token overlap
            overlap = len(tool_tokens & coll_tokens)
            # Also check substring containment
            if coll_lower in tool_lower or tool_lower in coll_lower:
                overlap += 2
            # Singular/plural: "reservation" in "reservations"
            for ct in coll_tokens:
                for tt in tool_tokens:
                    if (ct.startswith(tt) or tt.startswith(ct)) and abs(len(ct) - len(tt)) <= 1:
                        overlap += 1
            if overlap > best_score:
                best_name, best_coll, best_score = coll_name, coll, overlap
        if best_score > 0:
            return best_name, best_coll

        return "", None

    # ------------------------------------------------------------------
    # Read / Write handlers
    # ------------------------------------------------------------------

    def _handle_read(self, collection: list | dict, args: dict) -> Any:
        """Query a collection using arguments as filters."""
        if isinstance(collection, dict):
            # Try direct key lookup with any argument value
            for val in args.values():
                key = str(val)
                if key in collection:
                    return collection[key]
            # Return all entries matching filter
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
            # Find record by ID arg value as dict key
            for val in id_args.values():
                key = str(val)
                if key in collection and isinstance(collection[key], dict):
                    collection[key].update(value_args)
                    return {"status": "success", "updated": collection[key]}
            # Fallback: find by matching and update
            for _entry_key, entry in collection.items():
                if isinstance(entry, dict) and self._matches(entry, id_args):
                    entry.update(value_args)
                    return {"status": "success", "updated": entry}

        elif isinstance(collection, list):
            for record in collection:
                if isinstance(record, dict) and self._matches(record, id_args):
                    record.update(value_args)
                    return {"status": "success", "updated": record}

        # No matching record — still report success for grading purposes
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
# Helpers
# ---------------------------------------------------------------------------


def _format_tool_definitions(tool_definitions: list[dict]) -> str:
    """Format tool definitions as human-readable function signatures.

    Converts OpenAI function-calling style schemas into clear descriptions
    that help the model understand each tool's name, purpose, and parameters.
    """
    lines: list[str] = []
    for tool in tool_definitions:
        fn = tool.get("function", tool)
        name = fn.get("name", "unknown")
        desc = fn.get("description", "")
        params = fn.get("parameters", {})
        props = params.get("properties", {})
        required = set(params.get("required", []))

        # Build parameter list
        param_parts: list[str] = []
        for pname, pschema in props.items():
            ptype = pschema.get("type", "any")
            pdesc = pschema.get("description", "")
            req_marker = " (required)" if pname in required else " (optional)"
            enum_vals = pschema.get("enum")
            enum_str = f", enum: {enum_vals}" if enum_vals else ""
            param_parts.append(
                f"    - {pname}: {ptype}{req_marker}{enum_str}" + (f" — {pdesc}" if pdesc else "")
            )

        lines.append(f"### {name}")
        if desc:
            lines.append(desc)
        if param_parts:
            lines.append("  Parameters:")
            lines.extend(param_parts)
        lines.append("")

    return "\n".join(lines)


def _extract_tool_calls(text: str) -> list[dict]:
    """Extract tool calls from model response text.

    Handles multiple formats:
    - JSON with "name"/"arguments" or "action"/"arguments"
    - Function call syntax: tool_name(arg1=val1, ...)
    """
    calls = []

    # Try JSON blocks first
    for match in re.finditer(r"```(?:json)?\s*\n(.*?)\n```", text, re.DOTALL):
        try:
            parsed = json.loads(match.group(1).strip())
            if isinstance(parsed, dict) and ("name" in parsed or "action" in parsed):
                calls.append(
                    {
                        "name": parsed.get("name", parsed.get("action", "")),
                        "arguments": parsed.get("arguments", parsed.get("parameters", {})),
                    }
                )
        except json.JSONDecodeError:
            pass

    if calls:
        return calls

    # Try inline JSON objects
    for match in re.finditer(r"\{", text):
        start = match.start()
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    candidate = text[start : i + 1]
                    if '"name"' in candidate or '"action"' in candidate:
                        try:
                            parsed = json.loads(candidate)
                            calls.append(
                                {
                                    "name": parsed.get("name", parsed.get("action", "")),
                                    "arguments": parsed.get(
                                        "arguments", parsed.get("parameters", {})
                                    ),
                                }
                            )
                        except json.JSONDecodeError:
                            pass
                    break

    return calls


def _is_stop_signal(text: str) -> bool:
    """Check if the model's response signals conversation end.

    Uses explicit stop tokens and end-of-conversation regex patterns
    to avoid false triggers on mid-sentence phrases.
    """
    lower = text.lower().strip()
    # Explicit stop tokens (instructed in system prompt)
    if "agent_stop" in lower or "conversation_complete" in lower:
        return True
    # End-of-conversation patterns — only match at sentence boundaries
    stop_patterns = [
        r"(?:^|\.\s+)is there anything else i can (?:help|assist)\b.*\??\s*$",
        r"(?:^|\.\s+)(?:have a (?:great|nice|good) day|thank you for (?:calling|contacting))\b.*$",
        r"\bgoodbye\b\s*[.!]?\s*$",
    ]
    return any(re.search(p, lower, re.MULTILINE) for p in stop_patterns)


def _check_actions(predicted_calls: list[dict], expected_actions: list[dict]) -> tuple[float, dict]:
    """Check if predicted tool calls cover all expected actions.

    Uses set-based matching (order doesn't matter, extra calls not penalized).
    Returns (score, metrics).
    """
    if not expected_actions:
        return 1.0, {"actions_matched": 0, "actions_expected": 0}

    # Track which predicted calls have been consumed (support duplicate tool names)
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
            # Check arguments match
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
# Fallback completer when no judge sampling client is provided
# ---------------------------------------------------------------------------


class _FallbackCompleter:
    """Simple fallback completer that returns synthetic user responses.

    Used when ``config.judge_sampling_client`` is ``None`` so that the
    benchmark can still run (with degraded user simulation quality).
    """

    async def __call__(self, messages: list[Message]) -> dict:
        return {"content": "I see, thank you. Is there anything else you need from me?"}


# ---------------------------------------------------------------------------
# Env
# ---------------------------------------------------------------------------


class Tau2BenchEnv(Env):
    """Multi-turn env for one tau2-Bench task.

    The model acts as a customer service agent. On each turn:
    1. Parse the model's response for tool calls or user-facing text
    2. If tool call → execute against backend DB → return tool result
    3. If text → send to user simulator → return user's response
    4. Track all tool calls for grading at termination
    """

    def __init__(
        self,
        system_prompt: str,
        tool_definitions: list[dict],
        user_scenario: dict,
        expected_actions: list[dict],
        db: dict[str, Any],
        user_completer: TinkerMessageCompleter | _FallbackCompleter,
        renderer: Renderer,
        example_id: str = "",
    ):
        self.system_prompt = system_prompt
        self.tool_definitions = tool_definitions
        self.user_scenario = user_scenario
        self.expected_actions = expected_actions
        self.renderer = renderer
        self.user_completer = user_completer
        self.example_id = example_id

        # Runtime state
        self.backend = ToolBackend(db, tool_definitions)
        self.messages: list[Message] = []
        self.turn_count = 0

    async def initial_observation(self):
        # System prompt with policy and tools
        tools_text = _format_tool_definitions(self.tool_definitions)
        system_content = (
            f"{self.system_prompt}\n\n"
            f"## Available Tools\n"
            f"You can call tools by responding with a JSON block:\n"
            f'```json\n{{"name": "<tool_name>", "arguments": {{...}}}}\n```\n\n'
            f"{tools_text}\n\n"
            f"When the conversation is complete, end your message with AGENT_STOP."
        )
        self.messages = [{"role": "system", "content": system_content}]

        # Generate initial user message from scenario
        initial_user_msg = await self._simulate_user_opening()
        self.messages.append({"role": "user", "content": initial_user_msg})

        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return model_input, stop

    async def step(self, action, *, extra=None):
        self.turn_count += 1
        # Use raw decode — tau2 needs tool calls which decode_response strips
        response_text = str(self.renderer.tokenizer.decode(action))

        # Append assistant message
        self.messages.append({"role": "assistant", "content": response_text})

        # Check for tool calls
        tool_calls = _extract_tool_calls(response_text)

        if tool_calls:
            # Execute tools and append results
            tool_results = []
            for call in tool_calls:
                result = self.backend.execute(call["name"], call.get("arguments", {}))
                tool_results.append(f"Tool `{call['name']}` returned:\n{result}")

            tool_response = "\n\n".join(tool_results)
            self.messages.append({"role": "user", "content": f"[Tool Results]\n{tool_response}"})

        # Check termination
        if _is_stop_signal(response_text) or self.turn_count >= MAX_TURNS:
            return self._finalize(response_text)

        # If no tool calls, this is a message to the user — simulate user response
        if not tool_calls:
            user_response = await self._simulate_user_response()
            self.messages.append({"role": "user", "content": user_response})

            # Check if user signals end
            if "user_stop" in user_response.lower() or "goodbye" in user_response.lower():
                return self._finalize(response_text)

        # Continue conversation
        model_input = self.renderer.build_generation_prompt(self.messages)
        stop = self.renderer.get_stop_sequences()
        return StepResult(
            reward=0.0,
            episode_done=False,
            next_observation=model_input,
            next_stop_condition=stop,
            metrics={"turn": self.turn_count},
            logs={},
        )

    def _finalize(self, last_response: str) -> StepResult:
        """Grade the completed conversation and return final StepResult."""
        all_predicted_calls = self.backend.call_log
        score, action_metrics = _check_actions(all_predicted_calls, self.expected_actions)

        return StepResult(
            reward=score,
            episode_done=True,
            next_observation=tinker.ModelInput.empty(),
            next_stop_condition=[],
            metrics={
                "correct": float(score >= 1.0),
                "action_score": score,
                **{k: float(v) for k, v in action_metrics.items()},
                "num_turns": self.turn_count,
            },
            logs={
                "example_id": self.example_id,
                "num_turns": self.turn_count,
                "predicted_actions": json.dumps(all_predicted_calls)[:500],
                "expected_actions": json.dumps(self.expected_actions)[:500],
                "last_response": last_response[:300],
            },
        )

    async def _simulate_user_opening(self) -> str:
        """Generate the initial user message from the scenario."""
        scenario_text = json.dumps(self.user_scenario, indent=2)[:3000]
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
        try:
            response = await self.user_completer(prompt_messages)
            return str(response.get("content", "Hi, I need help with my account."))
        except Exception as e:
            logger.warning(f"User simulator failed for opening: {e}")
            # Fallback: construct from scenario
            task_instructions = self.user_scenario.get("task_instructions", "")
            reason = self.user_scenario.get("reason_for_call", "")
            return f"Hi, {reason} {task_instructions}".strip() or "Hi, I need help with my account."

    async def _simulate_user_response(self) -> str:
        """Generate a user response based on the conversation and scenario."""
        scenario_text = json.dumps(self.user_scenario, indent=2)[:2000]
        sim_messages: list[Message] = [
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
        # Add conversation history (condensed).
        # The simulator plays the customer, so:
        # - Agent (assistant in main convo) → shown as "user" to simulator (what customer sees)
        # - Customer (user in main convo) → shown as "assistant" to simulator (its own prior replies)
        for msg in self.messages[-10:]:
            role = msg["role"]
            content = msg.get("content", "")[:500]
            if role == "system":
                continue
            elif role == "assistant":
                # Agent's message → what the customer sees → user role for simulator
                sim_messages.append({"role": "user", "content": content})
            elif role == "user" and not str(content).startswith("[Tool"):
                # Customer's prior reply → simulator's own output → assistant role
                sim_messages.append({"role": "assistant", "content": content})

        sim_messages.append(
            {"role": "user", "content": "Continue as the customer. What do you say next?"}
        )

        try:
            response = await self.user_completer(sim_messages)
            return str(response.get("content", "I see, thank you."))
        except Exception as e:
            logger.warning(f"User simulator failed: {e}")
            return "I see, thank you. Is there anything else you need from me?"


# ---------------------------------------------------------------------------
# Benchmark builder
# ---------------------------------------------------------------------------


class Tau2BenchBenchmarkBuilder(BenchmarkBuilder):
    """tau2-Bench: multi-turn customer service agent evaluation.

    The model acts as a customer service agent, interacting with simulated
    customers and calling tools against a backend database. Graded on whether
    all required actions were taken correctly.

    Requires ``config.judge_sampling_client`` for the user simulator.
    """

    name = "tau2_bench"
    requires_judge = True
    multi_turn = True
    recommended_timeout = 600

    def make_envs(self, renderer: Renderer, config: BenchmarkConfig) -> Sequence[Env]:
        ds = cast(Dataset, load_benchmark_dataset("sierra-research/tau2-bench"))
        ds = limit_dataset(ds, config.max_examples)

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
        for row in ds:
            row = dict(row)
            # Extract task data
            task_id = row.get("task_id", row.get("id"))
            system_prompt = row.get("system_prompt", row.get("instructions", row.get("policy", "")))
            tools = row.get("tools", row.get("available_actions", []))
            if isinstance(tools, str):
                try:
                    tools = json.loads(tools)
                except json.JSONDecodeError:
                    tools = []
            if not isinstance(tools, list):
                tools = []

            user_scenario = row.get("user_scenario", row.get("scenario", {}))
            if isinstance(user_scenario, str):
                try:
                    user_scenario = json.loads(user_scenario)
                except json.JSONDecodeError:
                    user_scenario = {"task_instructions": user_scenario}
            if not isinstance(user_scenario, dict):
                user_scenario = {}

            # Expected actions for grading
            eval_criteria = row.get("evaluation_criteria", row.get("gold_actions", {}))
            if isinstance(eval_criteria, str):
                try:
                    eval_criteria = json.loads(eval_criteria)
                except json.JSONDecodeError:
                    eval_criteria = {}
            expected_actions = []
            if isinstance(eval_criteria, dict):
                expected_actions = eval_criteria.get("actions", [])
            elif isinstance(eval_criteria, list):
                expected_actions = eval_criteria

            # Database
            db = row.get("db", row.get("database", row.get("initial_state", {})))
            if isinstance(db, str):
                try:
                    db = json.loads(db)
                except json.JSONDecodeError:
                    db = {}
            if not isinstance(db, dict):
                db = {}

            if task_id is not None:
                example_id = f"tau2_bench_{task_id}"
            else:
                example_id = make_example_id("tau2_bench", str(system_prompt))

            envs.append(
                Tau2BenchEnv(
                    system_prompt=str(system_prompt)[:8000],
                    tool_definitions=tools,
                    user_scenario=user_scenario,
                    expected_actions=expected_actions,
                    db=db,
                    user_completer=user_completer,
                    renderer=renderer,
                    example_id=example_id,
                )
            )

        return envs


# Auto-register
from tinker_cookbook.eval.benchmarks import register  # noqa: E402

register(Tau2BenchBenchmarkBuilder())
