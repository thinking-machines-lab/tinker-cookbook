"""Tests for taubench type system — enums and dataclasses."""

from tinker_cookbook.recipes.taubench.components.types import (
    AskSonnetMode,
    ExplorationMode,
    ActionType,
    ObservationType,
    ParsedAction,
    Tau2StepResult,
    ExternalLLMConfig,
)


class TestAskSonnetMode:
    def test_values(self):
        assert AskSonnetMode.DIRECT_INJECTION.value == "direct"
        assert AskSonnetMode.CONDITIONING.value == "conditioning"

    def test_members(self):
        assert set(AskSonnetMode) == {AskSonnetMode.DIRECT_INJECTION, AskSonnetMode.CONDITIONING}


class TestExplorationMode:
    def test_values(self):
        assert ExplorationMode.EPSILON_GREEDY.value == "epsilon"
        assert ExplorationMode.RAO_BLACKWELL.value == "rao_blackwell"

    def test_members(self):
        assert set(ExplorationMode) == {
            ExplorationMode.EPSILON_GREEDY,
            ExplorationMode.RAO_BLACKWELL,
        }


class TestActionType:
    def test_values(self):
        assert ActionType.TOOL_CALL.value == "tool_call"
        assert ActionType.ASK_SONNET.value == "ask_sonnet"
        assert ActionType.TEXT.value == "text"


class TestObservationType:
    def test_values(self):
        assert ObservationType.USER_MESSAGE.value == "user"
        assert ObservationType.TOOL_RESULT.value == "tool"
        assert ObservationType.OTHER.value == "other"


class TestParsedAction:
    def test_defaults(self):
        pa = ParsedAction(raw_content="hello", action_type=ActionType.TEXT)
        assert pa.tool_name is None
        assert pa.tool_args is None
        assert pa.parse_success is True
        assert pa.original_message == {}

    def test_tool_call_construction(self):
        pa = ParsedAction(
            raw_content='{"name":"get_order","arguments":{}}',
            action_type=ActionType.TOOL_CALL,
            tool_name="get_order",
            tool_args={"order_id": "123"},
        )
        assert pa.action_type == ActionType.TOOL_CALL
        assert pa.tool_name == "get_order"
        assert pa.tool_args == {"order_id": "123"}

    def test_ask_sonnet_construction(self):
        pa = ParsedAction(
            raw_content="ask_sonnet",
            action_type=ActionType.ASK_SONNET,
            tool_name="ask_sonnet",
            tool_args={},
        )
        assert pa.action_type == ActionType.ASK_SONNET


class TestTau2StepResult:
    def test_defaults(self):
        sr = Tau2StepResult(
            obs_type=ObservationType.USER_MESSAGE,
            obs_content="hello",
            raw_obs="user: hello",
            reward=0.0,
            terminated=False,
            truncated=False,
        )
        assert sr.info == {}
        assert sr.reward == 0.0

    def test_full_construction(self):
        sr = Tau2StepResult(
            obs_type=ObservationType.TOOL_RESULT,
            obs_content="result data",
            raw_obs="tool: result data",
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"task_id": "t1"},
        )
        assert sr.terminated is True
        assert sr.info == {"task_id": "t1"}


class TestExternalLLMConfig:
    def test_defaults(self):
        cfg = ExternalLLMConfig(model="claude-sonnet-4-5-20250929")
        assert cfg.temperature == 0.0
        assert cfg.max_tokens == 1024

    def test_custom(self):
        cfg = ExternalLLMConfig(model="gpt-4", temperature=0.7, max_tokens=2048)
        assert cfg.model == "gpt-4"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 2048
