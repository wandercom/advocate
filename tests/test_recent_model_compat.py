from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
import json

import pytest
from click.testing import CliRunner

from advocate.cli import main
from advocate.engine import _parse_findings_json, review as run_review
from advocate.models import Dimension, Persona, PersonaReport, Review, Severity
from advocate.provider import AnthropicProvider, LLMProvider, OpenAIProvider, create_provider
from advocate.report import print_review


def test_anthropic_default_uses_current_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ADVOCATE_MODEL", raising=False)
    monkeypatch.delenv("ADVOCATE_ANTHROPIC_MODEL", raising=False)

    provider = create_provider("anthropic")

    assert provider.model == "claude-opus-5"


def test_model_env_precedence(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ADVOCATE_MODEL", "generic-model")
    monkeypatch.setenv("ADVOCATE_ANTHROPIC_MODEL", "anthropic-model")

    assert create_provider("anthropic").model == "anthropic-model"
    assert create_provider("anthropic", "explicit-model").model == "explicit-model"


def _mock_anthropic_client() -> Mock:
    usage = SimpleNamespace(input_tokens=2, output_tokens=1)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="OK")],
        usage=usage,
    )
    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=response)
    return mock_client


@pytest.mark.asyncio
async def test_anthropic_provider_defaults_to_standard_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")

    with patch("anthropic.AsyncAnthropic", return_value=_mock_anthropic_client()) as client_cls:
        result = await AnthropicProvider("claude-sonnet-4-6").complete(
            "system prompt",
            "user prompt",
            16,
        )

    client_cls.assert_called_once_with(
        base_url="https://api.anthropic.com", api_key="generic-key"
    )
    assert result == ("OK", 2, 1)


@pytest.mark.asyncio
async def test_anthropic_provider_honors_key_order_env(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv(
        "ADVOCATE_ANTHROPIC_API_KEY_ENV", "ORG_ANTHROPIC_API_KEY, ANTHROPIC_API_KEY"
    )
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with patch("anthropic.AsyncAnthropic", return_value=_mock_anthropic_client()) as client_cls:
        await AnthropicProvider("claude-sonnet-4-6").complete("system", "user", 16)

    assert client_cls.call_args.kwargs["api_key"] == "org-key"


@pytest.mark.asyncio
async def test_anthropic_provider_honors_key_order_config_file(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    cfg = tmp_path / "advocate" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text(
        'anthropic_api_key_env = ["ORG_ANTHROPIC_API_KEY", "ANTHROPIC_API_KEY"]\n'
    )
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ORG_ANTHROPIC_API_KEY", "org-key")
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with patch("anthropic.AsyncAnthropic", return_value=_mock_anthropic_client()) as client_cls:
        await AnthropicProvider("claude-sonnet-4-6").complete("system", "user", 16)

    # The file's order, not the default, decides between two set keys.
    assert client_cls.call_args.kwargs["api_key"] == "org-key"


def test_configured_names_all_unset_does_not_fall_back_to_sdk_default(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from advocate.provider import _anthropic_api_key

    monkeypatch.setenv("ADVOCATE_ANTHROPIC_API_KEY_ENV", "ORG_ANTHROPIC_API_KEY")
    monkeypatch.delenv("ORG_ANTHROPIC_API_KEY", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "excluded-key")

    with pytest.raises(RuntimeError, match="ORG_ANTHROPIC_API_KEY"):
        _anthropic_api_key()


@pytest.mark.parametrize("override", [",", " , "])
def test_empty_key_order_override_is_rejected(
    monkeypatch: pytest.MonkeyPatch, override: str
) -> None:
    from advocate.provider import _anthropic_api_key

    monkeypatch.setenv("ADVOCATE_ANTHROPIC_API_KEY_ENV", override)
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with pytest.raises(RuntimeError, match="names no environment variables"):
        _anthropic_api_key()


def test_empty_key_order_config_is_rejected(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from advocate.provider import _anthropic_api_key

    cfg = tmp_path / "advocate" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("anthropic_api_key_env = []\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")

    with pytest.raises(RuntimeError, match="names no environment variables"):
        _anthropic_api_key()


def test_malformed_key_config_fails_loudly(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    from advocate.provider import _anthropic_api_key

    cfg = tmp_path / "advocate" / "config.toml"
    cfg.parent.mkdir(parents=True)
    cfg.write_text("anthropic_api_key_env = 3\n")
    monkeypatch.setenv("XDG_CONFIG_HOME", str(tmp_path))

    with pytest.raises(RuntimeError, match="anthropic_api_key_env"):
        _anthropic_api_key()


@pytest.mark.asyncio
async def test_anthropic_provider_ignores_ambient_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Regression (2026-08-30): the SDK reads ANTHROPIC_BASE_URL by default,
    so a coding-agent gateway in the ambient shell hijacked Advocate's
    requests and 404ed on models the billing key has. Advocate must pin the
    public API unless ADVOCATE_ANTHROPIC_BASE_URL opts in explicitly."""
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")
    monkeypatch.setenv("ANTHROPIC_BASE_URL", "https://gateway.invalid")
    monkeypatch.delenv("ADVOCATE_ANTHROPIC_BASE_URL", raising=False)
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="OK")], usage=usage
    )
    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client) as client_cls:
        await AnthropicProvider("claude-opus-5").complete("system", "user", 16)

    assert client_cls.call_args.kwargs["base_url"] == "https://api.anthropic.com"


@pytest.mark.asyncio
async def test_anthropic_provider_honors_explicit_advocate_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("ANTHROPIC_API_KEY", "generic-key")
    monkeypatch.setenv("ADVOCATE_ANTHROPIC_BASE_URL", "https://pinned.example")
    usage = SimpleNamespace(input_tokens=1, output_tokens=1)
    response = SimpleNamespace(
        content=[SimpleNamespace(type="text", text="OK")], usage=usage
    )
    mock_client = Mock()
    mock_client.messages.create = AsyncMock(return_value=response)

    with patch("anthropic.AsyncAnthropic", return_value=mock_client) as client_cls:
        await AnthropicProvider("claude-opus-5").complete("system", "user", 16)

    assert client_cls.call_args.kwargs["base_url"] == "https://pinned.example"


@pytest.mark.asyncio
async def test_openai_provider_uses_responses_api_for_recent_models() -> None:
    usage = SimpleNamespace(input_tokens=11, output_tokens=7)
    response = SimpleNamespace(output_text="ok", usage=usage)
    mock_client = Mock()
    mock_client.responses.create = AsyncMock(return_value=response)

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        text, input_tokens, output_tokens = await OpenAIProvider("gpt-5.4-mini").complete(
            "system prompt",
            "user prompt",
            128,
        )

    assert text == "ok"
    assert input_tokens == 11
    assert output_tokens == 7
    mock_client.responses.create.assert_awaited_once_with(
        model="gpt-5.4-mini",
        instructions="system prompt",
        input="user prompt",
        max_output_tokens=128,
    )


@pytest.mark.asyncio
async def test_openai_provider_forwards_opt_in_reasoning_effort(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    usage = SimpleNamespace(input_tokens=3, output_tokens=1)
    response = SimpleNamespace(output_text="answer", output=[], usage=usage)
    mock_client = Mock()
    mock_client.responses.create = AsyncMock(return_value=response)
    monkeypatch.setenv("OPENAI_REASONING_EFFORT", "none")

    with patch("openai.AsyncOpenAI", return_value=mock_client):
        result = await OpenAIProvider("qwen3.5:cloud").complete(
            "system",
            "user",
            123,
        )

    assert result == ("answer", 3, 1)
    mock_client.responses.create.assert_awaited_once_with(
        model="qwen3.5:cloud",
        instructions="system",
        input="user",
        max_output_tokens=123,
        reasoning={"effort": "none"},
    )


class FailingProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "test"

    async def complete(self, system: str, user: str, max_tokens: int = 4096) -> tuple[str, int, int]:
        raise RuntimeError("model 404")


class EmptyFindingsProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "test"

    async def complete(self, system: str, user: str, max_tokens: int = 4096) -> tuple[str, int, int]:
        return "```json\n[]\n```\nSummary: no issues found.", 10, 5


class ShapeDriftProvider(LLMProvider):
    @property
    def provider_name(self) -> str:
        return "test"

    async def complete(self, system: str, user: str, max_tokens: int = 4096) -> tuple[str, int, int]:
        findings = [
            {
                "severity": "HIGH",
                "dimension": "failure-modes",
                "title": {"nested": "dict title"},
                "detail": ["list", "detail"],
                "evidence": {"line": 12},
                "recommendation": None,
            },
            "not-json",
            {
                "severity": 123,
                "dimension": None,
                "title": 42,
                "detail": True,
            },
        ]
        return json.dumps(json.dumps(findings)), 10, 5


@pytest.mark.asyncio
async def test_review_marks_persona_provider_failures_incomplete() -> None:
    result = await run_review(
        content="def f(): pass",
        target="example.py",
        target_type="file",
        llm=FailingProvider("retired-model"),
        personas=[Persona.red_team],
    )

    assert result.total_findings == 0
    assert not result.is_complete()
    assert result.failed_reports()[0].error == "model 404"


@pytest.mark.asyncio
async def test_valid_empty_findings_json_is_not_parse_failure() -> None:
    result = await run_review(
        content="def f(): pass",
        target="example.py",
        target_type="file",
        llm=EmptyFindingsProvider("current-model"),
        personas=[Persona.good_friend],
    )

    assert result.total_findings == 0
    assert result.is_complete()
    assert result.persona_reports[0].error is None
    assert "PARSE_FAILED" not in result.persona_reports[0].summary


@pytest.mark.asyncio
async def test_stringified_and_misshapen_findings_are_coerced_per_item() -> None:
    result = await run_review(
        content="def f(): pass",
        target="example.py",
        target_type="file",
        llm=ShapeDriftProvider("current-model"),
        personas=[Persona.red_team],
    )

    assert result.is_complete()
    assert result.total_findings == 2
    first, second = result.all_findings()
    assert first.severity == Severity.high
    assert first.dimension == Dimension.failure_modes
    assert "dict title" in first.title
    assert "list" in first.detail
    assert "line" in first.evidence
    assert second.severity == Severity.medium
    assert second.dimension == Dimension.concept
    assert second.title == "42"
    assert second.detail == "True"


def test_print_review_does_not_render_failed_persona_as_positive(capsys: pytest.CaptureFixture[str]) -> None:
    result = Review(
        target="example.py",
        target_type="file",
        persona_reports=[
            PersonaReport(
                persona=Persona.red_team,
                ok=False,
                error="model 404",
                summary="FAILED: model 404",
            )
        ],
    )

    print_review(result, color=False)

    output = capsys.readouterr().out
    assert "REVIEW INCOMPLETE: 1/1 personas failed" in output
    assert "PERSONA FAILED: model 404" in output
    assert "strong positive signal" not in output


def test_cli_exits_nonzero_when_review_incomplete(monkeypatch: pytest.MonkeyPatch) -> None:
    import advocate.engine
    import advocate.provider
    import advocate.report

    class DummyProvider(LLMProvider):
        @property
        def provider_name(self) -> str:
            return "test"

        async def preflight(self) -> None:
            return None

        async def complete(self, system: str, user: str, max_tokens: int = 4096) -> tuple[str, int, int]:
            return "", 0, 0

    async def fake_review(**kwargs: object) -> Review:
        return Review(
            target="<stdin>",
            target_type="stdin",
            persona_reports=[
                PersonaReport(
                    persona=Persona.red_team,
                    ok=False,
                    error="model 404",
                    summary="FAILED: model 404",
                )
            ],
        )

    monkeypatch.setattr(advocate.provider, "create_provider", lambda provider, model: DummyProvider("dummy"))
    monkeypatch.setattr(advocate.engine, "review", fake_review)
    monkeypatch.setattr(advocate.report, "print_review", lambda review, color=True: None)

    result = CliRunner().invoke(main, ["review", "--stdin", "-p", "red_team"], input="content")

    assert result.exit_code == 2
    assert "REVIEW INCOMPLETE: 1/1 personas failed" in result.output


# ---- findings-JSON parsing regressions (2026-08-30 whole-factory run) ----
#
# These live here, not in tests/*/contract_test.py: pytest collects only
# test_*.py (pyproject python_files), so a regression added to a contract
# file never runs.


def test_parse_findings_json_brackets_inside_strings() -> None:
    """A depth counter sees brackets inside quoted evidence -- code like
    d["key"] or list[Finding] -- and cuts the array mid-string. raw_decode
    is string-aware and must not."""
    text = (
        '[{"title": "A", "evidence": "d[\\"key\\"] and list[Finding]"}, '
        '{"title": "B"}]\n\nOverall the design holds.'
    )
    findings, summary = _parse_findings_json(text)
    assert len(findings) == 2
    assert findings[0]["title"] == "A"
    assert findings[1]["title"] == "B"
    assert "the design holds" in summary


def test_parse_findings_json_largest_array_wins() -> None:
    """A stray empty array in prose must not shadow the findings array."""
    text = 'No findings so far []. But then: [{"title": "Real"}]'
    findings, _ = _parse_findings_json(text)
    assert len(findings) == 1
    assert findings[0]["title"] == "Real"


def test_parse_findings_json_salvages_valid_objects() -> None:
    """A single malformed finding no longer costs the whole response."""
    text = '[{"title": "A"}, {"title": broken}, {"title": "C"}]'
    findings, _ = _parse_findings_json(text)
    titles = [f.get("title") for f in findings]
    assert "A" in titles
    assert "C" in titles
    assert "broken" not in titles
