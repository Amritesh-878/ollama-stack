"""The harness is the only component whose failure modes are silent, so each one gets a test."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

import pytest
import responses

from ollama_stack.client import OllamaClient, Reply, conversation_text, estimate_tokens
from ollama_stack.implement import handoff, harness, tools
from ollama_stack.implement.handoff import changes
from ollama_stack.implement.harness import Outcome
from ollama_stack.implement.tools import (
    DEFAULT_GATE,
    Files,
    Gate,
    GateResult,
    GateRun,
    ToolError,
    as_list,
    executes,
    is_junk,
    run_gate,
)

CHAT = "http://127.0.0.1:11434/api/chat"


def _repo(root: Path) -> Path:
    """A real git repo, because dirty-tree refusal and numstat are not worth faking."""
    root.mkdir(parents=True, exist_ok=True)
    subprocess.run(["git", "init", "-q"], cwd=root, check=True)
    (root / "sample.py").write_text("alpha = 1\nbeta = 2\nalpha = 1\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=root, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "init"],
        cwd=root,
        check=True,
    )
    return root


def _call(name: str, **args: Any) -> dict[str, Any]:
    return {"function": {"name": name, "arguments": args}}


def _outcome(**over: Any) -> harness.Outcome:
    made = harness.Outcome(turns=1, finished=True, summary="did the thing")
    for key, value in over.items():
        setattr(made, key, value)
    return made


def _gate(code: int, output: str = "1 failed, 267 passed") -> GateResult:
    return GateResult((GateRun("tests", "uv run pytest", code, output),))


def test_edit_applies_a_unique_match(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    files = Files(repo)
    assert "edited" in files.edit("sample.py", "beta = 2", "beta = 3")
    assert (repo / "sample.py").read_text(encoding="utf-8") == "alpha = 1\nbeta = 3\nalpha = 1\n"


def test_edit_refuses_a_non_unique_match_and_changes_nothing(tmp_path: Path) -> None:
    """A first-match guess edits the wrong occurrence, which is as bad as regenerating the file."""
    repo = _repo(tmp_path / "r")
    before = (repo / "sample.py").read_text(encoding="utf-8")
    with pytest.raises(ToolError) as caught:
        Files(repo).edit("sample.py", "alpha = 1", "alpha = 9")
    assert "appears 2 times" in str(caught.value)
    assert (repo / "sample.py").read_text(encoding="utf-8") == before


def test_edit_refuses_an_absent_match_and_names_the_likely_cause(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(ToolError) as caught:
        Files(repo).edit("sample.py", "gamma = 3", "gamma = 4")
    assert "line-number prefixes" in str(caught.value)


def test_edit_refuses_an_empty_old_string(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(ToolError):
        Files(repo).edit("sample.py", "", "anything")


def test_write_refuses_a_path_that_already_exists(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(ToolError) as caught:
        Files(repo).create("sample.py", "replaced")
    assert "Use edit_file" in str(caught.value)


def test_write_creates_a_new_file_and_its_parent(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    Files(repo).create("pkg/new.py", "x = 1\n")
    assert (repo / "pkg" / "new.py").read_text(encoding="utf-8") == "x = 1\n"


def test_write_refuses_to_create_build_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(ToolError):
        Files(repo).create("__pycache__/x.pyc", "junk")


def test_a_path_outside_the_repo_is_refused(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    (tmp_path / "outside.txt").write_text("secret", encoding="utf-8")
    with pytest.raises(ToolError) as caught:
        Files(repo).read("../outside.txt")
    assert "resolves outside" in str(caught.value)


def test_the_workspace_is_readable_so_a_cited_contract_can_be_opened(tmp_path: Path) -> None:
    """A run that could not read a contract its task file cited proceeded partly blind."""
    repo = _repo(tmp_path / "r")
    space = tmp_path / "specs"
    (space / "contracts").mkdir(parents=True)
    (space / "contracts" / "FEAT.md").write_text("the shape\n", encoding="utf-8")
    assert "the shape" in Files(repo, space).read("contracts/FEAT.md")


def test_the_workspace_is_read_only(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    space = tmp_path / "specs"
    space.mkdir()
    (space / "notes.md").write_text("original\n", encoding="utf-8")
    files = Files(repo, space)
    with pytest.raises(ToolError):
        files.edit("../specs/notes.md", "original", "changed")
    with pytest.raises(ToolError):
        files.create("../specs/added.md", "new")
    assert (space / "notes.md").read_text(encoding="utf-8") == "original\n"


def test_no_workspace_means_only_the_repo_and_the_message_says_so(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    with pytest.raises(ToolError) as caught:
        Files(repo).read("absent.md")
    assert "No workspace is mounted" in str(caught.value)


def test_listing_hides_build_output(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "sample.pyc").write_text("junk", encoding="utf-8")
    subprocess.run(["git", "add", "-Af"], cwd=repo, check=True)
    listed = Files(repo).listing()
    assert "sample.py" in listed
    assert ".pyc" not in listed


def test_a_stringified_file_list_recovers_to_a_real_list() -> None:
    """files_changed came back once as the string "['a.py', 'b.py']" instead of an array."""
    assert as_list("['a.py', 'b.py']") == ["a.py", "b.py"]
    assert as_list(["a.py"]) == ["a.py"]
    assert as_list("a.py") == ["a.py"]
    assert as_list("") == []
    assert as_list(None) == []
    assert as_list(12) == []


def test_junk_is_recognised_by_directory_and_by_suffix() -> None:
    assert is_junk("src/__pycache__/x.pyc")
    assert is_junk("build/thing.pyo")
    assert is_junk(".pytest_cache/v/cache")
    assert not is_junk("src/ollama_stack/cli.py")


def test_the_gate_stops_at_the_first_failure_so_tests_never_run_over_a_red_typecheck(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "r")
    gate = (
        Gate("lint", ("git", "--version")),
        Gate("typecheck", ("git", "nonexistent-subcommand")),
        Gate("tests", ("git", "--version")),
    )
    result = run_gate(repo, gate)
    assert [run.label for run in result.runs] == ["lint", "typecheck"]
    assert result.passed is False
    assert result.failures == ["typecheck"]


def test_a_missing_gate_tool_is_reported_rather_than_crashing(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    result = run_gate(repo, (Gate("lint", ("definitely-not-a-real-binary",)),))
    assert result.passed is False
    assert "not on PATH" in result.runs[0].output


def test_the_gate_runs_in_the_target_repo_through_uv_not_the_drivers_own_tools(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The driver interpreter must stay independent of the repo it is changing."""
    repo = _repo(tmp_path / "r")
    seen: list[dict[str, Any]] = []

    def fake(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append({"command": tuple(command), "cwd": kwargs.get("cwd")})
        return subprocess.CompletedProcess(command, 0, "ok", "")

    monkeypatch.setattr(subprocess, "run", fake)
    # Said outright rather than left to the machine: whether uv is installed here is not
    # what this test is about, and letting it decide made the test pass for that reason.
    fake_bin = tmp_path / "bin"
    monkeypatch.setattr(tools, "on_path", lambda name, path=None: str(fake_bin / name))
    run_gate(repo, DEFAULT_GATE)
    binaries = [Path(entry["command"][0]) for entry in seen]
    assert [b.stem for b in binaries] == ["uv", "uv", "uv"]
    # Absolute, so the repo being judged cannot supply the tool that judges it.
    assert all(b.is_absolute() for b in binaries), binaries
    assert {entry["cwd"] for entry in seen} == {repo}


def test_git_is_run_by_full_path_not_by_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The repo being inspected is the last directory allowed to supply the git that reads it."""
    repo = _repo(tmp_path / "r")
    seen: list[tuple[str, ...]] = []

    def fake(command: Any, **kwargs: Any) -> subprocess.CompletedProcess[str]:
        seen.append(tuple(command))
        return subprocess.CompletedProcess(command, 0, "", "")

    fake_bin = tmp_path / "bin"
    monkeypatch.setattr(tools, "on_path", lambda name, path=None: str(fake_bin / name))
    monkeypatch.setattr(subprocess, "run", fake)
    tools.git(repo, "status", "--porcelain")
    assert seen[0][0] == str(fake_bin / "git")
    assert Path(seen[0][0]).is_absolute()


def test_no_git_on_path_is_refused_rather_than_run_as_a_bare_name(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = _repo(tmp_path / "r")
    monkeypatch.setattr(tools, "on_path", lambda name, path=None: None)

    def never(*args: Any, **kwargs: Any) -> Any:
        raise AssertionError("git was run without being found on PATH")

    monkeypatch.setattr(subprocess, "run", never)
    with pytest.raises(ToolError, match="git is not on PATH"):
        tools.git(repo, "status", "--porcelain")


def test_a_gate_tool_that_is_not_installed_fails_the_run_rather_than_passing_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No uv means nothing was checked, and nothing checked must never read as checked."""
    repo = _repo(tmp_path / "r")
    monkeypatch.setattr(tools, "on_path", lambda name, path=None: None)
    result = run_gate(repo, DEFAULT_GATE)
    assert not result.passed
    assert result.runs[0].code == 127
    assert "not on PATH" in result.runs[0].output


def test_a_failing_gate_produces_a_failed_handoff_whatever_the_model_claims(
    tmp_path: Path,
) -> None:
    """The exit code decides. This is the defence against success declared over red tests."""
    repo = _repo(tmp_path / "r")
    outcome = _outcome(summary="All tests pass and the task is complete.", gate=_gate(1))
    text = handoff.render(outcome, repo, "TRIAL", "some:tag", "local/trial")
    assert outcome.passed is False
    assert "**Status:** FAILED" in text
    assert "not from the model's summary" in text
    assert "All tests pass" in text


def test_a_passing_gate_produces_a_passed_handoff_that_still_says_unaudited(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "r")
    text = handoff.render(_outcome(gate=_gate(0, "265 passed")), repo, "T", "t", "local/t")
    assert "**Status:** PASSED" in text
    assert "NOT AUDITED. DO NOT MERGE." in text


def test_a_run_that_never_ran_the_gate_says_nothing_was_checked(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    text = handoff.render(_outcome(), repo, "T", "t", "local/t")
    assert "The gate never ran" in text
    assert "**Status:** FAILED" in text


def test_the_handoff_reports_a_reported_versus_actual_discrepancy_both_ways(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "r")
    (repo / "sample.py").write_text("alpha = 9\n", encoding="utf-8")
    outcome = _outcome(reported=["docs/never.md"], gate=_gate(0))
    text = handoff.render(outcome, repo, "T", "t", "local/t")
    assert "reported but not changed: `docs/never.md`" in text
    assert "changed but not reported: `sample.py`" in text


def test_the_handoff_says_so_when_the_report_and_the_diff_agree(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    (repo / "sample.py").write_text("alpha = 9\nbeta = 2\nalpha = 1\n", encoding="utf-8")
    text = handoff.render(_outcome(reported=["sample.py"], gate=_gate(0)), repo, "T", "t", "b")
    assert "agree exactly" in text


def test_the_handoff_flags_a_whole_file_rewrite(tmp_path: Path) -> None:
    """A 900-line diff for a five-line task is invisible to the suite and obvious here."""
    repo = _repo(tmp_path / "r")
    big = repo / "big.py"
    big.write_text("\n".join(f"row = {n}" for n in range(900)) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "big"],
        cwd=repo,
        check=True,
    )
    big.write_text("\n".join(f"row = {n * 2}" for n in range(900)) + "\n", encoding="utf-8")
    text = handoff.render(_outcome(reported=["big.py"], gate=_gate(0)), repo, "T", "t", "b")
    assert "| `big.py` |" in text
    assert "**YES**" in text
    assert "at least half its lines touched" in text


def test_a_five_line_change_to_a_900_line_file_is_not_flagged_as_a_rewrite(
    tmp_path: Path,
) -> None:
    """The headline criterion: a surgical edit must read as surgical in the handoff."""
    repo = _repo(tmp_path / "r")
    big = repo / "big.py"
    rows = [f"row = {n}" for n in range(900)]
    big.write_text("\n".join(rows) + "\n", encoding="utf-8")
    subprocess.run(["git", "add", "-A"], cwd=repo, check=True)
    subprocess.run(
        ["git", "-c", "user.email=t@t", "-c", "user.name=t", "commit", "-qm", "big"],
        cwd=repo,
        check=True,
    )
    files = Files(repo)
    for index in range(5):
        files.edit("big.py", f"row = {index * 100}\n", f"row = {index * 100} + 1\n")
    changes = handoff.changes(repo)
    assert len(changes) == 1
    assert (changes[0].added, changes[0].removed) == (5, 5)
    assert changes[0].rewritten is False


def test_the_handoff_reports_build_output_sitting_in_the_tree(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    cache = repo / "__pycache__"
    cache.mkdir()
    (cache / "sample.pyc").write_text("junk", encoding="utf-8")
    text = handoff.render(_outcome(gate=_gate(0)), repo, "T", "t", "b")
    assert "build output is sitting in the tree" in text


def test_the_gate_generates_caches_and_they_never_reach_the_index(tmp_path: Path) -> None:
    """__pycache__ broke a final `git add` before, so a real run must be checked for it."""
    repo = _repo(tmp_path / "r")
    run_gate(repo, (Gate("compile", ("git", "--version")),))
    (repo / "__pycache__").mkdir()
    (repo / "__pycache__" / "sample.pyc").write_text("junk", encoding="utf-8")
    assert handoff.staged_junk(repo) == ["__pycache__/"]
    tracked = Files(repo).listing()
    assert ".pyc" not in tracked


def test_a_dirty_tree_is_detected(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    assert harness.dirty(repo) == ""
    (repo / "sample.py").write_text("changed\n", encoding="utf-8")
    assert "sample.py" in harness.dirty(repo)


def test_work_happens_on_a_local_branch(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    harness.branch(repo, "local/trial")
    current = subprocess.run(
        ["git", "branch", "--show-current"], cwd=repo, capture_output=True, text=True, check=True
    ).stdout.strip()
    assert current == "local/trial"


def test_nothing_in_the_harness_can_push_or_commit() -> None:
    """The isolation guarantee is that the code to break it does not exist."""
    for module in ("harness", "tools", "handoff"):
        source = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "ollama_stack"
            / "implement"
            / f"{module}.py"
        ).read_text(encoding="utf-8")
        for forbidden in ('"push"', '"commit"', '"merge"', '"reset"', '"add"'):
            assert forbidden not in source, f"{module}.py names git {forbidden}"


def test_the_system_prompt_names_the_platform_and_the_shell() -> None:
    """A previous run burned six of thirty-five turns on Unix commands that do not exist here."""
    import platform

    prompt = harness.Harness(OllamaClient(), "fast", Files(Path.cwd()), "task").system_prompt()
    assert platform.system() in prompt
    assert "PLATFORM:" in prompt
    assert "NEVER regenerate a whole file" in prompt


def test_the_turn_budget_is_stricter_than_the_clients_on_both_measured_grounds() -> None:
    """24576 estimated tokens of code reads about 34600 real, which overflows num_ctx 32768."""
    client = OllamaClient(num_ctx=32768)
    budget = harness.turn_budget(client)
    assert budget < client.prompt_budget
    assert budget < client.num_ctx // 2
    assert budget * 4 / 2.84 < client.num_ctx


def test_a_stringified_tool_argument_object_still_parses() -> None:
    name, args = harness._arguments(
        {"function": {"name": "read_file", "arguments": '{"path": "a.py"}'}}
    )
    assert (name, args) == ("read_file", {"path": "a.py"})


def test_unparseable_tool_arguments_do_not_crash_the_loop() -> None:
    name, args = harness._arguments({"function": {"name": "read_file", "arguments": "{oops"}})
    assert (name, args) == ("read_file", {})


def _driver(
    repo: Path, seen: list[int], finish_on: int, ratio: float = 1.41
) -> Any:
    """Answers every chat with a tool call, and reports the count a code payload really reads."""
    state = {"turn": 0}

    def reply(request: Any) -> tuple[int, dict[str, str], str]:
        state["turn"] += 1
        payload = json.loads(request.body)
        estimate = estimate_tokens(
            "\n".join(str(m.get("content", "")) for m in payload["messages"])
        )
        seen.append(estimate)
        if state["turn"] >= finish_on:
            calls = [_call("finish", summary="done", files_changed=["big.py"])]
        else:
            calls = [_call("read_file", path="big.py")]
        body = {
            "message": {"content": "", "tool_calls": calls},
            "prompt_eval_count": int(estimate * ratio),
            "eval_count": 8,
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(body)

    return reply


@responses.activate
def test_thirty_five_turns_complete_without_tripping_the_truncation_guard(
    tmp_path: Path,
) -> None:
    """The default outcome before this task was an abort around turn 15 of 35."""
    repo = _repo(tmp_path / "r")
    (repo / "big.py").write_text("value = 1  # padding to make this read expensive\n" * 190,
                                 encoding="utf-8")
    seen: list[int] = []
    responses.add_callback(responses.POST, CHAT, callback=_driver(repo, seen, 35))
    client = OllamaClient(num_ctx=32768)
    run = harness.Harness(client, "fast", Files(repo), "implement the thing", max_turns=35)
    outcome = run.run()
    assert outcome.turns == 35
    assert outcome.finished is True
    assert len(seen) == 35, "a turn was refused before it was sent"
    assert not [note for note in outcome.notes if "stopped on turn" in note]
    assert max(seen) <= run.budget, f"a turn reached {max(seen)} against a budget of {run.budget}"
    assert outcome.compactions > 0, "compaction never fired, so this proves nothing"
    assert outcome.peak_prompt_tokens < client.num_ctx


@responses.activate
def test_without_compaction_the_same_run_would_hit_the_guard(tmp_path: Path) -> None:
    """The collision this task had to resolve, reproduced with compaction disabled."""
    repo = _repo(tmp_path / "r")
    (repo / "big.py").write_text("value = 1  # padding to make this read expensive\n" * 190,
                                 encoding="utf-8")
    seen: list[int] = []
    responses.add_callback(responses.POST, CHAT, callback=_driver(repo, seen, 35))
    run = harness.Harness(OllamaClient(num_ctx=32768), "fast", Files(repo), "task", max_turns=35)
    run._compact = lambda messages: 0  # type: ignore[method-assign]
    outcome = run.run()
    assert outcome.turns < 35
    assert any("prompt budget" in note for note in outcome.notes)
    assert len(seen) < 35, "the over-budget turn was refused before it could be sent"


@responses.activate
def test_a_refused_tool_call_becomes_a_message_the_model_can_act_on(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    sent: list[dict[str, Any]] = []

    def reply(request: Any) -> tuple[int, dict[str, str], str]:
        payload = json.loads(request.body)
        sent.append(payload)
        turn = len(sent)
        calls = (
            [_call("edit_file", path="sample.py", old_string="alpha = 1", new_string="x")]
            if turn == 1
            else [_call("finish", summary="stopped", files_changed=[])]
        )
        body = {
            "message": {"content": "", "tool_calls": calls},
            "prompt_eval_count": 200,
            "eval_count": 4,
        }
        return 200, {"Content-Type": "application/json"}, json.dumps(body)

    responses.add_callback(responses.POST, CHAT, callback=reply)
    harness.Harness(OllamaClient(), "fast", Files(repo), "task", max_turns=4).run()
    tools = [m for m in sent[-1]["messages"] if m.get("role") == "tool"]
    assert "REFUSED" in str(tools[0]["content"])
    assert "appears 2 times" in str(tools[0]["content"])


@responses.activate
def test_the_model_call_goes_through_the_client_so_num_ctx_is_always_sent(
    tmp_path: Path,
) -> None:
    """This is the rule the client exists to make unforgettable, at a third call site."""
    repo = _repo(tmp_path / "r")
    responses.add_callback(responses.POST, CHAT, callback=_driver(repo, [], 1))
    harness.Harness(OllamaClient(num_ctx=8192), "fast", Files(repo), "task").run()
    sent = json.loads(responses.calls[0].request.body or "{}")
    assert sent["options"]["num_ctx"] == 8192
    assert sent["think"] is False
    assert next(entry["function"]["name"] for entry in sent["tools"]) == "read_file"


@responses.activate
def test_finishing_without_ever_running_the_gate_is_recorded_as_resting_on_nothing(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "r")
    responses.add_callback(responses.POST, CHAT, callback=_driver(repo, [], 1))
    outcome = harness.Harness(OllamaClient(), "fast", Files(repo), "task").run()
    assert outcome.finished is True
    assert outcome.passed is False
    assert any("rests on nothing" in note for note in outcome.notes)


@responses.activate
def test_a_model_that_stops_calling_tools_is_nudged_once_and_then_the_run_ends(
    tmp_path: Path,
) -> None:
    repo = _repo(tmp_path / "r")
    body = {"message": {"content": "I think I am done."}, "prompt_eval_count": 30, "eval_count": 5}
    responses.add(responses.POST, CHAT, json=body)
    outcome = harness.Harness(OllamaClient(), "fast", Files(repo), "task", max_turns=9).run()
    assert outcome.turns == 2
    assert outcome.finished is False
    assert any("stopped making tool calls" in note for note in outcome.notes)


@responses.activate
def test_the_turn_limit_is_reported_rather_than_looping(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    responses.add_callback(responses.POST, CHAT, callback=_driver(repo, [], 99))
    outcome = harness.Harness(OllamaClient(), "fast", Files(repo), "task", max_turns=3).run()
    assert outcome.turns == 3
    assert any("turn limit" in note for note in outcome.notes)


@responses.activate
def test_ollama_going_away_mid_run_stops_cleanly_with_the_turn_named(tmp_path: Path) -> None:
    repo = _repo(tmp_path / "r")
    responses.add(responses.POST, CHAT, json={"error": "model runner has terminated"}, status=500)
    outcome = harness.Harness(OllamaClient(), "fast", Files(repo), "task").run()
    assert outcome.finished is False
    assert any("turn 1" in note for note in outcome.notes)


def test_a_file_the_gate_executes_cannot_be_written(tmp_path: Path) -> None:
    """conftest.py is imported by pytest before any test, so writing it is code execution."""
    repo = _repo(tmp_path)
    files = Files(repo)
    for name in ("conftest.py", "pyproject.toml", "noxfile.py", "sub/conftest.py"):
        with pytest.raises(ToolError, match="run by the toolchain"):
            files.create(name, "import os")


def test_the_git_directory_is_closed_to_the_editor(tmp_path: Path) -> None:
    """core.fsmonitor in .git/config runs on the next status, which this harness itself calls."""
    repo = _repo(tmp_path)
    with pytest.raises(ToolError, match="run by the toolchain"):
        Files(repo).edit(".git/config", "[core]", "[core]\n\tfsmonitor = \"cmd /c echo x\"")


def test_the_executed_guard_reads_dot_git_as_a_directory_not_a_prefix() -> None:
    """lstrip takes a character set, so it turned .git/config into git/config and let it through."""
    assert executes(".git/config")
    assert executes("./conftest.py")
    assert not executes("src/app.py")


def test_an_edit_payload_is_counted_even_though_it_rides_in_tool_calls() -> None:
    """Summing only content estimated a 400000-character edit at 0 tokens."""
    calls = [{"function": {"name": "edit_file", "arguments": {"new_string": "x" * 40000}}}]
    messages = [{"role": "assistant", "content": "", "tool_calls": calls}]
    assert estimate_tokens(conversation_text(messages)) > 9000


def test_the_clamp_signature_is_evidence_even_when_the_estimate_is_wrong() -> None:
    """Gating it behind sent_estimate discarded a correct detection of the measured overflow."""
    cut = Reply("x", "m", 32768, 16386, 10, [], sent_estimate=3)
    assert cut.suspect_truncation
    assert not Reply("x", "m", 32768, 20, 10, [], sent_estimate=15).suspect_truncation


def test_a_gate_that_passed_before_a_later_edit_does_not_count(tmp_path: Path) -> None:
    """Gate green, then break a file, then finish: this used to report every command exited 0."""
    green = GateResult(runs=(GateRun("ruff", "ruff check .", 0, "ok"),))
    outcome = Outcome(finished=True, gate=green, gate_tree="A", final_tree="B")
    assert green.passed
    assert outcome.stale_gate
    assert not outcome.passed


def test_a_created_file_appears_in_the_change_table(tmp_path: Path) -> None:
    """numstat is tracked-only, so a new file was absent AND called a fabricated report."""
    repo = _repo(tmp_path)
    (repo / "brand_new.py").write_text("x = 1\n", encoding="utf-8")
    assert any(c.path == "brand_new.py" for c in changes(repo))
