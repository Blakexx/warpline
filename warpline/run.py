"""THE RUNNER, `warpline run`: the user's own command run once, instrumented, and what it produced collected into one
directory beside the run manifest (docs/run-layer.md). The readers run nothing; this is the one place a run is started.

    warpline run [--stamps] [--profile SPEC]... [--place NAME] -o DIR -- <command> [<argument>...]

The runner never switches a stamp and never builds the task: the command is the user's, built stamped or plain as the user
chose. What each instrument does is its module's, given through its `cli` as `RUN_TAKES` (a flag to a `Takes`): the recorder
takes `--stamps` (it tells the task where to write its records and collects them), the checker takes `--profile` (it runs the
command under the profiler and exports what the profiler captured). The runner composes them: the task's environment gains each
instrument's variables, the command runs under the instrument's prefix, and after it each instrument collects its files; those
files are the inputs of `run-manifest` (`Takes.gives`), and the manifest is written from them by the same code. This file imports
neither module (docs/modules.md).

THE DIRECTORY `DIR` is the run's: it is made, and refused when it exists and holds anything, so nothing of another run is
collected. A command that fails (a nonzero exit) is not a run: nothing is collected and the runner exits 2.
"""
from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable


class RunError(ValueError):
    """A run that cannot be made or collected, in words: the directory, the command, an instrument's files."""


@dataclass(frozen=True)
class Started:
    """What an instrument adds to the task's start: a command the task runs under (the profiler), and variables of its
    environment (where the records go)."""
    prefix: tuple[str, ...] = ()
    env: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Collected:
    """What an instrument found after the task: its files (the `run-manifest` input `Takes.gives`), what else it wrote beside
    them, and the lines the run manifest says of them (the profiler's version)."""
    paths: tuple[str, ...]
    beside: tuple[str, ...] = ()
    said: tuple[str, ...] = ()


@dataclass(frozen=True)
class Takes:
    """What a module does with one flag of `warpline run`: `start(out, values)` before the task (the run's directory and the
    flag's values), `collect(out, values)` after it. `gives` is the `run-manifest` flag the collected files are read as."""
    gives: str
    start: Callable[[Path, list[str]], Started]
    collect: Callable[[Path, list[str]], Collected]


def directory(out: str) -> Path:
    """The run's directory, made: absolute (the task is told it), refused when it exists and is not empty."""
    path = Path(out).resolve()
    if path.exists() and (not path.is_dir() or any(path.iterdir())):
        raise RunError(f"{out} exists and is not an empty directory: a run's directory holds that run alone")
    path.mkdir(parents=True, exist_ok=True)
    return path


def run(out: str, instruments: list[tuple[str, Takes, list[str]]], command: list[str]) -> tuple[Path, dict[str, Collected]]:
    """Run `command` once, its run's directory `out` made (`directory`), under the instruments (flag, what its module takes,
    its values): each starts, the command runs with their environment and under their prefixes (in the order given), and
    each collects. Returns the directory and what each flag collected."""
    if not command:
        raise RunError("no command: give the task after `--` (`warpline run ... -o DIR -- ./task args`)")
    out = directory(out)
    started = [(flag, takes, values, takes.start(out, values)) for flag, takes, values in instruments]
    env = dict(os.environ)
    prefix: list[str] = []
    for *_, s in started:
        env.update(s.env)
        prefix += s.prefix
    done = subprocess.run([*prefix, *command], env=env)
    if done.returncode:
        raise RunError(f"the command exited {done.returncode}: a failed task is not a run, and nothing is collected")
    return out, {flag: takes.collect(out, values) for flag, takes, values, _ in started}
