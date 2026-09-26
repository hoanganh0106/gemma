#!/usr/bin/env python3
"""Sequential MOA submissions from frozen source. Run in WSL/Linux.

Uses the authenticated moa-submitter CLI; standard Python library only.
See MOA_SUBMIT_LOOP.md for usage and recovery.
"""
import argparse
import contextlib
import hashlib
import json
import math
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

TERMINAL = {"completed", "failed", "rejected", "cancelled", "canceled"}
ACTIVE = {"pending", "queued", "received", "validating", "building", "evaluating", "running", "submitted"}
ID = r"[0-9a-fA-F]{8}(?:-[0-9a-fA-F-]{27,})?"


def say(message):
    print(time.strftime("%Y-%m-%d %H:%M:%S"), message, flush=True)


def save(path, value):
    temp = path.with_suffix(".tmp")
    with temp.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, ensure_ascii=False)
        stream.flush()
        os.fsync(stream.fileno())
    temp.replace(path)


def source_hashes(root):
    paths = [root / "Cargo.toml", root / "src/ops.rs"]
    if not (root / "src/device").is_dir():
        raise ValueError(f"Missing src/device: {root}")
    paths += sorted((root / "src/device").rglob("*"))
    paths += [root / name for name in ("Cargo.lock", "rust-toolchain.toml") if (root / name).exists()]
    result = {}
    for path in paths:
        if path.is_symlink():
            raise ValueError(f"Source symlink is unsupported: {path}")
        if path.is_dir():
            continue
        result[path.relative_to(root).as_posix()] = hashlib.sha256(path.read_bytes()).hexdigest()
    return result


def field(output, label):
    match = re.search(rf"^{re.escape(label)}:\s*(.*?)\s*$", output, re.M)
    return match.group(1) if match else None


def detail(output, expected_id):
    actual = field(output, "Submission")
    if not actual or not (actual.startswith(expected_id) or expected_id.startswith(actual)):
        raise ValueError("Status response has missing/different submission ID")
    status = (field(output, "Status") or "").lower()
    if status not in TERMINAL | ACTIVE:
        raise ValueError(f"Unknown submission status: {status!r}")
    score = None
    if status == "completed":
        try:
            score = float(field(output, "Score"))
        except (TypeError, ValueError) as exc:
            raise ValueError("Completed submission has no numeric score; not submitting again") from exc
        if not math.isfinite(score) or score < 0:
            raise ValueError("Invalid score")
    return status, score


def listing(output):
    rows = {}
    for line in output.splitlines():
        match = re.match(rf"^\s*({ID})\s+(.+?)\s+(\w+)\s+(-|\d+(?:\.\d+)?)\s+\d{{4}}-", line)
        if match:
            ident, _, status, _ = match.groups()
            if status.lower() not in TERMINAL | ACTIVE:
                raise ValueError(f"Unknown account submission status: {status}")
            rows[ident] = status.lower()
        elif re.match(rf"^\s*{ID}\s", line):
            raise ValueError(f"Cannot parse account submission row: {line}")
    if not rows and not re.search(r"no submissions", output, re.I):
        raise ValueError("Cannot parse account submission list; refusing to submit")
    return rows


@contextlib.contextmanager
def account_lock():
    # All invocations of this script for this Linux user share one lock.
    import fcntl
    lock_dir = Path.home() / ".cache/moa-submit-loop"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / "account.lock").open("a") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("Another MOA loop is running for this Linux user") from exc
        yield


def run_cli(cli, args, timeout):
    result = subprocess.run([cli, *args], capture_output=True, text=True, timeout=timeout)
    return result.returncode, result.stdout + result.stderr


def read_cli(cli, args, poll_seconds):
    while True:
        try:
            code, output = run_cli(cli, args, 90)
            if code == 0:
                return output
            say(f"Read failed ({code}); retry in {poll_seconds}s: {output.strip()}")
        except (subprocess.TimeoutExpired, OSError) as exc:
            say(f"Read unavailable; retry in {poll_seconds}s: {exc}")
        time.sleep(poll_seconds)


def load_or_create(args, directory):
    state_path = directory / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text(encoding="utf-8"))
        if state["source"] != str(args.source):
            raise ValueError("State belongs to another source. Select a different --state-dir")
        if source_hashes(directory / "source") != state["hashes"]:
            raise ValueError("Frozen source was modified; refusing to continue")
        return state
    if any(directory.iterdir()):
        raise ValueError("State directory is not empty but has no state.json; choose a new directory")
    hashes = source_hashes(args.source)
    snapshot = directory / "source"
    for name in hashes:
        destination = snapshot / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(args.source / name, destination)
    if source_hashes(snapshot) != hashes or source_hashes(args.source) != hashes:
        raise ValueError("Source changed during snapshot creation; choose a new state directory and retry")
    state = {"source": str(args.source), "hashes": hashes, "phase": "idle", "active_id": None,
             "attempts": 0, "history": [], "best_score": None, "consecutive_failures": 0}
    save(state_path, state)
    return state


def loop(args, cli, directory):
    state = load_or_create(args, directory)
    path = directory / "state.json"
    logs = directory / "logs"
    logs.mkdir(exist_ok=True)
    if state["phase"] == "submitting":
        if not args.recover_id:
            raise ValueError("Previous submit outcome is uncertain. Inspect MOA status/log; resume with "
                             "--recover-id ID only after identifying the accepted submission. No automatic resend.")
        rows = listing(read_cli(cli, ["status", "--all"], args.poll_seconds))
        matches = [ident for ident in rows if ident.startswith(args.recover_id) or args.recover_id.startswith(ident)]
        if len(matches) != 1 or matches[0] in state["before_ids"]:
            raise ValueError("Recovery ID must be a new submission in this account after the uncertain attempt")
        state.update(phase="waiting", active_id=matches[0])
        save(path, state)
    elif args.recover_id:
        raise ValueError("--recover-id is only valid after an uncertain submit")
    say(f"Source: {args.source}; frozen source: {directory / 'source'}; target: {args.target_score}")
    while True:
        if state["active_id"]:
            ident = state["active_id"]
            output = read_cli(cli, ["status", ident], args.poll_seconds)
            (logs / f"{ident}.status.txt").write_text(output, encoding="utf-8")
            status, score = detail(output, ident)
            log = read_cli(cli, ["log", ident], args.poll_seconds)
            (logs / f"{ident}.log").write_text(log, encoding="utf-8")
            say(f"{ident}: {status}" + (f"; score={score}" if score is not None else ""))
            if status not in TERMINAL:
                time.sleep(args.poll_seconds)
                continue
            state["history"].append({"id": ident, "status": status, "score": score})
            state.update(active_id=None, phase="idle", next_submit_at=time.time() + args.cooldown_seconds)
            if status == "completed":
                state["best_score"] = max(score, state["best_score"] or 0)
                state["consecutive_failures"] = 0
            else:
                state["consecutive_failures"] += 1
            save(path, state)
        if state["best_score"] is not None and state["best_score"] >= args.target_score:
            say(f"Target reached: best score={state['best_score']}; stopped")
            return 0
        if state["consecutive_failures"] and (not args.continue_on_failure or
                state["consecutive_failures"] >= args.max_consecutive_failures):
            say("Stopped after failed/rejected/cancelled submission; inspect saved log")
            return 2
        if args.max_submissions and state["attempts"] >= args.max_submissions:
            say(f"Reached limit of {args.max_submissions} attempts; best score={state['best_score']}")
            return 0
        time.sleep(max(0, state.get("next_submit_at", 0) - time.time()))
        rows = listing(read_cli(cli, ["status", "--all"], args.poll_seconds))
        pending = [ident for ident, status in rows.items() if status not in TERMINAL]
        if pending:
            say(f"Account has active submissions {pending}; waiting {args.poll_seconds}s")
            time.sleep(args.poll_seconds)
            continue
        if source_hashes(directory / "source") != state["hashes"]:
            raise ValueError("Frozen source was modified; refusing to submit")
        # Persist intent BEFORE the external side effect. Never blindly retry submit.
        state.update(phase="submitting", before_ids=list(rows), attempts=state["attempts"] + 1)
        save(path, state)
        try:
            code, output = run_cli(cli, ["submit", "--source", str(directory / "source")], 180)
        except (subprocess.TimeoutExpired, OSError) as exc:
            raise ValueError("Submit outcome uncertain; inspect account and use --recover-id to resume") from exc
        (logs / f"submit-{state['attempts']:04d}.txt").write_text(output, encoding="utf-8")
        ident = field(output, "Submission")
        if not ident or not re.fullmatch(ID, ident):
            raise ValueError(f"Submit returned {code} without a usable ID. Outcome uncertain; inspect saved log")
        state.update(phase="waiting", active_id=ident)
        save(path, state)
        say(f"Accepted submission {ident} (attempt {state['attempts']})")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True, help="Version/repository directory to freeze and submit")
    parser.add_argument("--target-score", type=float, required=True, help="Stop when a completed submission in this run reaches this score")
    parser.add_argument("--state-dir", type=Path, help="Persistent run directory; same directory resumes the frozen version")
    parser.add_argument("--poll-seconds", type=int, default=300)
    parser.add_argument("--cooldown-seconds", type=int, default=30)
    parser.add_argument("--max-submissions", type=int, default=0, help="Total attempts in this state directory; 0 = unlimited")
    parser.add_argument("--continue-on-failure", action="store_true")
    parser.add_argument("--max-consecutive-failures", type=int, default=3)
    parser.add_argument("--recover-id", help="Manually identify an accepted submission after an uncertain submit")
    parser.add_argument("--cli", default="moa-submitter")
    parser.add_argument("--dry-run", action="store_true", help="Inspect source and config only; no network calls or submissions")
    args = parser.parse_args(argv)
    if not math.isfinite(args.target_score) or args.target_score <= 0:
        parser.error("--target-score must be finite and positive")
    if args.poll_seconds < 1 or args.cooldown_seconds < 0 or args.max_submissions < 0 or args.max_consecutive_failures < 1:
        parser.error("Invalid interval or attempt limit")
    args.source = args.source.expanduser().resolve()
    suffix = hashlib.sha256(str(args.source).encode()).hexdigest()[:10]
    directory = (args.state_dir or Path(__file__).resolve().parents[1] / "submission_staging" /
                 f"moa-loop-{args.source.name}-{suffix}").expanduser().resolve()
    if args.dry_run:
        if (directory / "state.json").exists():
            state = json.loads((directory / "state.json").read_text(encoding="utf-8"))
            if state["source"] != str(args.source) or source_hashes(directory / "source") != state["hashes"]:
                raise ValueError("Existing state/source mismatch")
            say(f"Resume frozen version: phase={state['phase']}, active={state['active_id']}, best={state['best_score']}")
        else:
            say(f"Source valid: {len(source_hashes(args.source))} files will be frozen")
        say(f"State: {directory}; target={args.target_score}; poll={args.poll_seconds}s; max={args.max_submissions or 'unlimited'}")
        return 0
    cli = shutil.which(args.cli)
    if not cli:
        raise ValueError("moa-submitter not found; run in WSL with ~/.cargo/bin on PATH")
    with account_lock():
        directory.mkdir(parents=True, exist_ok=True)
        return loop(args, cli, directory)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        say("Stopped locally. Remote submission continues; rerun the same command to resume.")
        sys.exit(130)
    except (ValueError, OSError, KeyError) as error:
        say(f"ERROR: {error}")
        sys.exit(2)
