"""The SSH seam: hardened, non-interactive ssh over subprocess, mocked in every test.

Mirrors do_client's doctl-subprocess pattern. wait_for_ssh / wait_for_bootstrap poll
on an injectable clock so deadline tests are instant. scp_up / scp_down are one-shot:
a nonzero exit or timeout raises RemoteError.
"""

from __future__ import annotations

import json
import subprocess
import time

from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    REMOTE_POLL_INTERVAL_SECONDS,
    SCP_TIMEOUT_SECONDS,
    SSH_ATTEMPT_TIMEOUT_SECONDS,
    SSH_TIMEOUT_SECONDS,
)

# Hardened, non-interactive: never prompt, use only the -i key (not agent keys),
# bound each connect, don't pollute known_hosts.
SSH_OPTS = [
    "-o", "BatchMode=yes",
    "-o", "IdentitiesOnly=yes",
    "-o", "StrictHostKeyChecking=accept-new",
    "-o", "ConnectTimeout=15",
    "-o", "UserKnownHostsFile=/dev/null",
    "-o", "LogLevel=ERROR",
]


class RemoteError(RuntimeError):
    """SSH was not reachable, or bootstrap was not verified, before the deadline."""


def _run_ssh(host, argv, *, key_path, timeout):
    """Run `ssh -i <key> <opts> root@<host> <argv...>`; return the CompletedProcess."""
    return subprocess.run(
        ["ssh", "-i", key_path, *SSH_OPTS, f"root@{host}", *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def wait_for_ssh(
    host,
    *,
    key_path,
    timeout=SSH_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll `ssh ... true` until it exits 0; raise RemoteError at the deadline."""
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        try:
            ok = _run_ssh(host, ["true"], key_path=key_path,
                          timeout=SSH_ATTEMPT_TIMEOUT_SECONDS).returncode == 0
        except subprocess.TimeoutExpired:
            ok = False
        if ok:
            return
        if now() >= deadline:
            raise RemoteError(f"ssh to {host} not reachable after {timeout}s")
        sleep(interval)


def wait_for_bootstrap(
    host,
    *,
    key_path,
    timeout=BOOTSTRAP_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll /opt/ml-lab/bootstrap-ready.json until ready AND timer active; else RemoteError.

    Parses the marker as JSON (not string-match), so both flags must be exactly True.
    """
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        marker = None
        try:
            result = _run_ssh(
                host,
                ["cat", "/opt/ml-lab/bootstrap-ready.json"],
                key_path=key_path,
                timeout=SSH_ATTEMPT_TIMEOUT_SECONDS,
            )
            if result.returncode == 0:
                try:
                    parsed = json.loads(result.stdout)
                except json.JSONDecodeError:
                    parsed = None
                if (
                    isinstance(parsed, dict)
                    and parsed.get("ready") is True
                    and parsed.get("self_destruct_timer_active") is True
                ):
                    marker = parsed
        except subprocess.TimeoutExpired:
            marker = None
        if marker is not None:
            return marker
        if now() >= deadline:
            raise RemoteError(f"bootstrap not verified on {host} after {timeout}s")
        sleep(interval)


def _run_scp(argv, *, key_path, timeout):
    """Run `scp -i <key> <SSH_OPTS...> <argv...>`; return the CompletedProcess."""
    return subprocess.run(
        ["scp", "-i", key_path, *SSH_OPTS, *argv],
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _checked_scp(argv, *, key_path, timeout, what):
    """Run a one-shot scp; raise RemoteError on timeout or nonzero exit."""
    try:
        result = _run_scp(argv, key_path=key_path, timeout=timeout)
    except subprocess.TimeoutExpired as e:
        raise RemoteError(f"scp {what} timed out after {timeout}s") from e
    if result.returncode != 0:
        raise RemoteError(f"scp {what} failed ({result.returncode}): {result.stderr.strip()}")


def scp_up(host, local_path, remote_path, *, key_path, timeout=SCP_TIMEOUT_SECONDS):
    """Copy a local file up to root@<host>:<remote_path>; RemoteError on failure."""
    _checked_scp(
        [local_path, f"root@{host}:{remote_path}"],
        key_path=key_path, timeout=timeout, what=f"up {local_path}",
    )


def scp_down(host, remote_path, local_path, *, key_path, timeout=SCP_TIMEOUT_SECONDS, recursive=False):
    """Pull root@<host>:<remote_path> down to local_path; RemoteError on failure."""
    argv = (["-r"] if recursive else []) + [f"root@{host}:{remote_path}", local_path]
    _checked_scp(argv, key_path=key_path, timeout=timeout, what=f"down {remote_path}")
