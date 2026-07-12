"""The SSH seam: hardened, non-interactive ssh over subprocess, mocked in every test.

Mirrors do_client's doctl-subprocess pattern. wait_for_ssh / wait_for_bootstrap poll
on an injectable clock so deadline tests are instant. scp is deferred to S3c.
"""

from __future__ import annotations

import json
import subprocess
import time

from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    REMOTE_POLL_INTERVAL_SECONDS,
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
