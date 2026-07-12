"""The gpu-up orchestration: create, verify readiness, hand off — or tear down.

Invariant: once a droplet exists, every non-clean exit routes through
destroy_and_verify; a fully verified success leaves it alive (the debug path).
All seams (do_client, remote, create, teardown) are mocked in tests.
"""

from __future__ import annotations

import time

from ml_lab.gpu import do_client
from ml_lab.gpu.constants import REMOTE_POLL_INTERVAL_SECONDS, SSH_TIMEOUT_SECONDS
from ml_lab.gpu.remote import RemoteError


def wait_for_public_ip(
    droplet_id,
    *,
    timeout=SSH_TIMEOUT_SECONDS,
    interval=REMOTE_POLL_INTERVAL_SECONDS,
    now=None,
    sleep=None,
):
    """Poll get_droplet until a public IPv4 appears; raise RemoteError at the deadline."""
    now = now or time.monotonic
    sleep = sleep or time.sleep
    deadline = now() + timeout
    while True:
        try:
            droplet = do_client.get_droplet(droplet_id)
        except do_client.DOClientError:
            droplet = None  # transient; the poll, not one call, is the truth
        ip = do_client.public_ipv4(droplet) if droplet else None
        if ip:
            return ip
        if now() >= deadline:
            raise RemoteError(f"no public IP for droplet {droplet_id} after {timeout}s")
        sleep(interval)
