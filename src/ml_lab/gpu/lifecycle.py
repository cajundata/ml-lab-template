"""The gpu-up orchestration: create, verify readiness, hand off — or tear down.

Invariant: once a droplet exists, every non-clean exit routes through
destroy_and_verify; a fully verified success leaves it alive (the debug path).
All seams (do_client, remote, create, teardown) are mocked in tests.
"""

from __future__ import annotations

import time

from ml_lab.gpu import do_client
from ml_lab.gpu.cloud_init import render_cloud_init
from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    DEFAULT_TTL_SECONDS,
    DO_REGION,
    DO_SIZE_SLUG,
    REMOTE_POLL_INTERVAL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SSH_TIMEOUT_SECONDS,
)
from ml_lab.gpu.create import create_lab_droplet, generate_run_id
from ml_lab.gpu.gpu_env import load_gpu_env
from ml_lab.gpu.remote import RemoteError, wait_for_bootstrap, wait_for_ssh
from ml_lab.gpu.teardown import destroy_and_verify


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


def _print_handoff(result, ip, ttl_seconds):
    print(
        "Created GPU droplet:\n"
        f"  id: {result['id']}\n"
        f"  name: {result['name']}\n"
        f"  region: {DO_REGION}\n"
        f"  size: {DO_SIZE_SLUG}\n"
        f"  public-ip: {ip}\n"
        f"  ttl: {ttl_seconds} seconds\n"
        f"  self-destruct retry: {SELF_DESTRUCT_RETRY_SECONDS} seconds\n"
        f"  destroy: make gpu-down DROPLET_ID={result['id']}\n"
        "Self-destruct timer has been verified active on the droplet.\n"
        "Do not power off this droplet. Destroy it.",
        flush=True,
    )


def gpu_up(*, ttl_seconds=DEFAULT_TTL_SECONDS, enforce_budget=True, env=None, now=None) -> dict:
    """Create a GPU droplet, verify SSH + bootstrap, hand off — or tear down on any failure.

    On a fully verified success the droplet is LEFT ALIVE (debug path). On any
    post-create exception (including KeyboardInterrupt) it is destroyed and the
    error re-raised. A preflight failure creates no droplet, so nothing is destroyed.
    """
    if env is None:
        env = load_gpu_env()
    run_now = now if now is not None else time.time()
    run_id = generate_run_id(run_now)
    user_data = render_cloud_init(
        run_id=run_id, destroy_token=env.destroy_token, ttl_seconds=ttl_seconds
    )

    droplet_id = None
    try:
        result = create_lab_droplet(
            user_data,
            ttl_seconds=ttl_seconds,
            enforce_budget=enforce_budget,
            ssh_key_ids=env.ssh_key_ids,
            run_id=run_id,
            now=run_now,
        )
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        _print_handoff(result, ip, ttl_seconds)
        return {**result, "ip": ip}
    except BaseException:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)
        raise
