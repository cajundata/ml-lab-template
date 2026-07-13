"""GPU lifecycle orchestration: gpu_up (debug) and gpu_run (full benchmark).

Both share the create → wait-for-IP → wait-for-SSH → wait-for-bootstrap spine.
They differ at the end: gpu_up LEAVES a verified droplet alive (debug path) and
destroys only on a non-clean exit; gpu_run runs the benchmark, pulls + uploads the
bundle, and ALWAYS destroys in a finally. Once a droplet exists, every teardown
routes through destroy_and_verify. All seams (do_client, remote, create, benchmark,
spaces, teardown) are mocked in tests.
"""

from __future__ import annotations

import time

from ml_lab.gpu import do_client
from ml_lab.gpu.cloud_init import render_cloud_init
from ml_lab.gpu.constants import (
    BOOTSTRAP_TIMEOUT_SECONDS,
    DEFAULT_TTL_SECONDS,
    REMOTE_POLL_INTERVAL_SECONDS,
    SELF_DESTRUCT_RETRY_SECONDS,
    SSH_TIMEOUT_SECONDS,
)
from ml_lab.gpu.benchmark import deliver_and_run_benchmark, pull_artifacts
from ml_lab.gpu.create import create_lab_droplet, generate_run_id
from ml_lab.gpu.gpu_env import load_gpu_env, load_spaces_env
from ml_lab.gpu.remote import RemoteError, wait_for_bootstrap, wait_for_ssh
from ml_lab.gpu.spaces import upload_bundle
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
        f"  region: {result['region']}\n"
        f"  size: {result['size']}\n"
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
    error re-raised — unless destroy_and_verify itself fails, in which case that
    TeardownError propagates instead (the louder alarm; the original is chained as
    its __context__). A preflight failure creates no droplet, so nothing is destroyed.
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


def gpu_run(*, ttl_seconds=DEFAULT_TTL_SECONDS, env=None, spaces=None, now=None) -> int:
    """Full lifecycle: create, benchmark, pull, upload to Spaces, always destroy.

    Returns the benchmark's exit code. A completed-but-failed benchmark still pulls
    and uploads the partial bundle before the finally destroys. Any raised failure
    (SSH/bootstrap/benchmark timeout, transport, or upload) destroys then propagates —
    destroy beats artifact preservation. Preflight (env + Spaces credentials) runs
    before any droplet is created, so a misconfiguration strands nothing. If
    destroy_and_verify itself fails, that TeardownError is the louder alarm and
    overrides any in-flight error (chained as __context__).
    """
    if env is None:
        env = load_gpu_env()
    if spaces is None:
        spaces = load_spaces_env()
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
            enforce_budget=True,
            ssh_key_ids=env.ssh_key_ids,
            run_id=run_id,
            now=run_now,
        )
        droplet_id = result["id"]
        ip = wait_for_public_ip(droplet_id, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_ssh(ip, key_path=env.ssh_key_path, timeout=SSH_TIMEOUT_SECONDS)
        wait_for_bootstrap(ip, key_path=env.ssh_key_path, timeout=BOOTSTRAP_TIMEOUT_SECONDS)
        code = deliver_and_run_benchmark(ip, run_id, key_path=env.ssh_key_path)
        dest = pull_artifacts(ip, run_id, key_path=env.ssh_key_path)
        uri = upload_bundle(dest, run_id, env=spaces)
        print(f"Uploaded benchmark bundle to {uri}", flush=True)
        return code
    finally:
        if droplet_id is not None:
            destroy_and_verify(droplet_id)
