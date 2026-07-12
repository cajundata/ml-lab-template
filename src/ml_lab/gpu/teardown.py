"""The one function every teardown path routes through.

Success = the droplet does not exist AND audit is clean. Once armed, SIGINT is
ignored so operator impatience cannot strand a billable droplet mid-teardown.
"""

from __future__ import annotations

import signal
import time

from ml_lab.gpu import audit as audit_mod
from ml_lab.gpu import do_client
from ml_lab.gpu.constants import (
    DESTROY_POLL_INTERVAL_SECONDS,
    DESTROY_POLL_TIMEOUT_SECONDS,
)


class TeardownError(RuntimeError):
    """Teardown could not confirm the droplet is gone and the account is clean."""


def _poll_until_absent(droplet_id: int) -> None:
    deadline = time.monotonic() + DESTROY_POLL_TIMEOUT_SECONDS
    while True:
        try:
            droplet = do_client.get_droplet(droplet_id)
        except do_client.DOClientError:
            droplet = "unknown"  # transient; the poll, not the call, is the truth
        if droplet is None:
            return
        if time.monotonic() >= deadline:
            raise TeardownError(
                f"droplet {droplet_id} still present after {DESTROY_POLL_TIMEOUT_SECONDS}s"
            )
        time.sleep(DESTROY_POLL_INTERVAL_SECONDS)


def destroy_and_verify(droplet_id: int) -> None:
    previous = signal.getsignal(signal.SIGINT)
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    print(
        f"Teardown in progress — droplet {droplet_id} is being destroyed. Do not interrupt.",
        flush=True,
    )
    try:
        do_client.destroy_droplet(droplet_id)  # 404 or transient error → poll anyway
        _poll_until_absent(droplet_id)
        report = audit_mod.collect_audit()
        if not audit_mod.is_clean(report):
            raise TeardownError(
                "audit dirty after destroy:\n" + audit_mod.format_report(report)
            )
    finally:
        signal.signal(signal.SIGINT, previous)
