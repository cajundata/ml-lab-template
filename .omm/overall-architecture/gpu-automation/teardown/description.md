`src/ml_lab/gpu/teardown.py` — 56 lines, and the most important file in the repo. **Every teardown path routes through `destroy_and_verify()`.**

Its definition of success is the whole idea: *the droplet does not exist AND the account audit is clean.* Not "the DELETE returned 2xx" — an accepted DELETE is a promise, not a fact.

Three things it does that are easy to skip and expensive to omit:

1. **It ignores SIGINT while armed.** `signal.SIG_IGN` is installed before the destroy and restored in a `finally`. It prints "Teardown in progress — do not interrupt." A second Ctrl-C during teardown does nothing. Operator impatience cannot strand a billable droplet mid-destroy (proven live at S4).
2. **It polls to absence rather than trusting the API.** `_poll_until_absent` re-checks until `get_droplet` returns `None`, treating a transient `DOClientError` as "unknown, keep polling" — the poll, not any one call, is the truth. It ignores `destroy_droplet`'s return value entirely: a 404 or a transient error both just mean "poll anyway".
3. **It audits after.** Gone is not enough; the *account* has to be clean. A dirty audit raises `TeardownError` with the full report.
