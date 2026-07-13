`src/ml_lab/gpu/do_client.py` — the DigitalOcean seam. Intent-level functions only; mocked in every test. Two transports on purpose: **doctl** (subprocess) for list/get/create/destroy, and the **DO REST API** (via `requests`) for the destroy-token probe.

**The landmine this module exists to contain:** `doctl -o json` writes error detail — **including 404** — to **STDOUT, not stderr**. Every error path here therefore concatenates `stderr + stdout` before inspecting it. Getting this wrong produced a false `TeardownError` at S4 live that would have broken every run, and separately masked an invalid-SSH-key error into a blank message. **Any new doctl call must follow this pattern.**

The three interesting functions:
- **`get_droplet()`** returns `None` on a clean 404 (the droplet is genuinely gone) and raises `DOClientError` on anything else. That distinction is what `_poll_until_absent` is built on — "gone" and "we could not tell" must never be confused.
- **`destroy_droplet()`** returns `'accepted'` / `'gone'` / `'error'` rather than raising. `destroy_and_verify` deliberately ignores it and polls to absence anyway.
- **`probe_destroy_token()`** sends `DELETE /v2/droplets/1` against a known-nonexistent droplet. **404 = good** (authenticates, carries delete scope); 401/403 = the token is rejected; anything else is unexpected and raises. This is how we know, before creating anything, that we can destroy it.

`create_droplet()` writes the cloud-init to a temp file, applies tags **atomically** via `--tag-names`, and deliberately does **not** pass `--wait` — the id must return immediately so it can be printed before anything else can go wrong.
