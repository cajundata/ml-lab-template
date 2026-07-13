**Why the seam pattern earns its keep here.** These are not integrations you can safely "just try". A bad doctl call bills by the hour. A live test suite would provision H100s. So the seams are not an abstraction for its own sake — they are what make 193 tests runnable on a laptop with no GPU, no DigitalOcean account, and no network, while the code they cover is spending real money in production.

The shape is consistent enough to copy from:
- **One module per external dependency.** `do_client.py`, `remote.py`, `spaces.py`.
- **Intent-level functions**, not a thin transport wrapper — `probe_destroy_token()`, not `http_delete()`.
- **Errors normalized into one exception type** per seam (`DOClientError`, `RemoteError`, `SpacesError`), so callers never handle a `CalledProcessError` or a boto3 `ClientError`.
- **Injectable everything** — the boto3 client, the clock, the sleep. Deadline tests run instantly.
- **Ambiguity resolves toward safety.** A transient `DOClientError` during absence-polling means "keep polling", never "probably fine".

**The one fact to carry away if you carry only one:** `doctl -o json` puts errors on **stdout**, including 404s. Read both streams. Always.
