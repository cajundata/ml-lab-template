**`doctl -o json` writes errors — including 404 — to STDOUT, not stderr.**

Every error path in this module reads `stderr + stdout` concatenated. This is not defensive padding; it is load-bearing:
- It caused a **false `TeardownError`** at S4 live (the clean 404 that means "the droplet is gone" was invisible on stderr) — which would have broken every single run.
- It separately **masked an invalid-SSH-key error** into a blank message, which cost real debugging time.

**Any new doctl call must follow the same pattern.** If you add one and only read `stderr`, it will appear to work until the day the API returns an error, and then it will lie to you.
