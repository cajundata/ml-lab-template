**`doctl -o json` writes error detail — INCLUDING 404 — to STDOUT, not stderr.**

Every error path in `do_client.py` therefore concatenates `stderr + stdout` before inspecting it. This is not defensive padding; it is load-bearing. At S4 live it:
- produced a **false `TeardownError`** (the clean 404 that means "the droplet is gone" was invisible on stderr) — which would have broken **every single run**; and
- **masked an invalid-SSH-key error** into a blank message, costing real debugging time.

**Any new doctl call must follow the same pattern.** If you add one that reads only `stderr`, it will appear to work perfectly until the day the API returns an error — and then it will lie to you.

**Second, softer concern:** doctl is a subprocess, so its error surface is stringly-typed. `create.py` detects a capacity failure by matching the substring `"not available in this region"`. That works today and is checked against real DO output, but a doctl release could reword it and silently turn a graceful multi-SKU retry into a hard failure. If creates start failing instead of retrying, look here first.
