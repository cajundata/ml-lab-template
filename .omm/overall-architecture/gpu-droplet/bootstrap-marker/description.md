`/opt/ml-lab/bootstrap-ready.json` — one small file, and the entire contract between the local orchestrator and the remote machine:

```json
{"ready": true, "self_destruct_timer_active": true}
```

Written by cloud-init as the **last** step of `runcmd`, only after the self-destruct timer has been asserted active for the second time. Polled by `remote.wait_for_bootstrap`, which **parses it as JSON** — deliberately not a string match — and requires **both** flags to be exactly `True`.

The consequence is the design's cleanest guarantee: **a droplet that could not arm its own self-destruct never reports ready, and a droplet that never reports ready is torn down instead of used.** There is no path where the local side proceeds to run a benchmark on a machine that has no safety net. The failure mode degrades toward "destroy it", which is the correct direction to fall.
