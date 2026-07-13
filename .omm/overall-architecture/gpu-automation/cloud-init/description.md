`src/ml_lab/gpu/cloud_init.py` — renders `scripts/cloud-init-gpu.yaml.tmpl` into the `user_data` string that `create_lab_droplet` requires. Pure string templating: no network, no droplet, trivially testable.

It substitutes on **collision-proof `@@PLACEHOLDER@@` markers** rather than Python's `.format()` or f-strings. That is not stylistic — the template is a bash-bearing YAML file full of literal `${...}` shell expansions, and any templating engine that treats `{}` or `$` as special would mangle them. `@@RUN_ID@@`-style markers cannot collide with bash.

Two guards:
- **`destroy_token` is required** and raises `ValueError` if empty, with the reason spelled out: self-destruct cannot arm without it. A droplet that boots with a blank token has no safety net.
- **After rendering, it scans for any leftover `@@...@@`** and raises if it finds one. A silently unfilled placeholder would ship a broken self-destruct script to a machine that bills by the hour.

The destroy token lands only in `/etc/ml-lab/run.env`, at mode 0600.
