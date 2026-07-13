`make gpu-audit` — **free, safe, and the most important command in the repo.** It answers the only question that ever really matters: *is anything billing right now?*

Collects lab droplets **plus** related billable resources (volumes, snapshots, reserved IPs, load balancers). Droplets match on the `ml-lab` tag **or** the `ml-lab-gpu-` name prefix, so a droplet whose tagging somehow failed is still caught by its name.

Clean output says so plainly. Dirty output prints, per droplet: id, name, status, region, size, image, public IP, age, TTL expiry, whether it is **OVERDUE**, and the exact `make gpu-down DROPLET_ID=<id>` command to reap it.

**It exits nonzero when the account is dirty.** That exit code is the contract — it is designed to be usable as a preflight or CI check, not merely read by a human.

It is also the final assertion inside `destroy_and_verify`: teardown is not considered complete until this reports clean. So it is both the human's tool and the machine's definition of done.
