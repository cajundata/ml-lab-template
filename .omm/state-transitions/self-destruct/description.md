The remote reaper — **the only safety layer that survives the local machine disappearing entirely.**

`/opt/ml-lab/self_destruct.sh`, driven by `ml-lab-self-destruct.timer`, both planted by cloud-init at boot. The script is six lines:

```bash
source /etc/ml-lab/run.env                                  # the destroy-scoped token, mode 0600
DROPLET_ID="$(curl -fsS http://169.254.169.254/metadata/v1/id)"   # ask DO who I am
curl -fsS -X DELETE -H "Authorization: Bearer ${DO_DROPLET_DESTROY_TOKEN}" \
  "https://api.digitalocean.com/v2/droplets/${DROPLET_ID}"        # delete myself
```

The droplet learns its own id from the DO metadata service and deletes itself. **It needs nothing from the machine that created it** — no SSH, no network path home, no local process still alive.

The timer is `OnBootSec=<TTL>` (7200s) with **`OnUnitActiveSec=300s`**. That retry is not decoration: a one-shot timer would be a safety net with a hole in it, silently defeated by a single transient API failure. This one **retries every 5 minutes, forever**, until the DELETE lands.

**Verified live at S4**, and worth stating plainly because it is the claim everything else rests on: a `kill -9`'d run — with zero local involvement, no `finally`, no handler — was reaped by this timer.

**Exit:** `destroyed`.
