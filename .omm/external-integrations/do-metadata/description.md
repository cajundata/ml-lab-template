`http://169.254.169.254/metadata/v1/id` — the DigitalOcean link-local metadata service, called **only from inside the droplet**, and for exactly one purpose: **so the droplet can learn its own droplet id and delete itself.**

```bash
DROPLET_ID="$(curl -fsS http://169.254.169.254/metadata/v1/id)"
```

That one line is what makes the self-destruct timer **independent of the local machine**. The droplet does not need to be *told* who it is — no id has to be baked into cloud-init at render time, and no local process has to stay alive to supply it. The machine asks the platform, and then asks the platform to delete it.

That independence is the entire reason the safety net survives a `kill -9`, a closed laptop, or a network partition. It is the smallest component in the system and arguably the one doing the most work.

**Failure mode worth naming:** if this service were unreachable, the droplet could not learn its own id and could not self-destruct — the last safety layer would silently be gone. The mitigation is the timer's `OnUnitActiveSec=300s` retry: it keeps trying, every five minutes, forever.
