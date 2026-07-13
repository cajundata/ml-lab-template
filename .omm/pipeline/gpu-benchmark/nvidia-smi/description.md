Informational. Raw `nvidia-smi` output, written as text rather than JSON.

It captures **stdout + stderr concatenated** — deliberately, so that a *failing* `nvidia-smi` call is still diagnosable from the artifact bundle. A probe that swallowed stderr would leave you with an empty file and no idea why.

Captured twice, in fact: once by cloud-init during bootstrap (straight into the artifacts dir, before anything else can go wrong), and again by this probe during the benchmark. If the driver state changed between boot and benchmark, the bundle will show it.
