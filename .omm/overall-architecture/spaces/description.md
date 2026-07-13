External. DigitalOcean Spaces — the S3-compatible object store where GPU benchmark bundles are archived, so the evidence outlives the droplet that produced it.

Reached through `src/ml_lab/gpu/spaces.py` via boto3, pointed at `https://nyc3.digitaloceanspaces.com`. `upload_bundle()` walks the pulled bundle directory and uploads every file to `s3://<bucket>/ml-pathway/phase0/<run-id>/<relpath>`, returning the `s3://` URI of the run prefix. Keys are always POSIX-normalized. A missing or empty bundle directory is itself an error (`SpacesError`) — silently uploading nothing would defeat the purpose.

Two deliberate details:
- **boto3 is imported lazily**, inside the client factory. Importing this module — or collecting a test that injects a fake client — never needs the SDK loaded.
- **Spaces credentials are validated separately** from the GPU credentials (`load_spaces_env` vs `load_gpu_env`). Only `gpu_run` uploads, so `gpu_up` must not fail just because Spaces keys are unset. Both are checked in preflight, *before* a droplet exists, so a misconfiguration strands nothing.

Region note: `nyc2` (where droplets prefer to live) has no Spaces. `nyc3` is the nearest Spaces region, and is pinned for that reason.
