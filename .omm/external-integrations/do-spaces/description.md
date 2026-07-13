DigitalOcean Spaces — S3-compatible object storage, reached with **boto3** through `src/ml_lab/gpu/spaces.py`. The archive that makes GPU evidence **outlive the droplet that produced it**.

`upload_bundle()` walks the pulled bundle and uploads every file to `s3://<bucket>/ml-pathway/phase0/<run-id>/<relpath>`, returning the `s3://` URI of the run prefix. Keys are POSIX-normalized regardless of local path separators.

Three deliberate details:
- **boto3 is imported lazily**, inside the client factory. Importing this module — or collecting a test that injects a fake client — never needs the SDK loaded.
- **The client is injectable**, so no test ever touches boto3's wire.
- **A missing or empty bundle directory raises `SpacesError`** rather than uploading nothing and reporting success. A silent no-op here would mean losing the only record of a GPU run that just cost real money.

**Credentials are validated separately** from the GPU credentials (`load_spaces_env` vs `load_gpu_env`), because only `gpu_run` uploads — `gpu_up` must not fail merely because Spaces keys are unset. Both are checked in preflight, *before* a droplet exists, so a misconfiguration strands nothing.

Region note: **`nyc2` (where droplets prefer to live) has no Spaces.** `nyc3` is the nearest Spaces region, and is pinned for exactly that reason.
