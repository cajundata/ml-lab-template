`src/ml_lab/gpu/spaces.py` — the DO Spaces seam. Same shape as `do_client` and `remote`: one external dependency (boto3), one public function, an injectable client factory so no test ever touches boto3's wire.

`upload_bundle(local_dir, run_id, env)` walks the pulled bundle, uploads every file to `s3://<bucket>/ml-pathway/phase0/<run-id>/<relpath>`, and returns the `s3://` URI of the run prefix. Keys are POSIX-normalized regardless of local path separators.

Guards worth noting: a **missing or empty** bundle directory raises `SpacesError` rather than uploading nothing and reporting success — a silent no-op here would mean losing the only evidence of a GPU run that just cost real money. Any boto3 failure is wrapped in `SpacesError` with the original chained.

boto3 is **imported lazily inside the factory**, so importing this module (or collecting a test that injects a fake client) never requires the SDK to load.
