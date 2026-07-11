# Artifacts directory

This directory is the local staging area for GPU benchmark bundles pulled from DigitalOcean droplets before upload to DO Spaces. Nothing here is committed except this README and the `.gitkeep` placeholder. Contents are transient by design: the durable copy of every bundle lives in Spaces, not in this repo.

## Layout

Each GPU run pulls its bundle into `artifacts/<run-id>/`:

- `benchmark.json` holds structured benchmark results.
- `system.json` records OS, Python version, GPU name, GPU memory, driver, and CUDA visibility.
- `nvidia-smi.txt` is the raw nvidia-smi output captured on the droplet.
- `torch_cuda.json` records PyTorch CUDA availability and a small matrix-multiply timing.
- `vllm_smoke.json` records the vLLM import, model load, and generation smoke result.
- `benchmark.log` is the benchmark command log.
- `bootstrap.log` is a copy of the droplet's `/var/log/cloud-init-output.log`, pulled during the artifact step so bootstrap behavior can be inspected after the droplet is destroyed.

After a successful `make gpu-run`, the bundle is uploaded from this machine to `s3://<spaces-bucket>/ml-pathway/phase0/<run-id>/`. Local copies can be deleted at any time once the upload is confirmed.