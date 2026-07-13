**REQUIRED.** The floor: is there a usable GPU here at all?

Two checks, and the second is the one that counts:
1. `torch.cuda.is_available()` — CUDA is visible.
2. **A 256x256 matmul actually executed on the device** (`a @ b` with both tensors `device="cuda"`, then `.sum().item()` to force synchronization).

`ok` is true only if both pass. Visibility alone proves nothing — a driver can report a device that cannot run a kernel. Making the tensors resident on CUDA and forcing a result back is what turns "CUDA is available" into "CUDA works".

Also records `elapsed_s`. **Proven live at S4** on the `gpu-h100x1-base` image running on Hopper (H100/H200).
