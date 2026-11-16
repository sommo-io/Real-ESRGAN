# Modal deployment

Runs Real-ESRGAN on Modal with the same model and settings as the Replicate model
[`nightmareai/real-esrgan`](https://replicate.com/nightmareai/real-esrgan) (the `cog` branch of this fork:
`RealESRGAN_x4plus`, `tile=0`, fp16, GFPGAN v1.3 for faces), behind the same RunPod-style queue API as our
demucs and resemble-enhance apps. Upstream files stay untouched; everything is in `modal_app/`.

```bash
modal_app/deploy.sh     # modal deploy + pre-build memory snapshots (needs modal CLI and uv)
```

Keep this directory out of a folder named `modal` (it would shadow the `modal` package).

## API

`modal deploy` prints the URL (`https://<workspace>--real-esrgan-api.modal.run`). It is protected by Modal
proxy auth: send the `Modal-Key` / `Modal-Secret` headers from a proxy auth token.

| request | response |
|---|---|
| `POST /run` `{"input": {...}}` | `{"id": "fc-...", "status": "IN_QUEUE"}` |
| `POST /runsync` `{"input": {...}, "wait"?: 1-90}` | the finished job, or `{"id", "status": "IN_PROGRESS" \| "IN_QUEUE"}` after `wait` s (default 90) |
| `GET /status/{id}` | `{"id", "status": "IN_PROGRESS" \| "COMPLETED" \| "FAILED" \| ..., "output" \| "error"}` |
| `POST /cancel/{id}` | `{"id", "status": "CANCELLED"}` |

`input` (same names and defaults as on Replicate):

| field | |
|---|---|
| `image` | http(s) URL, or base64 (a `data:` URI prefix is fine) |
| `scale` | output size factor, `0 < scale <= 10`, default 4 |
| `face_enhance` | GFPGAN on detected faces, default `true` |

Inputs above 1448×1448 px are rejected, as on Replicate.

`output`: `{"width", "height", "image": {"gcs_uri", "url"}, "inference_seconds", "queue_seconds", "total_seconds", "peak_gpu_mb"}`.

From Python (no HTTP auth needed):

```python
import time, modal
up = modal.Cls.from_name("real-esrgan", "Upscaler")()
out = up.upscale.remote("job-id", {"image": "https://...", "scale": 2, "face_enhance": True}, time.time())
```

`compare.py` runs the same images through Replicate and Modal and reports size, pixel difference and latency.

## Setup

- GPU L4, falling back to A10 and then L40S; 2 CPU, 8 GB RAM, timeout 5 min, scale to zero after 2 min
  idle, no container cap of its own (shares the workspace's 10 GPUs with demucs and resemble-enhance).
- Weights (RealESRGAN_x4plus, GFPGANv1.3, facexlib detector and parser) are baked into the image.
- Memory snapshot: imports and the CPU model load are snapshotted; a new container restores it and only
  moves the models to the GPU.
- GCS output is configured only through Modal secrets, so no bucket names live in this public repo:
  `demucs-gcs` (`GCS_SERVICE_ACCOUNT_JSON`) and `real-esrgan-config` (`GCS_BUCKET`, `GCS_PREFIX`).
  Create the latter with `modal secret create real-esrgan-config GCS_BUCKET=<bucket> GCS_PREFIX=<prefix>`.
  The objects must be publicly readable (the caller downloads `url`); a short lifecycle rule on the prefix
  is enough, callers copy the result right away.
