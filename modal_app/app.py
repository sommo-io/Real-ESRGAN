"""Modal deployment of Real-ESRGAN, with a RunPod-style queue API (same as the demucs and
resemble-enhance apps).

Runs modal_app/handler.py: the same model and settings as the Replicate model
nightmareai/real-esrgan (RealESRGAN_x4plus + GFPGAN v1.3 face enhance), so inputs and outputs match.

Deploy from the repo root:
    modal_app/deploy.sh

HTTP API (Modal proxy auth: send Modal-Key / Modal-Secret headers from a proxy auth token):
    POST /run            {"input": {...}, "webhook"?: "https://..."}  -> {"id": "fc-...", "status": "IN_QUEUE"}
    POST /runsync        same body, plus "wait"?: seconds (1-90, default 90)
                         -> finished body, or {"id", "status": "IN_PROGRESS" | "IN_QUEUE"} after `wait`
    GET  /status/{id}                                  -> {"id", "status", "output" | "error"}
    POST /cancel/{id}                                  -> {"id", "status": "CANCELLED"}
status is IN_QUEUE, IN_PROGRESS (a worker picked it up), COMPLETED, FAILED, CANCELLED or TIMED_OUT.

With "webhook", the final /status body is POSTed there for every terminal status, signed with
WEBHOOK_SECRET from the Modal secret "audio-webhook" (see _send_webhook).

GCS upload is configured only through Modal secrets (nothing deployment-specific is kept in this
public repo): "demucs-gcs" holds GCS_SERVICE_ACCOUNT_JSON, "real-esrgan-config" holds GCS_BUCKET and
GCS_PREFIX.
"""

import time

import modal

# Tried in order when a container starts: L4 (~$0.80/h), then A10 (~$1.10/h), then L40S (~$1.95/h,
# often free) instead of waiting for L4/A10 capacity. Snapshots are per GPU type.
GPU = ["L4", "A10", "L40S"]
WEIGHTS_DIR = "/models"
# Same files the Replicate model downloads (its cog.yaml), from the official releases.
WEIGHTS = {
    "RealESRGAN_x4plus.pth": "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    "GFPGANv1.3.pth": "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.0/GFPGANv1.3.pth",
}

app = modal.App("real-esrgan")

# call id -> start time, written when a worker picks the job up, so /status can tell
# IN_QUEUE from IN_PROGRESS (Modal itself only knows "not finished yet").
started = modal.Dict.from_name("real-esrgan-started", create_if_missing=True)

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libgl1", "libglib2.0-0", "wget")
    .pip_install("torch==2.5.1", "torchvision==0.20.1", "numpy<2")
    # basicsr builds from source and imports torch in setup.py: no build isolation.
    .run_commands("pip install --no-build-isolation basicsr==1.4.2")
    .pip_install(
        "facexlib==0.3.0",
        "gfpgan==1.3.8",
        "opencv-python-headless==4.10.0.84",
        "google-cloud-storage==2.18.2",
        "requests",
    )
    # basicsr 1.4.2 imports torchvision.transforms.functional_tensor, removed in torchvision 0.17.
    .run_commands(
        "sed -i 's/torchvision.transforms.functional_tensor/torchvision.transforms.functional/' "
        "$(python -c 'import sysconfig; print(sysconfig.get_paths()[\"purelib\"])')/basicsr/data/degradations.py"
    )
    # The realesrgan package from this repo (as Replicate's predict.py does with setup.py develop).
    .add_local_dir(
        "realesrgan", "/opt/real-esrgan/realesrgan", copy=True, ignore=["**/__pycache__", "weights/*.pth"]
    )
    .add_local_file("setup.py", "/opt/real-esrgan/setup.py", copy=True)
    .add_local_file("setup.cfg", "/opt/real-esrgan/setup.cfg", copy=True)
    .add_local_file("VERSION", "/opt/real-esrgan/VERSION", copy=True)
    .add_local_file("README.md", "/opt/real-esrgan/README.md", copy=True)
    .add_local_file("requirements.txt", "/opt/real-esrgan/requirements.txt", copy=True)
    .run_commands("pip install --no-deps --no-build-isolation /opt/real-esrgan")
    .run_commands(
        f"mkdir -p {WEIGHTS_DIR}",
        *[f"wget -q -O {WEIGHTS_DIR}/{name} {url}" for name, url in WEIGHTS.items()],
    )
    # Memory snapshots keep os.environ as it was when they were taken, so a container restored
    # from one still sees the old values of a changed Modal secret (GCS_BUCKET, GCS_PREFIX).
    # After changing real-esrgan-config, bump SNAPSHOT_CONFIG and redeploy: that rebuilds the
    # snapshot. (Rev 2: GCS_PREFIX moved to the bucket's short-lived previews prefix.)
    .env({"ESRGAN_WEIGHTS_DIR": WEIGHTS_DIR, "PYTHONPATH": "/opt/worker", "SNAPSHOT_CONFIG": "2"})
    .add_local_file("modal_app/handler.py", "/opt/worker/handler.py", copy=True)
    # One CPU load bakes facexlib's face detector and parser weights into the image too, so cold
    # starts never download anything.
    .run_commands("cd /tmp && python -c \"import torch, handler; handler.load(torch.device('cpu'))\"")
)


@app.cls(
    gpu=GPU,
    cpu=2.0,
    memory=8192,
    image=image,
    secrets=[modal.Secret.from_name("demucs-gcs"), modal.Secret.from_name("real-esrgan-config")],
    # A 1448x1448 input (the limit) at 4x takes a few seconds; anything near this is stuck.
    timeout=300,
    # Previews come in bursts while people upload; a cold start costs several seconds.
    # Idle time is billed (~$0.013/min on an L4).
    scaledown_window=120,
    # No max_containers: the workspace's 10 GPUs (Starter) are shared with demucs and
    # resemble-enhance (each capped at 5), and previews take whatever is free.
    enable_memory_snapshot=True,
)
class Upscaler:
    # Memory snapshot: imports and the CPU model load are captured once, so new containers
    # restore them instead of redoing them. There's no GPU during this phase.
    @modal.enter(snap=True)
    def load_cpu(self):
        t0 = time.perf_counter()
        import cv2
        import numpy as np
        import torch

        import handler

        self.handler = handler
        handler.load(torch.device("cpu"))
        # One tiny pass so lazy imports and first-call setup land in the snapshot, not in the
        # first request (a cancel mid-import would leave the container broken).
        handler.upscale(np.zeros((32, 32, 3), np.uint8), 2, True)
        cv2.imencode(".png", np.zeros((8, 8, 3), np.uint8))
        try:
            handler.gcs()
        except Exception as e:  # no credentials configured: outputs fall back to base64
            print(f"GCS client not created: {e}")
        print(f"models loaded and warmed on cpu in {time.perf_counter() - t0:.2f}s")

    @modal.enter(snap=False)
    def load_gpu(self):
        t0 = time.perf_counter()
        import torch

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.handler.to_device(device)
        print(f"models moved to {device} in {time.perf_counter() - t0:.2f}s")

    @modal.method()
    def warmup(self, hold_seconds: float = 0) -> dict:
        """Start a container (building a memory snapshot if its worker type has none) without
        touching storage. deploy.sh calls it in parallel after each deploy."""
        import os

        import torch

        time.sleep(hold_seconds)
        gpu = torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"
        return {"task": os.environ.get("MODAL_TASK_ID"), "device": gpu}

    @modal.method()
    def upscale(self, job_id: str, inp: dict, submitted_at: float) -> dict:
        import torch

        started_at = time.time()
        started[modal.current_function_call_id()] = started_at
        torch.cuda.reset_peak_memory_stats()
        try:
            out = self.handler.handler({"id": job_id, "input": inp})
        except (AttributeError, ImportError) as e:
            if "partially initialized module" in str(e) or isinstance(e, ImportError):
                # Module state is corrupt (e.g. an import interrupted by a cancel): stop taking
                # inputs so Modal replaces this container instead of failing every job on it.
                modal.experimental.stop_fetching_inputs()
            raise
        except torch.cuda.OutOfMemoryError as e:
            torch.cuda.empty_cache()
            return {"error": f"out of GPU memory: {e}"}
        out["queue_seconds"] = round(started_at - submitted_at, 3)
        out["total_seconds"] = round(time.time() - started_at, 3)
        out["peak_gpu_mb"] = torch.cuda.max_memory_allocated() // 2**20
        return out


def _status_body(call_id: str, out) -> dict:
    """The /status response for a finished call; webhooks send the same body."""
    if isinstance(out, dict) and "error" in out:
        return {"id": call_id, "status": "FAILED", "error": out["error"]}
    return {"id": call_id, "status": "COMPLETED", "output": out}


def _error_status(call_id: str, e: Exception) -> dict:
    """Map an exception from FunctionCall.get to a RunPod-style terminal status."""
    if isinstance(e, modal.exception.FunctionTimeoutError):
        status = "TIMED_OUT"
    elif "cancelled" in str(e).lower():  # a cancelled call surfaces as RemoteError("Function call was cancelled ...")
        status = "CANCELLED"
    else:
        status = "FAILED"
    return {"id": call_id, "status": status, "error": f"{type(e).__name__}: {e}"}


def _send_webhook(url: str, body: dict) -> None:
    """POST body to url, signed with WEBHOOK_SECRET; 5 attempts over ~12 min, never raises.

    Headers: X-Webhook-Timestamp: <unix seconds>
             X-Webhook-Signature: v1=<hex HMAC-SHA256(secret, f"{timestamp}.{raw body}")>
    """
    import hashlib
    import hmac
    import json
    import os

    import requests

    raw = json.dumps(body, separators=(",", ":")).encode()
    for attempt, delay in enumerate((0, 5, 30, 120, 600)):
        time.sleep(delay)
        ts = str(int(time.time()))
        sig = hmac.new(os.environ["WEBHOOK_SECRET"].encode(), f"{ts}.".encode() + raw, hashlib.sha256).hexdigest()
        headers = {"Content-Type": "application/json", "X-Webhook-Timestamp": ts, "X-Webhook-Signature": f"v1={sig}"}
        try:
            r = requests.post(url, data=raw, headers=headers, timeout=10)
            if r.status_code < 300:
                return
            print(f"webhook attempt {attempt + 1}: HTTP {r.status_code}")
        except requests.RequestException as e:
            print(f"webhook attempt {attempt + 1}: {e}")
    print(f"webhook to {url} failed; the result is still available via /status")


web_image = modal.Image.debian_slim(python_version="3.11").pip_install("fastapi[standard]", "requests")


@app.function(image=web_image, secrets=[modal.Secret.from_name("audio-webhook")], timeout=6 * 3600)
@modal.concurrent(max_inputs=1000)
async def watch(call_id: str, webhook: str) -> None:
    """Wait for a job to reach ANY terminal state and deliver the webhook.

    Runs outside the GPU worker, so timeouts, crashes and cancellations still produce a
    webhook (TIMED_OUT / FAILED / CANCELLED), like RunPod's platform-side webhooks.
    Only waits, so one small CPU container serves many jobs at once.
    """
    import asyncio

    try:
        body = _status_body(call_id, await modal.FunctionCall.from_id(call_id).get.aio())
    except Exception as e:
        body = _error_status(call_id, e)
    await asyncio.to_thread(_send_webhook, webhook, body)


@app.function(image=web_image)
@modal.concurrent(max_inputs=100)  # requests only enqueue/poll/wait, so one container serves many
@modal.asgi_app(requires_proxy_auth=True)
def api():
    import uuid

    from fastapi import FastAPI, HTTPException

    web = FastAPI()

    async def spawn(body: dict):
        inp = body.get("input")
        if not isinstance(inp, dict):
            raise HTTPException(400, "body must be {\"input\": {...}, \"webhook\"?: \"https://...\"}")
        webhook = body.get("webhook")
        if webhook is not None and not (isinstance(webhook, str) and webhook.startswith("https://")):
            raise HTTPException(400, "webhook must be an https:// URL")
        call = await Upscaler().upscale.spawn.aio(uuid.uuid4().hex, inp, time.time())
        if webhook:
            await watch.spawn.aio(call.object_id, webhook)
        return call

    @web.post("/run")
    async def run(body: dict):
        call = await spawn(body)
        return {"id": call.object_id, "status": "IN_QUEUE"}

    @web.post("/runsync")
    async def runsync(body: dict):
        """Waits up to `wait` s (default and max 90); if the job isn't done by then, returns
        IN_PROGRESS/IN_QUEUE and the id, to poll or cancel. A caller with a shorter deadline
        sends a shorter `wait`, so it always gets the id back in time to cancel."""
        wait = body.get("wait", 90)
        if not isinstance(wait, (int, float)) or isinstance(wait, bool) or not 1 <= wait <= 90:
            raise HTTPException(400, "wait must be a number of seconds from 1 to 90")
        call = await spawn(body)
        try:
            out = await call.get.aio(timeout=wait)
        except TimeoutError:  # builtin: not finished within 90 s (a job timeout is FunctionTimeoutError)
            running = await started.contains.aio(call.object_id)
            return {"id": call.object_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except Exception as e:
            return _error_status(call.object_id, e)
        return _status_body(call.object_id, out)

    def check_id(call_id: str) -> None:
        if not call_id.startswith("fc-"):  # Modal raises ExecutionError, not NotFound, for these
            raise HTTPException(404, "job not found")

    @web.get("/status/{call_id}")
    async def status(call_id: str):
        check_id(call_id)
        try:
            out = await modal.FunctionCall.from_id(call_id).get.aio(timeout=0)
        except TimeoutError:  # builtin: not finished yet (a job timeout is FunctionTimeoutError, handled below)
            running = await started.contains.aio(call_id)
            return {"id": call_id, "status": "IN_PROGRESS" if running else "IN_QUEUE"}
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        except Exception as e:  # the call itself raised (crash, timeout, cancel)
            return _error_status(call_id, e)
        return _status_body(call_id, out)

    @web.post("/cancel/{call_id}")
    async def cancel(call_id: str):
        check_id(call_id)
        try:
            await modal.FunctionCall.from_id(call_id).cancel.aio()
        except (modal.exception.NotFoundError, modal.exception.InvalidError):
            raise HTTPException(404, "job not found")
        return {"id": call_id, "status": "CANCELLED"}

    return web
