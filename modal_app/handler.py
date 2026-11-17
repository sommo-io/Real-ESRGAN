"""Real-ESRGAN upscale worker, the same model and settings as the Replicate model
nightmareai/real-esrgan (its `cog` branch, predict.py):

    RealESRGAN_x4plus (RRDBNet, 23 blocks), tile=0, tile_pad=10, pre_pad=0, half precision on GPU,
    output resized to `scale` x the input (outscale), and GFPGAN v1.3 ("clean" arch,
    channel_multiplier=2) on detected faces when face_enhance is set, pasted back onto the
    Real-ESRGAN background. Output is a PNG, like on Replicate.

Input (job["input"]):
    image          str   http(s) URL of the image, or base64 (a data: URI prefix is allowed)
    scale          float output size factor, 0 < scale <= 10 (default 4, as on Replicate)
    face_enhance   bool  restore faces with GFPGAN (default true, as on Replicate)
    gcs_bucket     str   upload the output to this GCS bucket (overrides GCS_BUCKET env)
    gcs_prefix     str   object name prefix (overrides GCS_PREFIX env)

Output:
    {"width", "height", "inference_seconds", "image": <file>}
    <file> is {"gcs_uri": "gs://...", "url": "https://..."} when a GCS bucket is set, else {"base64": ...}

GCS env vars (set through Modal secrets, never in this repo):
    GCS_BUCKET, GCS_PREFIX, GCS_SERVICE_ACCOUNT_JSON (raw JSON or base64 of it; if unset,
    Application Default Credentials are used)
"""

import base64
import binascii
import json
import os
import time

import cv2
import numpy as np
import requests
import torch

WEIGHTS_DIR = os.environ.get("ESRGAN_WEIGHTS_DIR", "/models")
ESRGAN_WEIGHTS = os.path.join(WEIGHTS_DIR, "RealESRGAN_x4plus.pth")
GFPGAN_WEIGHTS = os.path.join(WEIGHTS_DIR, "GFPGANv1.3.pth")
# facexlib's face detector and parser weights; GFPGANer looks them up as "gfpgan/weights" relative
# to the working directory, so the worker runs from WEIGHTS_DIR.
FACEXLIB_ROOT = os.path.join(WEIGHTS_DIR, "gfpgan", "weights")
# Replicate's model fails above ~2.1 MP (1448x1448) with tile=0; keep the same limit so a huge
# input can't take a GPU down with an out-of-memory error.
MAX_INPUT_PIXELS = int(os.environ.get("ESRGAN_MAX_INPUT_PIXELS", str(1448 * 1448)))

upsampler = None
face_enhancer = None
_gcs_client = None


def load(device: torch.device) -> None:
    """Build both models on `device` (the Modal app builds them on CPU, then calls to_device)."""
    global upsampler, face_enhancer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    from gfpgan import GFPGANer
    from realesrgan import RealESRGANer

    half = device.type == "cuda"
    model = RRDBNet(num_in_ch=3, num_out_ch=3, num_feat=64, num_block=23, num_grow_ch=32, scale=4)
    upsampler = RealESRGANer(
        scale=4, model_path=ESRGAN_WEIGHTS, model=model, tile=0, tile_pad=10, pre_pad=0, half=half, device=device
    )
    cwd = os.getcwd()
    os.chdir(WEIGHTS_DIR)
    try:
        face_enhancer = GFPGANer(
            model_path=GFPGAN_WEIGHTS,
            upscale=4,
            arch="clean",
            channel_multiplier=2,
            bg_upsampler=upsampler,
            device=device,
        )
    finally:
        os.chdir(cwd)


def _move(obj, device: torch.device) -> None:
    """Point a helper object at `device`: its modules, plain tensor attributes and `device` field."""
    for name, value in list(vars(obj).items()):
        if isinstance(value, torch.nn.Module):
            value.to(device)
            _move(value, device)
        elif isinstance(value, torch.Tensor):
            setattr(obj, name, value.to(device))
        elif isinstance(value, torch.device):
            setattr(obj, name, device)


def to_device(device: torch.device) -> None:
    """Move CPU-built models to the GPU (after a memory snapshot restore), half precision for
    Real-ESRGAN as on Replicate. The face detector and parser stay fp32 (half=False upstream)."""
    _move(upsampler, device)
    if device.type == "cuda":
        upsampler.model = upsampler.model.half()
        upsampler.half = True
    _move(face_enhancer, device)
    _move(face_enhancer.face_helper, device)


def gcs():
    global _gcs_client
    if _gcs_client is None:
        from google.cloud import storage
        from google.oauth2 import service_account

        raw = os.environ.get("GCS_SERVICE_ACCOUNT_JSON", "").strip()
        if raw:
            if not raw.startswith("{"):
                raw = base64.b64decode(raw).decode("utf-8")
            info = json.loads(raw)
            creds = service_account.Credentials.from_service_account_info(info)
            _gcs_client = storage.Client(project=info.get("project_id"), credentials=creds)
        else:
            _gcs_client = storage.Client()
    return _gcs_client


def _read_image(value: str) -> np.ndarray:
    if value.startswith(("http://", "https://")):
        resp = requests.get(value, timeout=60)
        resp.raise_for_status()
        data = resp.content
    else:
        if value.startswith("data:"):
            value = value.split(",", 1)[1]
        data = base64.b64decode(value, validate=False)
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError("not a readable image")
    return img


def upscale(img: np.ndarray, scale: float, face_enhance: bool) -> np.ndarray:
    """predict.py from the Replicate model, with the GFPGAN helper reused instead of rebuilt."""
    with torch.no_grad():
        if face_enhance:
            face_enhancer.upscale = scale
            face_enhancer.face_helper.upscale_factor = scale
            _, _, output = face_enhancer.enhance(img, has_aligned=False, only_center_face=False, paste_back=True)
        else:
            output, _ = upsampler.enhance(img, outscale=scale)
    return output


def handler(job: dict) -> dict:
    inp = job.get("input") or {}
    job_id = job.get("id", "local")
    try:
        scale = float(inp.get("scale", 4))
    except (TypeError, ValueError):
        return {"error": "scale must be a number"}
    if not 0 < scale <= 10:
        return {"error": f"scale must be in (0, 10], got {scale}"}
    face_enhance = bool(inp.get("face_enhance", True))
    bucket = inp.get("gcs_bucket") or os.environ.get("GCS_BUCKET")
    prefix = inp.get("gcs_prefix", os.environ.get("GCS_PREFIX", ""))

    if not isinstance(inp.get("image"), str) or not inp["image"]:
        return {"error": "Provide 'image' (URL or base64)"}
    try:
        img = _read_image(inp["image"])
    except (requests.RequestException, ValueError, binascii.Error) as e:
        return {"error": f"Could not read input image: {e}"}
    h, w = img.shape[:2]
    if h * w > MAX_INPUT_PIXELS:
        return {"error": f"input is {w}x{h}; the limit is {MAX_INPUT_PIXELS} pixels"}

    t0 = time.perf_counter()
    try:
        output = upscale(img, scale, face_enhance)
    finally:
        # RealESRGANer keeps its last input/output tensors; free them between jobs.
        for name in ("img", "output"):
            if hasattr(upsampler, name):
                delattr(upsampler, name)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    inference_seconds = round(time.perf_counter() - t0, 3)

    ok, png = cv2.imencode(".png", output)
    if not ok:
        return {"error": "Could not encode the output PNG"}
    data = png.tobytes()
    if bucket:
        name = f"{prefix}{job_id}.png"
        try:
            blob = gcs().bucket(bucket).blob(name)
            blob.upload_from_string(data, content_type="image/png")
        except Exception as e:
            raise RuntimeError(f"GCS upload to gs://{bucket} failed: {e}") from e
        file = {"gcs_uri": f"gs://{bucket}/{name}", "url": blob.public_url}
    else:
        file = {"base64": base64.b64encode(data).decode("ascii")}

    return {
        "width": int(output.shape[1]),
        "height": int(output.shape[0]),
        "inference_seconds": inference_seconds,
        "image": file,
    }
