"""Run the same images through Replicate (nightmareai/real-esrgan) and this Modal app, and compare.

    REPLICATE_API_TOKEN=... uv run --no-project --with modal --with requests --with numpy \
        --with opencv-python-headless python modal_app/compare.py <images dir> [out dir] [--scale 2] [--modal-only]

For each image: output sizes, PSNR between the two outputs, and latency. Outputs are saved to the out dir
(default ./compare-out) as <name>.replicate.png and <name>.modal.png for a visual check.
Without --modal-only it needs REPLICATE_API_TOKEN in the environment (never commit it).
"""

import argparse
import base64
import mimetypes
import os
import statistics
import time
from pathlib import Path

import cv2
import modal
import numpy as np
import requests

REPLICATE_VERSION = "b3ef194191d13140337468c916c2c5b96dd0cb06dffc032a022a31807f6a5ea8"


def data_uri(path: Path) -> str:
    mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
    return f"data:{mime};base64," + base64.b64encode(path.read_bytes()).decode()


def replicate(image: str, scale: float) -> tuple[bytes, float]:
    headers = {"Authorization": f"Bearer {os.environ['REPLICATE_API_TOKEN']}", "Prefer": "wait=60"}
    t0 = time.time()
    r = requests.post(
        "https://api.replicate.com/v1/predictions",
        json={"version": REPLICATE_VERSION, "input": {"image": image, "scale": scale, "face_enhance": True}},
        headers=headers,
        timeout=120,
    )
    r.raise_for_status()
    pred = r.json()
    while pred["status"] not in ("succeeded", "failed", "canceled"):
        time.sleep(1)
        pred = requests.get(pred["urls"]["get"], headers=headers, timeout=30).json()
    if pred["status"] != "succeeded":
        raise RuntimeError(pred.get("error"))
    seconds = time.time() - t0
    return requests.get(pred["output"], timeout=60).content, seconds


def on_modal(upscaler, image: str, scale: float) -> tuple[bytes, float, dict]:
    t0 = time.time()
    out = upscaler.upscale.remote(f"compare-{int(t0 * 1000)}", {"image": image, "scale": scale, "face_enhance": True}, t0)
    seconds = time.time() - t0
    if "error" in out:
        raise RuntimeError(out["error"])
    img = out["image"]
    data = base64.b64decode(img["base64"]) if "base64" in img else requests.get(img["url"], timeout=60).content
    return data, seconds, out


def decode(data: bytes) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_UNCHANGED)


def psnr(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape:
        return float("nan")
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    peak = 65535.0 if a.dtype == np.uint16 else 255.0
    return float("inf") if mse == 0 else 10 * np.log10(peak**2 / mse)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("images")
    ap.add_argument("out", nargs="?", default="compare-out")
    ap.add_argument("--scale", type=float, default=2)
    ap.add_argument("--modal-only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(exist_ok=True)
    upscaler = modal.Cls.from_name("real-esrgan", "Upscaler")()
    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png", ".webp"))
    modal_times, replicate_times = [], []
    for p in paths:
        image = data_uri(p)
        line = f"{p.name:40s}"
        try:
            m_data, m_s, m_out = on_modal(upscaler, image, args.scale)
            (out_dir / f"{p.stem}.modal.png").write_bytes(m_data)
            modal_times.append(m_s)
            m = decode(m_data)
            line += f" modal {m.shape[1]}x{m.shape[0]} {m_s:5.1f}s (gpu {m_out['inference_seconds']:.1f}s)"
        except Exception as e:
            print(f"{line} modal failed: {e}")
            continue
        if not args.modal_only:
            try:
                r_data, r_s = replicate(image, args.scale)
                (out_dir / f"{p.stem}.replicate.png").write_bytes(r_data)
                replicate_times.append(r_s)
                r = decode(r_data)
                line += f" | replicate {r.shape[1]}x{r.shape[0]} {r_s:5.1f}s | PSNR {psnr(m, r):.1f} dB"
            except Exception as e:
                line += f" | replicate failed: {e}"
        print(line)

    for name, times in (("modal", modal_times), ("replicate", replicate_times)):
        if len(times) >= 2:
            p95 = statistics.quantiles(times, n=20)[-1]
            print(f"{name}: median {statistics.median(times):.1f}s, p95 {p95:.1f}s over {len(times)} runs")


if __name__ == "__main__":
    main()
