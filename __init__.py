"""
ComfyUI custom nodes for LTX-2.3 Video Generation API (ltx.video)
Supports: text-to-video, image-to-video, audio-to-video (lip-sync), extend, retake
"""

import os
import io
import time
import requests
import tempfile
import subprocess
import numpy as np
import torch
from PIL import Image

try:
    import folder_paths
    COMFY_AVAILABLE = True
except ImportError:
    COMFY_AVAILABLE = False

LTX_BASE_URL = "https://api.ltx.video/v1"
UGUU_UPLOAD_URL = "https://uguu.se/upload"


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_uguu(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes to uguu.se and return public HTTPS URL."""
    files = {"files[]": (filename, data, content_type)}
    r = requests.post(UGUU_UPLOAD_URL, files=files, timeout=60)
    r.raise_for_status()
    j = r.json()
    url = j["files"][0]["url"]
    print(f"[LTX] Uploaded {filename} → {url}")
    return url


def tensor_to_jpeg_bytes(image_tensor, max_dim=1920) -> bytes:
    """
    Convert ComfyUI IMAGE tensor [B, H, W, C] or [H, W, C] to JPEG bytes.
    Resizes to fit within max_dim to respect LTX API size limit.
    """
    if image_tensor.ndim == 4:
        image_tensor = image_tensor[0]  # take first batch

    np_img = (image_tensor.cpu().numpy() * 255).clip(0, 255).astype(np.uint8)
    pil = Image.fromarray(np_img)

    # Resize if too large (LTX rejects images > 1920x1080)
    w, h = pil.size
    if max(w, h) > max_dim:
        ratio = max_dim / max(w, h)
        pil = pil.resize((int(w * ratio), int(h * ratio)), Image.LANCZOS)
        print(f"[LTX] Resized image from {w}x{h} → {pil.size[0]}x{pil.size[1]}")

    buf = io.BytesIO()
    pil.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def audio_tensor_to_mp3_bytes(audio: dict) -> bytes:
    """
    Convert ComfyUI AUDIO dict {'waveform': Tensor[B,C,T], 'sample_rate': int}
    to MP3 bytes via ffmpeg (raw PCM pipe).
    LTX requires audio/mpeg MIME type.
    """
    waveform = audio["waveform"]   # [B, C, T] float32 in [-1, 1]
    sample_rate = audio["sample_rate"]

    # Take first batch, mix to mono or keep stereo
    if waveform.ndim == 3:
        waveform = waveform[0]     # [C, T]
    if waveform.shape[0] > 2:
        waveform = waveform[:2]    # max stereo

    channels = waveform.shape[0]
    # Convert to 16-bit PCM bytes
    pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
    # Interleave channels: [T, C] → flatten
    pcm_bytes = pcm.T.flatten().tobytes()

    # ffmpeg: read raw PCM from stdin, output MP3 to stdout
    import subprocess
    cmd = [
        "ffmpeg", "-y",
        "-f", "s16le",
        "-ar", str(sample_rate),
        "-ac", str(channels),
        "-i", "pipe:0",
        "-acodec", "libmp3lame",
        "-q:a", "2",
        "-f", "mp3",
        "pipe:1",
    ]
    proc = subprocess.run(cmd, input=pcm_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg MP3 encode failed: {proc.stderr.decode()[:200]}")
    print(f"[LTX] Audio encoded: {len(proc.stdout)/1024:.1f} KB MP3 @ {sample_rate}Hz {channels}ch")
    return proc.stdout


def get_output_path(prefix: str, ext: str = "mp4") -> str:
    if COMFY_AVAILABLE:
        out_dir = folder_paths.get_output_directory()
    else:
        out_dir = tempfile.gettempdir()
    ts = int(time.time())
    return os.path.join(out_dir, f"ltx_{prefix}_{ts}.{ext}")


def download_video(r: requests.Response, prefix: str) -> str:
    out_path = get_output_path(prefix)
    with open(out_path, "wb") as f:
        for chunk in r.iter_content(chunk_size=8192):
            f.write(chunk)
    print(f"[LTX] Video saved: {out_path}")
    return out_path


def video_bytes_to_image_tensor(video_bytes: bytes) -> torch.Tensor:
    """
    Decode video bytes → ComfyUI IMAGE tensor [F, H, W, C] float32 in [0,1].
    Uses ffmpeg to extract frames as PNG via pipe.
    """
    cmd = [
        "ffmpeg", "-i", "pipe:0",
        "-f", "image2pipe", "-vcodec", "png", "-",
        "-loglevel", "error"
    ]
    proc = subprocess.run(cmd, input=video_bytes, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ffmpeg frame extraction failed: {proc.stderr.decode()[:300]}")

    # Split PNG stream into individual frames
    raw = proc.stdout
    frames = []
    i = 0
    while i < len(raw):
        # PNG signature: 8 bytes \x89PNG\r\n\x1a\n
        start = raw.find(b'\x89PNG\r\n\x1a\n', i)
        if start == -1:
            break
        end = raw.find(b'\x89PNG\r\n\x1a\n', start + 8)
        chunk = raw[start:end] if end != -1 else raw[start:]
        img = Image.open(io.BytesIO(chunk)).convert("RGB")
        frames.append(np.array(img, dtype=np.float32) / 255.0)
        i = end if end != -1 else len(raw)

    if not frames:
        raise RuntimeError("No frames decoded from LTX video output.")

    print(f"[LTX] Decoded {len(frames)} frames")
    return torch.from_numpy(np.stack(frames))  # [F, H, W, C]


def ltx_post(endpoint: str, api_key: str, payload: dict) -> tuple:
    """POST to LTX API, save video, return (image_tensor, video_path)."""
    url = f"{LTX_BASE_URL}/{endpoint}"
    print(f"[LTX] → POST {endpoint} | payload keys: {list(payload.keys())}")
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=300,
    )
    if r.status_code != 200:
        raise RuntimeError(f"LTX API {r.status_code}: {r.text[:400]}")

    video_bytes = r.content
    out_path = get_output_path(endpoint.replace("-", "_"))
    with open(out_path, "wb") as f:
        f.write(video_bytes)
    print(f"[LTX] Video saved: {out_path}")

    frames = video_bytes_to_image_tensor(video_bytes)
    return frames, out_path


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Audio to Video (Lip-sync)  ← MAIN NEW NODE
# ─────────────────────────────────────────────────────────────────────────────

class LTXAudioToVideo:
    """
    LTX-2.3 Audio-to-Video: lip-sync video from image + audio.
    Accepts native ComfyUI IMAGE tensor and local audio path.
    Automatically uploads both to uguu.se (HTTPS) as required by LTX API.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "LTX API key from ltx.video/api-keys",
                }),
                "image": ("IMAGE",),
                "audio": ("AUDIO", {
                    "tooltip": "Connect a Load Audio node here.",
                }),
                "prompt": ("STRING", {
                    "default": "A person speaking naturally, slight head movement, realistic lip sync.",
                    "multiline": True,
                }),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "resolution": (["1080x1920", "1920x1080"], {
                    "default": "1080x1920",
                    "tooltip": "Audio-to-video only supports these two resolutions.",
                }),
                "duration": ("INT", {
                    "default": 0,
                    "min": 0,
                    "max": 30,
                    "tooltip": "Duration in seconds. Set 0 to auto-match audio length.",
                }),
                "negative_prompt": ("STRING", {
                    "default": "blurry, distorted, low quality, static, frozen",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, image, audio, prompt,
                 model="ltx-2-3-pro", resolution="1080x1920",
                 duration=0, negative_prompt="blurry, distorted, low quality"):

        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        # 1. Upload image
        print("[LTX] Uploading image...")
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=1920)
        image_url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")

        # 2. Convert AUDIO tensor → MP3 and upload
        print("[LTX] Encoding + uploading audio...")
        audio_bytes = audio_tensor_to_mp3_bytes(audio)
        audio_url = upload_to_uguu(audio_bytes, "ltx_audio.mp3", "audio/mpeg")

        # 3. Build payload
        payload = {
            "audio_uri": audio_url,
            "image_uri": image_url,
            "prompt": prompt,
            "model": model,
            "resolution": resolution,
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if duration and duration > 0:
            payload["duration"] = duration

        frames, video_path = ltx_post("audio-to-video", api_key.strip(), payload)
        return (frames, video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Text to Video
# ─────────────────────────────────────────────────────────────────────────────

class LTXTextToVideo:
    """Generate video from text prompt using LTX-2.3 API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "resolution": (["1920x1080", "1080x1920", "1440x1080", "4096x2160"], {"default": "1920x1080"}),
                "duration": ("INT", {"default": 8, "min": 6, "max": 20}),
                "negative_prompt": ("STRING", {
                    "default": "blurry, distorted, low quality",
                    "multiline": True,
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647,
                                 "tooltip": "-1 for random."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, prompt,
                 model="ltx-2-3-pro", resolution="1920x1080",
                 duration=8, negative_prompt="", seed=-1):
        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        payload = {
            "prompt": prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if seed >= 0:
            payload["seed"] = seed

        frames, video_path = ltx_post("text-to-video", api_key.strip(), payload)
        return (frames, video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Image to Video
# ─────────────────────────────────────────────────────────────────────────────

class LTXImageToVideo:
    """Animate a static image using LTX-2.3 API. Accepts ComfyUI IMAGE tensor."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image": ("IMAGE",),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "resolution": (["1920x1080", "1080x1920", "1440x1080", "4096x2160"], {"default": "1920x1080"}),
                "duration": ("INT", {"default": 8, "min": 6, "max": 20}),
                "negative_prompt": ("STRING", {
                    "default": "blurry, distorted, low quality",
                    "multiline": True,
                }),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, image, prompt,
                 model="ltx-2-3-pro", resolution="1920x1080",
                 duration=8, negative_prompt="", seed=-1):
        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        print("[LTX] Uploading image...")
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=1920)
        image_url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")

        payload = {
            "image_uri": image_url,
            "prompt": prompt,
            "model": model,
            "resolution": resolution,
            "duration": duration,
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt
        if seed >= 0:
            payload["seed"] = seed

        frames, video_path = ltx_post("image-to-video", api_key.strip(), payload)
        return (frames, video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Extend Video
# ─────────────────────────────────────────────────────────────────────────────

class LTXExtendVideo:
    """Extend an existing video at the start or end using LTX-2.3 API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HTTPS URL to the video to extend.",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "duration": ("INT", {"default": 6, "min": 2, "max": 20}),
                "direction": (["end", "beginning"], {"default": "end"}),
                "negative_prompt": ("STRING", {
                    "default": "blurry, distorted, low quality",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "extend"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def extend(self, api_key, video_url, prompt,
               model="ltx-2-3-pro", duration=6, direction="end", negative_prompt=""):
        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        payload = {
            "video_uri": video_url.strip(),
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "direction": direction,
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt

        frames, video_path = ltx_post("extend", api_key.strip(), payload)
        return (frames, video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Retake (regenerate section)
# ─────────────────────────────────────────────────────────────────────────────

class LTXRetakeVideo:
    """Regenerate a specific section of a video using LTX-2.3 API."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "video_url": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "HTTPS URL to the video to retake.",
                }),
                "prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "start_frame": ("INT", {"default": 0, "min": 0, "max": 999}),
                "end_frame": ("INT", {"default": 24, "min": 1, "max": 999}),
                "negative_prompt": ("STRING", {
                    "default": "blurry, distorted, low quality",
                    "multiline": True,
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("frames", "video_path")
    FUNCTION = "retake"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def retake(self, api_key, video_url, prompt,
               model="ltx-2-3-pro", start_frame=0, end_frame=24, negative_prompt=""):
        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        payload = {
            "video_uri": video_url.strip(),
            "prompt": prompt,
            "model": model,
            "start_frame": start_frame,
            "end_frame": end_frame,
        }
        if negative_prompt.strip():
            payload["negative_prompt"] = negative_prompt

        frames, video_path = ltx_post("retake", api_key.strip(), payload)
        return (frames, video_path)


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Image Uploader (utility)
# ─────────────────────────────────────────────────────────────────────────────

class LTXImageUploader:
    """
    Uploads a ComfyUI IMAGE tensor to uguu.se and returns a public HTTPS URL.
    Useful as input for LTXImageToVideo or LTXAudioToVideo when chaining nodes.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
            },
            "optional": {
                "max_dimension": ("INT", {
                    "default": 1920,
                    "min": 256,
                    "max": 4096,
                    "tooltip": "Images larger than this will be resized (LTX limit: 1920).",
                }),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("image_url",)
    FUNCTION = "upload"
    CATEGORY = "LTX Video"

    def upload(self, image, max_dimension=1920):
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=max_dimension)
        url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")
        return (url,)


# ─────────────────────────────────────────────────────────────────────────────
# REGISTRATIONS
# ─────────────────────────────────────────────────────────────────────────────

NODE_CLASS_MAPPINGS = {
    "LTXAudioToVideo": LTXAudioToVideo,
    "LTXTextToVideo": LTXTextToVideo,
    "LTXImageToVideo": LTXImageToVideo,
    "LTXExtendVideo": LTXExtendVideo,
    "LTXRetakeVideo": LTXRetakeVideo,
    "LTXImageUploader": LTXImageUploader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXAudioToVideo": "LTX Audio to Video 🎤➡️🎬",
    "LTXTextToVideo": "LTX Text to Video 📝➡️🎬",
    "LTXImageToVideo": "LTX Image to Video 🖼️➡️🎬",
    "LTXExtendVideo": "LTX Extend Video ➕🎬",
    "LTXRetakeVideo": "LTX Retake Section 🔁🎬",
    "LTXImageUploader": "LTX Image Uploader ☁️",
}

print("[LTX] LTX-2.3 ComfyUI nodes loaded ✅ — Audio/Image/Text to Video + Extend + Retake")
