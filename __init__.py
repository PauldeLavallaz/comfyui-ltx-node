"""
ComfyUI custom nodes for LTX-2.3 Video Generation API (ltx.video)
Supports: text-to-video, image-to-video, audio-to-video (lip-sync), extend, retake
"""

import os
import io
import time
import requests
import tempfile
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


def audio_path_to_mp3_bytes(audio_path: str) -> bytes:
    """
    Read an audio file. If it's OGG/WAV, convert to MP3 via ffmpeg.
    Returns MP3 bytes (LTX requires audio/mpeg MIME type).
    """
    audio_path = os.path.expanduser(audio_path.strip())
    if not os.path.isfile(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    ext = os.path.splitext(audio_path)[1].lower()
    if ext == ".mp3":
        with open(audio_path, "rb") as f:
            return f.read()

    # Convert to MP3
    tmp_mp3 = audio_path + "_ltx_tmp.mp3"
    ret = os.system(f'ffmpeg -y -i "{audio_path}" -acodec libmp3lame -q:a 2 "{tmp_mp3}" -loglevel quiet')
    if ret != 0:
        raise RuntimeError(f"ffmpeg conversion failed for {audio_path}")
    with open(tmp_mp3, "rb") as f:
        data = f.read()
    os.remove(tmp_mp3)
    return data


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


def ltx_post(endpoint: str, api_key: str, payload: dict) -> str:
    """POST to LTX API and return saved video path."""
    url = f"{LTX_BASE_URL}/{endpoint}"
    print(f"[LTX] → POST {endpoint} | payload keys: {list(payload.keys())}")
    r = requests.post(
        url,
        headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
        json=payload,
        timeout=300,
        stream=True,
    )
    if r.status_code != 200:
        raise RuntimeError(f"LTX API {r.status_code}: {r.text[:400]}")
    return download_video(r, endpoint.replace("-", "_"))


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
                "audio_path": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Local path to audio file (.mp3, .ogg, .wav). Will be converted to MP3.",
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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, image, audio_path, prompt,
                 model="ltx-2-3-pro", resolution="1080x1920",
                 duration=0, negative_prompt="blurry, distorted, low quality"):

        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        # 1. Upload image
        print("[LTX] Uploading image...")
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=1920)
        image_url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")

        # 2. Upload audio
        print("[LTX] Uploading audio...")
        audio_bytes = audio_path_to_mp3_bytes(audio_path)
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

        return (ltx_post("audio-to-video", api_key.strip(), payload),)


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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
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

        return (ltx_post("text-to-video", api_key.strip(), payload),)


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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
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

        return (ltx_post("image-to-video", api_key.strip(), payload),)


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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
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

        return (ltx_post("extend", api_key.strip(), payload),)


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

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
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

        return (ltx_post("retake", api_key.strip(), payload),)


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
