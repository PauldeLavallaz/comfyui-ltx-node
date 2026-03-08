"""
ComfyUI custom node for LTX-2 Video Generation API (ltx.video)
"""

import os
import requests
import tempfile
import folder_paths
import numpy as np
import torch

LTX_BASE_URL = "https://api.ltx.video/v1"


class LTXTextToVideo:
    """Generate video from text prompt using LTX-2 API"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "duration": ("INT", {"default": 8, "min": 6, "max": 20}),
                "resolution": (["1920x1080", "1080x1920", "1440x1080", "1080x1440", "4096x2160"], {"default": "1920x1080"}),
                "fps": (["24", "25", "48", "50"], {"default": "24"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, prompt, model, duration, resolution, fps):
        if not api_key:
            raise ValueError("LTX API Key requerida")

        payload = {
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "resolution": resolution,
        }

        print(f"[LTX] Generando video | model={model} | {resolution} | {duration}s")

        r = requests.post(
            f"{LTX_BASE_URL}/text-to-video",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
            stream=True
        )

        if r.status_code != 200:
            raise RuntimeError(f"LTX API error {r.status_code}: {r.text[:300]}")

        output_dir = folder_paths.get_output_directory()
        out_path = os.path.join(output_dir, f"ltx_t2v_{hash(prompt) % 100000}.mp4")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[LTX] Video guardado: {out_path}")
        return (out_path,)


class LTXImageToVideo:
    """Animate a static image using LTX-2 API"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "image_url": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "duration": ("INT", {"default": 8, "min": 6, "max": 20}),
                "resolution": (["1920x1080", "1080x1920", "1440x1080", "4096x2160"], {"default": "1920x1080"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, image_url, prompt, model, duration, resolution):
        if not api_key:
            raise ValueError("LTX API Key requerida")

        payload = {
            "image_uri": image_url,
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "resolution": resolution,
        }

        print(f"[LTX] Image-to-Video | model={model} | {resolution} | {duration}s")

        r = requests.post(
            f"{LTX_BASE_URL}/image-to-video",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
            stream=True
        )

        if r.status_code != 200:
            raise RuntimeError(f"LTX API error {r.status_code}: {r.text[:300]}")

        output_dir = folder_paths.get_output_directory()
        out_path = os.path.join(output_dir, f"ltx_i2v_{hash(image_url + prompt) % 100000}.mp4")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        print(f"[LTX] Video guardado: {out_path}")
        return (out_path,)


class LTXExtendVideo:
    """Extend an existing video using LTX-2 API"""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_key": ("STRING", {"default": "", "multiline": False}),
                "video_url": ("STRING", {"default": "", "multiline": False}),
                "prompt": ("STRING", {"default": "", "multiline": True}),
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "duration": ("INT", {"default": 6, "min": 2, "max": 20}),
                "direction": (["end", "beginning"], {"default": "end"}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("video_path",)
    FUNCTION = "extend"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def extend(self, api_key, video_url, prompt, model, duration, direction):
        payload = {
            "video_uri": video_url,
            "prompt": prompt,
            "model": model,
            "duration": duration,
            "direction": direction,
        }

        r = requests.post(
            f"{LTX_BASE_URL}/extend",
            headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"},
            json=payload,
            timeout=300,
            stream=True
        )

        if r.status_code != 200:
            raise RuntimeError(f"LTX API error {r.status_code}: {r.text[:300]}")

        output_dir = folder_paths.get_output_directory()
        out_path = os.path.join(output_dir, f"ltx_extend_{hash(video_url) % 100000}.mp4")
        with open(out_path, "wb") as f:
            for chunk in r.iter_content(chunk_size=8192):
                f.write(chunk)

        return (out_path,)


NODE_CLASS_MAPPINGS = {
    "LTXTextToVideo": LTXTextToVideo,
    "LTXImageToVideo": LTXImageToVideo,
    "LTXExtendVideo": LTXExtendVideo,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "LTXTextToVideo": "LTX Text to Video 🎬",
    "LTXImageToVideo": "LTX Image to Video 🖼️➡️🎬",
    "LTXExtendVideo": "LTX Extend Video ➕",
}

print("[LTX] Nodo LTX-2 Video API cargado ✅")
