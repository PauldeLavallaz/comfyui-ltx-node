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

# ─── VIDEO type helper (native ComfyUI compat) ───────────────────────────────
def _make_video_output(path: str):
    """Wrap video path as ComfyUI VIDEO type object for SaveVideo compatibility.

    ComfyUI's SaveVideo node calls video.get_dimensions() and video.save_to(),
    so we must return a VideoFromFile object, not a plain string.
    """
    try:
        from comfy_api.input_impl import VideoFromFile
        return VideoFromFile(path)
    except ImportError:
        pass
    try:
        from comfy_api.latest._input_impl import VideoFromFile
        return VideoFromFile(path)
    except (ImportError, AttributeError):
        pass
    return path


LTX_BASE_URL = "https://api.ltx.video/v1"
UGUU_UPLOAD_URL = "https://uguu.se/upload"


# ─────────────────────────────────────────────────────────────────────────────
# UPLOAD HELPERS
# ─────────────────────────────────────────────────────────────────────────────

def upload_to_uguu(data: bytes, filename: str, content_type: str) -> str:
    """Upload bytes and return public HTTPS URL with correct Content-Type.

    Strategy for audio:
      1. catbox.moe (permanent, high-availability, no hotlink restrictions)
      2. litterbox.catbox.moe (72h temp, fallback)
      3. uguu.se (fallback for images)

    The LTX API validates Content-Type headers; 0x0.st and uguu.se sometimes
    block or throttle external fetches from LTX servers (causes HTTP 400/500).
    catbox.moe is widely whitelisted and returns direct URLs with correct MIME.
    """
    is_audio = content_type.startswith("audio/")

    if is_audio:
        # 1. catbox.moe — permanent, no hotlink restrictions, widely accessible
        try:
            r = requests.post(
                "https://catbox.moe/user/api.php",
                data={"reqtype": "fileupload", "userhash": ""},
                files={"fileToUpload": (filename, data, content_type)},
                timeout=60,
            )
            r.raise_for_status()
            url = r.text.strip()
            if url.startswith("https://"):
                print(f"[LTX] Uploaded (catbox.moe) {filename} → {url}")
                return url
        except Exception as e:
            print(f"[LTX] catbox.moe failed ({e}), trying litterbox...")

        # 2. litterbox.catbox.moe — 72h temporary
        try:
            r = requests.post(
                "https://litterbox.catbox.moe/resources/internals/api.php",
                data={"reqtype": "fileupload", "time": "72h"},
                files={"fileToUpload": (filename, data, content_type)},
                timeout=60,
            )
            r.raise_for_status()
            url = r.text.strip()
            if url.startswith("https://"):
                print(f"[LTX] Uploaded (litterbox) {filename} → {url}")
                return url
        except Exception as e:
            print(f"[LTX] litterbox failed ({e}), falling back to uguu.se...")

    # Fallback / images: uguu.se
    files = {"files[]": (filename, data, content_type)}
    r = requests.post(UGUU_UPLOAD_URL, files=files, timeout=60)
    r.raise_for_status()
    j = r.json()
    url = j["files"][0]["url"]
    print(f"[LTX] Uploaded (uguu.se) {filename} → {url}")
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


def _find_ffmpeg() -> str | None:
    """Find ffmpeg binary in common locations."""
    import shutil
    if shutil.which("ffmpeg"):
        return "ffmpeg"
    for p in [
        "/usr/bin/ffmpeg", "/usr/local/bin/ffmpeg", "/opt/conda/bin/ffmpeg",
        "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
        "/usr/local/opt/ffmpeg/bin/ffmpeg",  # Intel Homebrew
    ]:
        if os.path.exists(p):
            return p
    return None


def _ffmpeg_to_mp3(raw_bytes: bytes, in_fmt: str = None) -> bytes | None:
    """Convert arbitrary audio bytes to MP3 via ffmpeg. Returns None if ffmpeg missing."""
    ffmpeg = _find_ffmpeg()
    if not ffmpeg:
        return None
    try:
        cmd = [ffmpeg, "-y"]
        if in_fmt:
            cmd += ["-f", in_fmt]
        cmd += ["-i", "pipe:0", "-vn", "-ar", "44100", "-ac", "2",
                "-b:a", "192k", "-f", "mp3", "pipe:1"]
        result = subprocess.run(cmd, input=raw_bytes, capture_output=True, timeout=30)
        if result.returncode == 0 and len(result.stdout) > 0:
            print(f"[LTX] ffmpeg → MP3: {len(result.stdout)/1024:.1f} KB")
            return result.stdout
    except Exception as e:
        print(f"[LTX] ffmpeg conversion failed: {e}")
    return None


def audio_tensor_to_bytes(audio: dict) -> tuple:
    """
    Convert ComfyUI AUDIO dict {'waveform': Tensor[B,C,T], 'sample_rate': int}
    to (bytes, mime_type, filename).

    Pipeline (any input format → MP3):
      1. ffmpeg from raw PCM → MP3  (most reliable, handles any sample rate/channels)
      2. torchaudio → MP3
      3. torchaudio → WAV
      4. Python wave module → WAV   (pure stdlib, zero deps)
    Returns (bytes, content_type, filename).
    """
    waveform = audio["waveform"]   # [B, C, T] float32 in [-1, 1]
    sample_rate = audio["sample_rate"]

    if waveform.ndim == 3:
        waveform = waveform[0]     # [C, T]
    if waveform.shape[0] > 2:
        waveform = waveform[:2]    # max stereo

    channels = waveform.shape[0]
    samples = waveform.shape[1]

    # ── 1. ffmpeg: PCM s16le → MP3 ──────────────────────────────────────────
    try:
        pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
        pcm_bytes = pcm.T.flatten().tobytes()
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            cmd = [ffmpeg, "-y",
                   "-f", "s16le", "-ar", str(sample_rate), "-ac", str(channels),
                   "-i", "pipe:0",
                   "-vn", "-ar", "44100", "-ac", "2", "-b:a", "192k",
                   "-f", "mp3", "pipe:1"]
            result = subprocess.run(cmd, input=pcm_bytes, capture_output=True, timeout=30)
            if result.returncode == 0 and len(result.stdout) > 0:
                print(f"[LTX] Audio via ffmpeg MP3: {len(result.stdout)/1024:.1f} KB")
                return result.stdout, "audio/mpeg", "ltx_audio.mp3"
    except Exception as e:
        print(f"[LTX] ffmpeg PCM→MP3 failed ({e}), trying torchaudio...")

    # ── 2. torchaudio MP3 ────────────────────────────────────────────────────
    try:
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, waveform.cpu(), sample_rate, format="mp3")
        data = buf.getvalue()
        print(f"[LTX] Audio via torchaudio MP3: {len(data)/1024:.1f} KB")
        return data, "audio/mpeg", "ltx_audio.mp3"
    except Exception as e:
        print(f"[LTX] torchaudio MP3 failed ({e}), trying WAV...")

    # ── 3. torchaudio WAV ────────────────────────────────────────────────────
    try:
        import torchaudio
        buf = io.BytesIO()
        torchaudio.save(buf, waveform.cpu(), sample_rate, format="wav")
        data = buf.getvalue()
        print(f"[LTX] Audio via torchaudio WAV: {len(data)/1024:.1f} KB")
        return data, "audio/wav", "ltx_audio.wav"
    except Exception as e:
        print(f"[LTX] torchaudio WAV failed ({e}), trying wave module...")

    # ── 4. Pure-Python wave (stdlib, zero deps) ──────────────────────────────
    import wave as wavemod
    pcm = (waveform.cpu().numpy() * 32767).clip(-32768, 32767).astype("int16")
    pcm_bytes = pcm.T.flatten().tobytes()
    buf = io.BytesIO()
    with wavemod.open(buf, "wb") as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    data = buf.getvalue()
    print(f"[LTX] Audio via wave module WAV: {len(data)/1024:.1f} KB")
    return data, "audio/wav", "ltx_audio.wav"


# Keep old name as alias for compatibility
def audio_tensor_to_mp3_bytes(audio: dict) -> bytes:
    data, _, __ = audio_tensor_to_bytes(audio)
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


def video_bytes_to_image_tensor(video_bytes: bytes) -> torch.Tensor:
    """
    Decode video bytes → ComfyUI IMAGE tensor [F, H, W, C] float32 in [0,1].
    Tries torchvision → cv2 → ffmpeg (in that order).
    """
    # Write to temp file (all decoders need a seekable file)
    tmp = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    tmp.write(video_bytes)
    tmp.close()
    tmp_path = tmp.name

    try:
        # 1. Try torchvision (always available in ComfyUI)
        try:
            import torchvision.io as tvio
            frames, _, _ = tvio.read_video(tmp_path, pts_unit="sec", output_format="THWC")
            # frames: [T, H, W, C] uint8
            result = frames.float() / 255.0
            print(f"[LTX] Decoded {len(result)} frames via torchvision")
            return result
        except Exception as e:
            print(f"[LTX] torchvision failed ({e}), trying cv2...")

        # 2. Try cv2
        try:
            import cv2
            cap = cv2.VideoCapture(tmp_path)
            frames = []
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            cap.release()
            if frames:
                result = torch.from_numpy(np.stack(frames).astype(np.float32) / 255.0)
                print(f"[LTX] Decoded {len(result)} frames via cv2")
                return result
        except Exception as e:
            print(f"[LTX] cv2 failed ({e}), trying ffmpeg...")

        # 3. Try ffmpeg (last resort)
        ffmpeg = _find_ffmpeg()
        if ffmpeg:
            cmd = [ffmpeg, "-i", tmp_path, "-f", "image2pipe", "-vcodec", "png", "-", "-loglevel", "error"]
            proc = subprocess.run(cmd, capture_output=True)
            if proc.returncode == 0:
                raw = proc.stdout
                frames = []
                i = 0
                while i < len(raw):
                    start = raw.find(b'\x89PNG\r\n\x1a\n', i)
                    if start == -1:
                        break
                    end = raw.find(b'\x89PNG\r\n\x1a\n', start + 8)
                    chunk = raw[start:end] if end != -1 else raw[start:]
                    img = Image.open(io.BytesIO(chunk)).convert("RGB")
                    frames.append(np.array(img, dtype=np.float32) / 255.0)
                    i = end if end != -1 else len(raw)
                if frames:
                    print(f"[LTX] Decoded {len(frames)} frames via ffmpeg")
                    return torch.from_numpy(np.stack(frames))

        raise RuntimeError("No video decoder available. Install torchvision, cv2, or ffmpeg.")
    finally:
        os.unlink(tmp_path)


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

    # Build ComfyUI UI dict for native video preview
    if COMFY_AVAILABLE:
        out_dir_base = folder_paths.get_output_directory()
        rel = os.path.relpath(out_path, out_dir_base)
        ui_dict = {"videos": [{"filename": os.path.basename(rel),
                                "subfolder": os.path.dirname(rel),
                                "type": "output"}]}
    else:
        ui_dict = {}

    video_obj = _make_video_output(out_path)
    return frames, video_obj, ui_dict


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Audio to Video (Lip-sync)  ← MAIN NEW NODE
# ─────────────────────────────────────────────────────────────────────────────

class LTXAudioToVideo:
    """
    LTX Audio-to-Video: lip-sync video from image + audio.
    Endpoint: POST https://api.ltx.video/v1/audio-to-video
    Returns 25fps video synchronised to audio.
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
                "model": (["ltx-2-3-pro", "ltx-2-pro"], {"default": "ltx-2-3-pro"}),
                "resolution": (["1080x1920", "1920x1080"], {
                    "default": "1080x1920",
                    "tooltip": "Portrait or landscape. Auto-detected from image if omitted.",
                }),
                "guidance_scale": ("FLOAT", {
                    "default": 7.5,
                    "min": 1.0,
                    "max": 20.0,
                    "step": 0.5,
                    "tooltip": "CFG scale. Higher = follows prompt more strictly. Default 9 when image provided.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps")
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, api_key, image, audio, prompt,
                 model="ltx-2-3-pro", resolution="1080x1920", guidance_scale=7.5):

        if not api_key.strip():
            raise ValueError("LTX API key is required.")

        # 1. Upload image to catbox.moe
        print("[LTX] Uploading image...")
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=1920)
        image_url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")

        # 2. Convert AUDIO tensor → MP3 and upload to catbox.moe
        print("[LTX] Encoding + uploading audio...")
        audio_bytes, audio_mime, audio_filename = audio_tensor_to_bytes(audio)
        audio_url = upload_to_uguu(audio_bytes, audio_filename, audio_mime)

        # 3. Call API
        payload = {
            "audio_uri": audio_url,
            "image_uri": image_url,
            "prompt": prompt,
            "model": model,
            "resolution": resolution,
            "guidance_scale": guidance_scale,
        }

        print(f"[LTX] audio-to-video | model={model} res={resolution} cfg={guidance_scale}")
        frames, video_path, ui = ltx_post("audio-to-video", api_key.strip(), payload)
        return {"ui": ui, "result": (frames, video_path, 25.0)}


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

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps")
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

        frames, video_path, ui = ltx_post("text-to-video", api_key.strip(), payload)
        return {"ui": ui, "result": (frames, video_path, 25.0)}


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

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps")
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

        frames, video_path, ui = ltx_post("image-to-video", api_key.strip(), payload)
        return {"ui": ui, "result": (frames, video_path, 25.0)}


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

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps")
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

        frames, video_path, ui = ltx_post("extend", api_key.strip(), payload)
        return {"ui": ui, "result": (frames, video_path, 25.0)}


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

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps")
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

        frames, video_path, ui = ltx_post("retake", api_key.strip(), payload)
        return {"ui": ui, "result": (frames, video_path, 25.0)}


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


# ─────────────────────────────────────────────────────────────────────────────
# NODE: Audio to Video via REPLICATE (fixed version)
# ─────────────────────────────────────────────────────────────────────────────

class LTXAudioToVideoReplicate:
    """
    LTX-2.3 Audio-to-Video via Replicate API.
    Properly uploads image + audio before calling Replicate (fixes base64 bug).
    Returns frames + video_path + fps, AND shows video preview in ComfyUI UI.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "replicate_api_key": ("STRING", {
                    "default": "",
                    "multiline": False,
                    "tooltip": "Replicate API token (r8_...)",
                }),
                "image": ("IMAGE",),
                "audio": ("AUDIO",),
                "prompt": ("STRING", {
                    "default": "A person speaking naturally, slight head movement, realistic lip sync.",
                    "multiline": True,
                }),
            },
            "optional": {
                "model": (["ltx-2-3-pro", "ltx-2-3-fast"], {"default": "ltx-2-3-pro"}),
                "aspect_ratio": (["9:16", "16:9", "1:1"], {"default": "9:16"}),
                "fps": ("INT", {"default": 25, "min": 24, "max": 50}),
                "seed": ("INT", {"default": -1, "min": -1, "max": 2147483647,
                                 "tooltip": "-1 for random."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "VIDEO", "FLOAT")
    RETURN_NAMES = ("frames", "video", "fps_out")
    FUNCTION = "generate"
    CATEGORY = "LTX Video"
    OUTPUT_NODE = True

    def generate(self, replicate_api_key, image, audio, prompt,
                 model="ltx-2-3-pro", aspect_ratio="9:16", fps=25, seed=-1):

        if not replicate_api_key.strip():
            raise ValueError("Replicate API key is required.")

        api_key = replicate_api_key.strip()

        # 1. Upload image to catbox.moe (public URL)
        print("[LTX-Replicate] Uploading image...")
        img_bytes = tensor_to_jpeg_bytes(image, max_dim=1920)
        image_url = upload_to_uguu(img_bytes, "ltx_image.jpg", "image/jpeg")

        # 2. Upload audio to catbox.moe (public URL)
        print("[LTX-Replicate] Encoding + uploading audio...")
        audio_bytes, audio_mime, audio_filename = audio_tensor_to_bytes(audio)
        audio_url = upload_to_uguu(audio_bytes, audio_filename, audio_mime)

        # 3. Build Replicate payload
        model_id = "lightricks/ltx-2.3-pro" if model == "ltx-2-3-pro" else "lightricks/ltx-2.3-fast"
        payload = {
            "input": {
                "task": "audio_to_video",
                "image": image_url,
                "audio": audio_url,
                "prompt": prompt,
                "resolution": "1080p",
                "aspect_ratio": aspect_ratio,
                "fps": fps,
            }
        }
        if seed >= 0:
            payload["input"]["seed"] = seed

        print(f"[LTX-Replicate] → POST {model_id} | image: {image_url[:60]}...")

        # 4. Submit prediction
        r = requests.post(
            f"https://api.replicate.com/v1/models/{model_id}/predictions",
            headers={
                "Authorization": f"Token {api_key}",
                "Content-Type": "application/json",
                "Prefer": "wait",
            },
            json=payload,
            timeout=300,
        )
        if r.status_code not in (200, 201):
            raise RuntimeError(f"Replicate API {r.status_code}: {r.text[:400]}")

        result = r.json()

        # 5. Poll if not done yet
        if result.get("status") not in ("succeeded", "failed"):
            pred_id = result["id"]
            print(f"[LTX-Replicate] Polling prediction {pred_id}...")
            for _ in range(120):  # max 4 min polling
                time.sleep(2)
                poll = requests.get(
                    f"https://api.replicate.com/v1/predictions/{pred_id}",
                    headers={"Authorization": f"Token {api_key}"},
                    timeout=30,
                )
                result = poll.json()
                status = result.get("status")
                print(f"[LTX-Replicate] Status: {status}")
                if status == "succeeded":
                    break
                if status == "failed":
                    raise RuntimeError(f"Replicate prediction failed: {result.get('error')}")

        if result.get("status") != "succeeded":
            raise RuntimeError(f"Replicate prediction timed out or failed: {result.get('error')}")

        output_url = result["output"]
        if isinstance(output_url, list):
            output_url = output_url[0]

        print(f"[LTX-Replicate] Output URL: {output_url}")

        # 6. Download video
        vid_r = requests.get(output_url, timeout=120)
        vid_r.raise_for_status()
        video_bytes = vid_r.content

        out_path = get_output_path("replicate_a2v")
        with open(out_path, "wb") as f:
            f.write(video_bytes)
        print(f"[LTX-Replicate] Video saved: {out_path}")

        # 7. Decode to frames
        frames = video_bytes_to_image_tensor(video_bytes)

        # 8. UI preview (shows video directly in ComfyUI without external nodes)
        # ComfyUI's output folder relative path for UI preview
        if COMFY_AVAILABLE:
            import folder_paths as fp
            out_dir = fp.get_output_directory()
            rel_path = os.path.relpath(out_path, out_dir)
            subfolder = os.path.dirname(rel_path)
            filename = os.path.basename(rel_path)
            ui_videos = [{"filename": filename, "subfolder": subfolder, "type": "output"}]
        else:
            ui_videos = []

        return {"ui": {"videos": ui_videos}, "result": (frames, out_path, float(fps))}


NODE_CLASS_MAPPINGS["LTXAudioToVideoReplicate"] = LTXAudioToVideoReplicate
NODE_DISPLAY_NAME_MAPPINGS["LTXAudioToVideoReplicate"] = "LTX Audio to Video (Replicate Fixed) 🎤✅"
