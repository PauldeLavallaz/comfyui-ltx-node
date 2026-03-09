# comfyui-ltx-node 🎬

ComfyUI custom nodes for **LTX-2.3 Pro** video generation API ([ltx.video](https://ltx.video)).

## Nodes

| Node | Input | Output |
|------|-------|--------|
| 🎤 **LTX Audio to Video** | IMAGE + audio path + prompt | lip-sync video |
| 📝 **LTX Text to Video** | prompt | video |
| 🖼️ **LTX Image to Video** | IMAGE + prompt | animated video |
| ➕ **LTX Extend Video** | video URL + prompt | extended video |
| 🔁 **LTX Retake Section** | video URL + frame range | retaken video |
| ☁️ **LTX Image Uploader** | IMAGE | HTTPS URL |

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PauldeLavallaz/comfyui-ltx-node
pip install requests pillow
```

## Usage

1. Get your API key from [ltx.video/api-keys](https://ltx.video/api-keys)
2. Add any **LTX** node to your workflow
3. Paste your API key in the `api_key` field
4. Connect inputs and run

## Audio to Video (Lip-sync)

The main node. Accepts:

- **image** — ComfyUI IMAGE (auto-resized to max 1920px, uploaded automatically)
- **audio_path** — local path to `.mp3`, `.ogg`, or `.wav` (auto-converted to MP3)
- **prompt** — describe the motion/scene
- **api_key** — your LTX API key
- **model** — `ltx-2-3-pro` (best) or `ltx-2-3-fast` (quick tests)
- **resolution** — `1080x1920` (portrait) or `1920x1080` (landscape)
- **duration** — seconds (0 = auto-match audio length)
- **negative_prompt** — what to avoid

> ⚠️ Audio-to-video only supports `1080x1920` or `1920x1080`.  
> Images larger than 1920px are auto-resized. OGG/WAV files are auto-converted to MP3.

## Notes

- All uploads go through [uguu.se](https://uguu.se) (temporary HTTPS hosting required by LTX API)
- Videos are saved to ComfyUI's `output/` directory
- `ffmpeg` required for OGG/WAV → MP3 conversion

## Models

| Model | Speed | Quality |
|-------|-------|---------|
| `ltx-2-3-pro` | ~30-60s | Best |
| `ltx-2-3-fast` | ~17s | Good for testing |
