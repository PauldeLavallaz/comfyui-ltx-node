# ComfyUI LTX-2 Video API Node

Custom ComfyUI nodes for [LTX-2](https://ltx.io) video generation API by Lightricks.

## Features

- 🎬 **Text to Video** — generate video from a text prompt
- 🖼️➡️🎬 **Image to Video** — animate a static image
- ➕ **Extend Video** — extend an existing video from start or end

## Installation

1. Clone into your ComfyUI custom_nodes folder:
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/PauldeLavallaz/comfyui-ltx-node
```

2. Restart ComfyUI

3. Nodes appear under **LTX Video** category

## Usage

Get your API key at [console.ltx.video](https://console.ltx.video) and paste it into the `api_key` field of each node.

### Models
| Model | Speed | Quality |
|-------|-------|---------|
| `ltx-2-3-fast` | Fast | Good |
| `ltx-2-3-pro` | Slower | Best |

### Supported Resolutions
- 1920x1080, 1080x1920, 1440x1080, 4096x2160

### Duration
- 6–20 seconds per request

## Nodes

### LTX Text to Video
Generate video from a text description.

### LTX Image to Video  
Animate a static image with a motion prompt. Requires a public image URL.

### LTX Extend Video
Extend an existing video from the beginning or end.
