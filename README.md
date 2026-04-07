# HydroSharkSampler for ComfyUI

A custom node package providing advanced sampling and scheduling for **Z-Image-Turbo** and other flow-based diffusion models. Fully compatible with **RES4LYF ClownsharkSampler** pipelines.

## Features
- **Perfect for Turbo Models:** Designed specifically for rectified-flow models at CFG 1.0.
- **Vibrant Colors:** Uses `cfg_rescale` to prevent over-saturation and blown-out highlights.
- **High Detail:** Adaptive scheduling ensures fine details aren't blurred out when generating images in just 8-9 steps.
- **All-in-One Node:** Includes a convenient KSampler node, or modular nodes for advanced workflows.

## Installation

### Method 1: ComfyUI Manager (Recommended)
You can install this node via the **ComfyUI Manager**:
1. Open ComfyUI and click **Manager**.
2. Click **Install via Git URL** (or **Install Custom Nodes** and search for `HydroSharkSampler`).
3. If using URL, paste: `https://github.com/MONKEYFOREVER2/HydroSharkSampler`
4. Restart ComfyUI.

### Method 2: Manual Installation (Git Clone)
1. Open a terminal in your `ComfyUI/custom_nodes/` folder:
   ```bash
   cd ComfyUI/custom_nodes
   git clone https://github.com/MONKEYFOREVER2/HydroSharkSampler.git
   ```
2. Restart ComfyUI.

*(Note: There are no extra python packages required!)*

## Nodes Included

You can find these under the **HydroShark** category in your ComfyUI node menu:
1. **HydroShark KSampler**: An all-in-one convenient sampler. Use this if you want a plug-and-play experience.
2. **HydroShark Scheduler**: Generates a custom schedule for use with standard `SamplerCustom`.
3. **HydroShark Sampler**: The core sampler for use with standard `SamplerCustom`.

## License
MIT License. Free for personal and commercial use.
