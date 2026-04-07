# HydroSharkSampler - ComfyUI Custom Node Package
# Optimized for Z-Image-Turbo (rectified-flow/CONST) and RES4LYF ClownsharkSampler pipelines

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]

print("\033[92m[HydroSharkSampler] Loaded: HydroSharkScheduler, HydroSharkSampler, HydroSharkKSampler\033[0m")
