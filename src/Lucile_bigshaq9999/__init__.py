import os
import sys

# Auto-configure HSA override for AMD RDNA3 / Hawk Point / Phoenix APUs (gfx1103) if on Linux
if sys.platform == "linux" and "HSA_OVERRIDE_GFX_VERSION" not in os.environ:
    if os.path.exists("/dev/kfd"):
        os.environ["HSA_OVERRIDE_GFX_VERSION"] = "11.0.0"

try:
    import torch

    # Disable Flash/Mem-Efficient SDP on ROCm/RDNA3 to prevent GPU hang during transformer attention
    if torch.cuda.is_available():
        torch.backends.cuda.enable_flash_sdp(False)
        torch.backends.cuda.enable_mem_efficient_sdp(False)
        torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass
