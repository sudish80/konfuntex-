import os
import sys

class EnvironmentDetector:
    @staticmethod
    def get_env_type():
        if os.environ.get("COLAB_GPU"):
            return "colab"
        # Check for local GPU via torch
        try:
            import torch
            if torch.cuda.is_available():
                return "local_gpu"
        except ImportError:
            pass
        return "local_cpu"

    @staticmethod
    def get_vram_gb():
        # Implementation to query actual GPU VRAM
        try:
            import torch
            if torch.cuda.is_available():
                # Get VRAM in GB
                return torch.cuda.get_device_properties(0).total_memory // (1024**3)
        except:
            pass
        return 4  # Default low-resource fallback
