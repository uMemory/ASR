from .config import load_config, get_model_config, project_root
from .logger import get_logger
from .cuda_dlls import add_cuda_dll_dirs

# Register NVIDIA DLL dirs eagerly so any `import faster_whisper` /
# `import whisperx` later in the process succeeds on Windows.
add_cuda_dll_dirs()

__all__ = [
    "load_config", "get_model_config", "project_root",
    "get_logger", "add_cuda_dll_dirs",
]
