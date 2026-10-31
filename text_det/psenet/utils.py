"""Small helpers shared by LearnOCR detectors (PSENet, DBNet, ...)."""
import torch


def choose_device(name="auto"):
    """Prefer CUDA, then Apple MPS, then CPU; reject unavailable devices."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if device.type == "mps" and not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()):
        raise RuntimeError("MPS requested but unavailable")
    return device
