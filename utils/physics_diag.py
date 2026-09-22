import torch

def check_wave_stability(model):
    """Checks the learnable wave parameters for potential instability."""
    for name, param in model.named_parameters():
        if 'alpha' in name:
            # Damping should stay positive to prevent energy explosion
            if torch.any(param < 0):
                print(f"⚠️ Warning: Negative damping detected in {name}")
        if 'omega' in name:
            # Extremely high frequencies can cause aliasing in FFT
            if torch.any(param > 100):
                print(f"⚠️ Warning: Very high frequency in {name}")

def get_vram_usage():
    """Returns current VRAM usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / (1024 ** 2)
    return 0