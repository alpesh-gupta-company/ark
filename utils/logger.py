import time

class SimpleLogger:
    def __init__(self, use_wandb=False):
        self.use_wandb = use_wandb
        self.start_time = time.time()

    def log_metrics(self, epoch, loss, vram):
        elapsed = time.time() - self.start_time
        print(f"[Epoch {epoch}] Loss: {loss:.4f} | VRAM: {vram:.1f}MB | Time: {elapsed:.1f}s")
        
        if self.use_wandb:
            import wandb
            wandb.log({"loss": loss, "vram_mb": vram, "epoch": epoch})