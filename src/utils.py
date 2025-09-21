import torch
import torch.nn as nn
import torch.optim as optim
import os
from config import Config


def check_for_checkpoint(config: Config) -> dict[str, any] | None:
    """Check if a checkpoint exists and return checkpoint info."""
    checkpoint_path = os.path.join(config.model_dir, 'checkpoint.pth')
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location='cpu')
    return None


def save_checkpoint(config: Config, model: nn.Module, optimizer: optim.Optimizer,
                   epoch: int, train_data: list[tuple[int, int, int]],
                   test_data: list[tuple[int, int, int]]) -> None:
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_data': train_data,
        'test_data': test_data,
        'config': config.__dict__
    }
    checkpoint_path = os.path.join(config.model_dir, 'checkpoint.pth')
    torch.save(checkpoint, checkpoint_path)
