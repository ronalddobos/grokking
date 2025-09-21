import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import matplotlib.pyplot as plt
import json
import os
from datetime import datetime
from dataclasses import dataclass


@dataclass
class Config:
    tag: str
    p: int = 113
    d_model: int = 128
    d_mlp: int = 512
    num_heads: int = 4
    lr: float = 1e-3
    weight_decay: float = 1.0
    frac_train: float = 0.3
    num_epochs: int = 50000
    seed: int = 0
    fn_name: str = 'add'
    device: str = 'cuda'

    @property
    def fn(self):
        return {'add': lambda x, y: (x + y) % self.p}[self.fn_name]

    @property
    def log_dir(self):
        return os.path.join("logs", self.tag)

    @property
    def model_dir(self):
        return os.path.join("models", self.tag)


class SimpleTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Embedding
        self.embed = nn.Parameter(torch.randn(config.d_model, config.p + 1) / np.sqrt(config.d_model))
        self.pos_embed = nn.Parameter(torch.randn(3, config.d_model) / np.sqrt(config.d_model))

        # Attention
        d_head = config.d_model // config.num_heads
        self.W_Q = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_K = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_V = nn.Parameter(torch.randn(config.num_heads, d_head, config.d_model) / np.sqrt(config.d_model))
        self.W_O = nn.Parameter(torch.randn(config.d_model, config.d_model) / np.sqrt(config.d_model))

        # MLP
        self.W_in = nn.Parameter(torch.randn(config.d_mlp, config.d_model) / np.sqrt(config.d_model))
        self.W_out = nn.Parameter(torch.randn(config.d_model, config.d_mlp) / np.sqrt(config.d_model))
        self.b_in = nn.Parameter(torch.zeros(config.d_mlp))
        self.b_out = nn.Parameter(torch.zeros(config.d_model))

        # Unembed
        self.W_U = nn.Parameter(torch.randn(config.d_model, config.p + 1) / np.sqrt(config.p + 1))

        self.register_buffer('mask', torch.tril(torch.ones(3, 3)))

    def forward(self, x):
        # Embed + pos
        x = torch.einsum('dbp -> bpd', self.embed[:, x]) + self.pos_embed

        # Attention
        q = torch.einsum('ihd,bpd->biph', self.W_Q, x)
        k = torch.einsum('ihd,bpd->biph', self.W_K, x)
        v = torch.einsum('ihd,bpd->biph', self.W_V, x)

        scores = torch.einsum('biph,biqh->biqp', k, q) / np.sqrt(self.config.d_model // self.config.num_heads)
        scores = scores.masked_fill(self.mask == 0, -1e10)
        attn = F.softmax(scores, dim=-1)

        z = torch.einsum('biph,biqp->biqh', v, attn)
        z = z.reshape(x.shape[0], x.shape[1], -1)
        attn_out = torch.einsum('df,bqf->bqd', self.W_O, z)

        # Residual
        x = x + attn_out

        # MLP
        mlp_out = F.relu(torch.einsum('md,bpd->bpm', self.W_in, x) + self.b_in)
        mlp_out = torch.einsum('dm,bpm->bpd', self.W_out, mlp_out) + self.b_out

        # Residual
        x = x + mlp_out

        # Unembed
        return x @ self.W_U


def check_for_checkpoint(config: Config):
    """Check if a checkpoint exists and return checkpoint info."""
    checkpoint_path = os.path.join(config.model_dir, "checkpoint.pth")
    if os.path.exists(checkpoint_path):
        return torch.load(checkpoint_path, map_location='cpu')
    return None


def save_checkpoint(config: Config, model, optimizer, epoch, train_data, test_data):
    """Save training checkpoint."""
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'epoch': epoch,
        'train_data': train_data,
        'test_data': test_data,
        'config': config.__dict__
    }
    checkpoint_path = os.path.join(config.model_dir, "checkpoint.pth")
    torch.save(checkpoint, checkpoint_path)


def train_model(config, resume=True):
    """Train the grokking model and save logs continuously."""
    start_time = datetime.now()
    start_epoch = 0

    # Set seeds for reproducibility
    torch.manual_seed(config.seed)
    np.random.seed(config.seed)
    random.seed(config.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.seed)

    # Create save directory structure
    os.makedirs(config.log_dir, exist_ok=True)
    os.makedirs(config.model_dir, exist_ok=True)

    # Check for existing checkpoint
    checkpoint = None
    if resume:
        checkpoint = check_for_checkpoint(config)

    if checkpoint:
        print(f"[{config.tag}] Resuming from epoch {checkpoint['epoch']}")
        start_epoch = checkpoint['epoch'] + 1
        train_data = checkpoint['train_data']
        test_data = checkpoint['test_data']

        # Load model and optimizer states
        model = SimpleTransformer(config).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # Fresh start
        print(f"[{config.tag}] Starting fresh training")

        # Save config with timestamps
        config_data = {
            **config.__dict__,
            "start_time": start_time.isoformat(),
        }
        config_path = os.path.join(config.log_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

        # Data
        pairs = [(i, j, config.p) for i in range(config.p) for j in range(config.p)]
        random.seed(config.seed)
        random.shuffle(pairs)
        split = int(config.frac_train * len(pairs))
        train_data, test_data = pairs[:split], pairs[split:]

        # Model
        model = SimpleTransformer(config).to(config.device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)

    def compute_loss_and_acc(data):
        x = torch.tensor(data).to(config.device)
        logits = model(x)[:, -1]
        labels = torch.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

    # Initialize log file
    log_file = os.path.join(config.log_dir, "training_log.jsonl")

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        train_loss, train_acc = compute_loss_and_acc(train_data)

        with torch.no_grad():
            test_loss, test_acc = compute_loss_and_acc(test_data)

        # Log every 100 epochs
        if epoch % 100 == 0:
            log_entry = {
                "epoch": epoch,
                "train_loss": train_loss.item(),
                "test_loss": test_loss.item(),
                "train_acc": train_acc.item(),
                "test_acc": test_acc.item()
            }

            # Append to log file
            with open(log_file, "a") as f:
                f.write(json.dumps(log_entry) + "\n")

            if epoch % 1000 == 0:
                print(f'[{config.tag}] Epoch {epoch}: train_loss={train_loss:.3f}, test_loss={test_loss:.3f}, '
                      f'train_acc={train_acc:.3f}, test_acc={test_acc:.3f}')

        # Save checkpoint every 1000 epochs
        if epoch % 1000 == 0:
            save_checkpoint(config, model, optimizer, epoch, train_data, test_data)

        train_loss.backward()
        optimizer.step()
        optimizer.zero_grad()

    # Save final checkpoint at the end
    save_checkpoint(config, model, optimizer, config.num_epochs - 1, train_data, test_data)

    # Record finish time (only update if we started fresh)
    if not checkpoint:
        finish_time = datetime.now()
        config_path = os.path.join(config.log_dir, "config.json")
        with open(config_path, "r") as f:
            config_data = json.load(f)
        config_data["finish_time"] = finish_time.isoformat()
        config_data["duration_seconds"] = (finish_time - start_time).total_seconds()
        with open(config_path, "w") as f:
            json.dump(config_data, f, indent=2)

    print(f"[{config.tag}] Training completed")
    print(f"[{config.tag}] Logs saved to {config.log_dir}")


def plot_training_curves(config: Config, show_plot=True):
    """Load logs and create training curve plots."""
    log_file = os.path.join(config.log_dir, "training_log.jsonl")

    # Load all logs
    logs = []
    with open(log_file, "r") as f:
        for line in f:
            logs.append(json.loads(line.strip()))

    epochs = [log["epoch"] for log in logs]
    train_losses = [log["train_loss"] for log in logs]
    test_losses = [log["test_loss"] for log in logs]
    train_accs = [log["train_acc"] for log in logs]
    test_accs = [log["test_acc"] for log in logs]

    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Loss plot (log scale)
    ax1.plot(epochs, train_losses, label='Train Loss', alpha=0.8)
    ax1.plot(epochs, test_losses, label='Test Loss', alpha=0.8)
    ax1.set_yscale('log')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss (log scale)')
    ax1.set_title('Training Progress - Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Accuracy plot
    ax2.plot(epochs, train_accs, label='Train Accuracy', alpha=0.8)
    ax2.plot(epochs, test_accs, label='Test Accuracy', alpha=0.8)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training Progress - Accuracy')
    ax2.set_ylim(0, 1)
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save figure
    fig_path = os.path.join(config.log_dir, "training_curves.png")
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f"Figure saved to {fig_path}")

    if show_plot:
        plt.show()
    else:
        plt.close()


def train_grokking(config: Config, resume=True):
    """Complete training pipeline with plotting."""
    train_model(config, resume=resume)
    plot_training_curves(config)


if __name__ == "__main__":
    config = Config(tag="P_113", num_epochs=10_000, p=113)
    train_grokking(config, resume=True)