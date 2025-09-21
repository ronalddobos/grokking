import torch
import torch.nn.functional as F
import numpy as np
import random
import json
import os
from datetime import datetime
from config import Config
from model import SimpleTransformer
from utils import check_for_checkpoint, save_checkpoint


def train_model(config: Config, resume: bool = True) -> None:
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
    checkpoint: dict[str, any] | None = None
    if resume:
        checkpoint = check_for_checkpoint(config)

    if checkpoint:
        print(f'[{config.tag}] Resuming from epoch {checkpoint["epoch"]}')
        start_epoch = checkpoint['epoch'] + 1
        train_data: list[tuple[int, int, int]] = checkpoint['train_data']
        test_data: list[tuple[int, int, int]] = checkpoint['test_data']

        # Load model and optimizer states
        model = SimpleTransformer(config).to(config.device)
        model.load_state_dict(checkpoint['model_state_dict'])

        optimizer = torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    else:
        # Fresh start
        print(f'[{config.tag}] Starting fresh training')

        # Save config with timestamps
        config_data = {
            **config.__dict__,
            'start_time': start_time.isoformat(),
        }
        config_path = os.path.join(config.log_dir, 'config.json')
        with open(config_path, 'w') as f:
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

    def compute_loss_and_acc(data: list[tuple[int, int, int]]) -> tuple[torch.Tensor, torch.Tensor]:
        x = torch.tensor(data).to(config.device)
        logits = model(x)[:, -1]
        labels = torch.tensor([config.fn(i, j) for i, j, _ in data]).to(config.device)
        loss = F.cross_entropy(logits, labels)
        acc = (logits.argmax(dim=-1) == labels).float().mean()
        return loss, acc

    # Initialize log file
    log_file = os.path.join(config.log_dir, 'training_log.jsonl')

    # Training loop
    for epoch in range(start_epoch, config.num_epochs):
        train_loss, train_acc = compute_loss_and_acc(train_data)

        with torch.no_grad():
            test_loss, test_acc = compute_loss_and_acc(test_data)

        # Log every 100 epochs
        if epoch % 100 == 0:
            log_entry = {
                'epoch': epoch,
                'train_loss': train_loss.item(),
                'test_loss': test_loss.item(),
                'train_acc': train_acc.item(),
                'test_acc': test_acc.item()
            }

            # Append to log file
            with open(log_file, 'a') as f:
                f.write(json.dumps(log_entry) + '\n')

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
        config_path = os.path.join(config.log_dir, 'config.json')
        with open(config_path, 'r') as f:
            config_data = json.load(f)
        config_data['finish_time'] = finish_time.isoformat()
        config_data['duration_seconds'] = (finish_time - start_time).total_seconds()
        with open(config_path, 'w') as f:
            json.dump(config_data, f, indent=2)

    print(f'[{config.tag}] Training completed')
    print(f'[{config.tag}] Logs saved to {config.log_dir}')
