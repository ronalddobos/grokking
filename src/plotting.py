import matplotlib.pyplot as plt
import json
import os
from config import Config


def plot_training_curves(config: Config, show_plot: bool = True) -> None:
    """Load logs and create training curve plots."""
    log_file = os.path.join(config.log_dir, 'training_log.jsonl')

    # Load all logs
    logs: list[dict[str, any]] = []
    with open(log_file, 'r') as f:
        for line in f:
            logs.append(json.loads(line.strip()))

    epochs = [log['epoch'] for log in logs]
    train_losses = [log['train_loss'] for log in logs]
    test_losses = [log['test_loss'] for log in logs]
    train_accs = [log['train_acc'] for log in logs]
    test_accs = [log['test_acc'] for log in logs]

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
    fig_path = os.path.join(config.log_dir, 'training_curves.png')
    plt.savefig(fig_path, dpi=300, bbox_inches='tight')
    print(f'Figure saved to {fig_path}')

    if show_plot:
        plt.show()
    else:
        plt.close()
