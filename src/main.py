from config import Config
from plotting import plot_training_curves
from src.trainer import train_model

if __name__ == '__main__':
    config = Config(tag='P_47_WeightDecay_10', num_epochs=10_000, p=113, weight_decay=10)

    # train_model(config, resume=True)
    plot_training_curves(config)
