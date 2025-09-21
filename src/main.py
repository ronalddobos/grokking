from config import Config
from trainer import train_grokking


if __name__ == '__main__':
    config = Config(tag='P_47', num_epochs=10_000, p=47)
    train_grokking(config, resume=True)
