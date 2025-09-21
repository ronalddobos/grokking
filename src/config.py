import os
from dataclasses import dataclass
from typing import Callable


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
    def fn(self) -> Callable[[int, int], int]:
        return {'add': lambda x, y: (x + y) % self.p}[self.fn_name]

    @property
    def log_dir(self) -> str:
        return os.path.join('logs', self.tag)

    @property
    def model_dir(self) -> str:
        return os.path.join('models', self.tag)
