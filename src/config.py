import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable
import git


def get_repo_root() -> str:
    """Find the repository root using GitPython."""
    try:
        repo = git.Repo(Path(__file__).resolve(), search_parent_directories=True)
        return str(repo.working_dir)
    except git.InvalidGitRepositoryError:
        # Fallback: assume we're in src/ and go up one level
        return str(Path(__file__).parent.parent)


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
        return os.path.join(get_repo_root(), 'logs', self.tag)

    @property
    def model_dir(self) -> str:
        return os.path.join(get_repo_root(), 'models', self.tag)
