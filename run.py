import hydra
from pathlib import Path
from omegaconf import DictConfig


@hydra.main(config_path='hyperparams', config_name='atari_drqdqn')
def main(cfg: DictConfig) -> None:
    from train import Workspace as W
    root_dir = Path.cwd()
    workspace = W(cfg)
    snapshot = root_dir / 'snapshot.pt'
    if snapshot.exists():
        print(f'resuming: {snapshot}')
        workspace.load_snapshot()
    workspace.train()


if __name__ == '__main__':
    main()
