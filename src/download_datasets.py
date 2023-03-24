from pathlib import Path

import hydra
import wget
from omegaconf import DictConfig

from utils import prep_args


@hydra.main(config_path="configs", config_name="master_config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = Path(cfg.pytorch_data_dir)
    dataset_names = ["potsdam", "cityscapes", "cocostuff", "potsdamraw"]
    url_base = (
        "https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/"
    )

    pytorch_data_dir.mkdir(exist_ok=True)
    for dataset_name in dataset_names:
        if not (
            (pytorch_data_dir / dataset_name).exists()
            or (pytorch_data_dir / f"{dataset_name}.zip").exists()
        ):
            print(f"\n Downloading {dataset_name}")
            wget.download(
                url_base + dataset_name + ".zip",
                str(pytorch_data_dir / f"{dataset_name}.zip"),
            )
        else:
            print(f"\n Found {dataset_name}, skipping download")


if __name__ == "__main__":
    prep_args()
    my_app()
