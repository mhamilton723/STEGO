from utils import *
import hydra
from omegaconf import DictConfig
import os
import wget


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    dataset_names = [
        "potsdam",
        "cityscapes",
        "cocostuff",
        "potsdamraw"]
    url_base = "https://marhamilresearch4.blob.core.windows.net/stego-public/pytorch_data/"

    os.makedirs(pytorch_data_dir, exist_ok=True)
    for dataset_name in dataset_names:
        if (not os.path.exists(join(pytorch_data_dir, dataset_name))) or \
                (not os.path.exists(join(pytorch_data_dir, dataset_name + ".zip"))):
            print("\n Downloading {}".format(dataset_name))
            wget.download(url_base + dataset_name + ".zip", join(pytorch_data_dir, dataset_name + ".zip"))
        else:
            print("\n Found {}, skipping download".format(dataset_name))


if __name__ == "__main__":
    prep_args()
    my_app()
