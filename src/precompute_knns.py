from pathlib import Path

import hydra
import numpy as np
import torch.multiprocessing
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ContrastiveSegDataset
from utils import get_transform, load_model, prep_args


def get_feats(model, loader):
    all_feats = []
    for pack in tqdm(loader):
        img = pack["img"]
        # feats = F.normalize(model.forward(img.cuda()).mean([2, 3]), dim=1)
        feats = F.normalize(model.forward(img).mean([2, 3]), dim=1)
        all_feats.append(feats.to("cpu", non_blocking=True))
    return torch.cat(all_feats, dim=0).contiguous()


@hydra.main(config_path="configs", config_name="train_config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = Path(cfg.pytorch_data_dir)
    output_root = Path(cfg.output_root)
    data_dir = output_root / "data"
    log_dir = output_root / "logs"
    data_dir.mkdir(exist_ok=True)
    log_dir.mkdir(exist_ok=True)
    (pytorch_data_dir / "nns").mkdir(exist_ok=True)

    seed_everything(seed=0)

    print(data_dir)
    print(output_root)

    dataset_names = cfg["crop_knn"]["dataset_names"]
    image_sets = cfg["crop_knn"]["image_sets"]
    crop_types = cfg["crop_knn"]["crop_types"]

    res = 224
    n_batches = 16

    if cfg.arch == "dino":
        from modules import DinoFeaturizer, LambdaLayer

        # no_ap_model = torch.nn.Sequential(
        #     DinoFeaturizer(20, cfg), LambdaLayer(lambda p: p[0])  # dim doesent matter
        # ).cuda()
        no_ap_model = torch.nn.Sequential(
            DinoFeaturizer(20, cfg), LambdaLayer(lambda p: p[0])
        )  # dim doesent matter
    else:
        # cut_model = load_model(cfg.model_type, output_root / "data").cuda
        cut_model = load_model(cfg.model_type, output_root / "data")
        # no_ap_model = torch.nn.Sequential(*list(cut_model.children())[:-1]).cuda()
        no_ap_model = torch.nn.Sequential(*list(cut_model.children())[:-1])
    par_model = torch.nn.DataParallel(no_ap_model)

    for crop_type in crop_types:
        for image_set in image_sets:
            for dataset_name in dataset_names:
                print(
                    f"crop_type={crop_type}, img_set={image_set}, "
                    f"dataset_name={dataset_name}"
                )
                nice_dataset_name = (
                    cfg.dir_dataset_name
                    if dataset_name == "directory"
                    else dataset_name
                )

                feature_cache_file = (
                    pytorch_data_dir
                    / "nns"
                    / f"nns_{cfg.model_type}_{nice_dataset_name}_{image_set}_{crop_type}_{res}.npz"
                )

                if feature_cache_file.exists():
                    print(f"Skipping, already exists: {feature_cache_file}")
                    continue

                print(f"{feature_cache_file} not found, computing")
                dataset = ContrastiveSegDataset(
                    pytorch_data_dir=pytorch_data_dir,
                    dataset_name=dataset_name,
                    crop_type=crop_type,
                    image_set=image_set,
                    transform=get_transform(res, False, "center"),
                    target_transform=get_transform(res, True, "center"),
                    cfg=cfg,
                )

                loader = DataLoader(
                    dataset,
                    256,
                    shuffle=False,
                    num_workers=cfg.num_workers,
                    pin_memory=False,
                )

                with torch.no_grad():
                    normed_feats = get_feats(par_model, loader)
                    all_nns = []
                    step = normed_feats.shape[0] // n_batches
                    print(normed_feats.shape)
                    for i in tqdm(range(0, normed_feats.shape[0], step)):
                        # torch.cuda.empty_cache()
                        batch_feats = normed_feats[i : i + step, :]
                        pairwise_sims = torch.einsum(
                            "nf,mf->nm", batch_feats, normed_feats
                        )
                        all_nns.append(torch.topk(pairwise_sims, 30)[1])
                        del pairwise_sims
                    nearest_neighbors = torch.cat(all_nns, dim=0)

                    np.savez_compressed(
                        feature_cache_file, nns=nearest_neighbors.numpy()
                    )
                    print("Saved NNs", cfg.model_type, nice_dataset_name, image_set)


if __name__ == "__main__":
    prep_args()
    my_app()
