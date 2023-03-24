import random
from pathlib import Path

import hydra
import numpy as np
import torch.multiprocessing
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from crf import dense_crf
from train_segmentation import LitUnsupervisedSegmenter
from utils import flexible_collate, get_transform, prep_args

torch.multiprocessing.set_sharing_strategy("file_system")


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = Path(root)
        self.transform = transform
        self.images = list(self.root.iterdir())

    def __getitem__(self, index):
        image = Image.open(self.root / self.images[index]).convert("RGB")
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="configs", config_name="master_config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    result_dir = Path("../results/predictions/") / f"{cfg.demo.experiment_name}"
    result_dir.mkdir(parents=True, exist_ok=True)
    (result_dir / "cluster").mkdir(parents=True, exist_ok=True)
    (result_dir / "linear").mkdir(parents=True, exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.demo.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
        root=cfg.demo.image_dir, transform=get_transform(cfg.demo.res, False, "center")
    )

    loader = DataLoader(
        dataset,
        cfg.demo.batch_size * 2,
        shuffle=False,
        num_workers=cfg.demo.num_workers,
        pin_memory=True,
        collate_fn=flexible_collate,
    )

    if cfg.use_cuda:
        model.eval().cuda()
    else:
        model.eval()
    if cfg.demo.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            if cfg.use_cuda:
                img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(
                code, img.shape[-2:], mode="bilinear", align_corners=False
            )

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            for j in range(img.shape[0]):
                single_img = img[j].cpu()
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                Image.fromarray(linear_crf.astype(np.uint8)).save(
                    result_dir / "linear" / new_name
                )
                Image.fromarray(cluster_crf.astype(np.uint8)).save(
                    result_dir / "cluster" / new_name
                )


if __name__ == "__main__":
    prep_args()
    my_app()
