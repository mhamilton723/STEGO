import os
from pathlib import Path

import hydra
import torch
from lightning_fabric.utilities.seed import seed_everything
from omegaconf import DictConfig, OmegaConf
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms as T
from torchvision.transforms.functional import crop, five_crop, get_image_size
from tqdm import tqdm

from data import ContrastiveSegDataset
from utils import ToTargetTensor, prep_args


def _random_crops(img, size, seed, n):
    """Crop the given image into four corners and the central crop.
    If the image is torch Tensor, it is expected
    to have [..., H, W] shape, where ... means an arbitrary number of leading dimensions

    .. Note::
        This transform returns a tuple of images and there may be a
        mismatch in the number of inputs and targets your ``Dataset`` returns.

    Args:
        img (PIL Image or Tensor): Image to be cropped.
        size (sequence or int): Desired output size of the crop. If size is an
            int instead of sequence like (h, w), a square crop (size, size) is
            made. If provided a sequence of length 1, it will be interpreted as (size[0], size[0]).

    Returns:
       tuple: tuple (tl, tr, bl, br, center)
                Corresponding top left, top right, bottom left, bottom right and center crop.
    """
    if isinstance(size, int):
        size = (int(size), int(size))
    elif isinstance(size, (tuple, list)) and len(size) == 1:
        size = (size[0], size[0])

    if len(size) != 2:
        raise ValueError("Please provide only two dimensions (h, w) for size.")

    image_width, image_height = get_image_size(img)
    crop_height, crop_width = size
    if crop_width > image_width or crop_height > image_height:
        raise ValueError(
            f"Requested crop size {size} is bigger than "
            f"input size {(image_height, image_width)}"
        )

    images = []
    for i in range(n):
        seed1 = hash((seed, i, 0))
        seed2 = hash((seed, i, 1))
        crop_height, crop_width = int(crop_height), int(crop_width)

        top = seed1 % (image_height - crop_height)
        left = seed2 % (image_width - crop_width)
        images.append(crop(img, top, left, crop_height, crop_width))

    return images


class RandomCropComputer(Dataset):
    def __init__(self, cfg, dataset_name, image_set, crop_type, crop_ratio):
        self.pytorch_data_dir = Path(cfg.pytorch_data_dir)
        self.crop_ratio = crop_ratio
        self.save_dir = (
            self.pytorch_data_dir
            / "cropped"
            / f"{dataset_name}_{crop_type}_crop_{crop_ratio}"
        )
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.cfg = cfg

        self.img_dir = self.save_dir / "img" / image_set
        self.label_dir = self.save_dir / "label" / image_set
        self.img_dir.mkdir(exist_ok=True)
        self.label_dir.mkdir(exist_ok=True)

        if crop_type == "random":
            cropper = self.random_cropper
        elif crop_type == "five":
            cropper = self.five_cropper
        else:
            raise ValueError(f"Unknown crop type {crop_type}")

        self.dataset = ContrastiveSegDataset(
            cfg.pytorch_data_dir,
            dataset_name,
            None,
            image_set,
            T.ToTensor(),
            ToTargetTensor(),
            cfg=cfg,
            num_neighbors=cfg.train.num_neighbors,
            pos_labels=False,
            pos_images=False,
            mask=False,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            extra_transform=cropper,
        )

    def random_cropper(self, i, x):
        return self.random_crops(i, x)

    def five_cropper(self, i, x):
        return self.five_crops(i, x)

    def _get_size(self, img):
        if len(img.shape) == 3:
            return [
                int(img.shape[1] * self.crop_ratio),
                int(img.shape[2] * self.crop_ratio),
            ]
        elif len(img.shape) == 2:
            return [
                int(img.shape[0] * self.crop_ratio),
                int(img.shape[1] * self.crop_ratio),
            ]
        else:
            raise ValueError(f"Bad image shape {img.shape}")

    def random_crops(self, i, img):
        return _random_crops(img, self._get_size(img), i, 5)

    def five_crops(self, i, img):
        return five_crop(img, self._get_size(img))

    def __getitem__(self, item):
        batch = self.dataset[item]
        imgs = batch["img"]
        labels = batch["label"]
        for crop_num, (img, label) in enumerate(zip(imgs, labels)):
            img_num = item * 5 + crop_num
            img_arr = (
                img.mul(255)
                .add_(0.5)
                .clamp_(0, 255)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
            )
            label_arr = (
                (label + 1)
                .unsqueeze(0)
                .permute(1, 2, 0)
                .to("cpu", torch.uint8)
                .numpy()
                .squeeze(-1)
            )
            Image.fromarray(img_arr).save(self.img_dir / f"{img_num}.jpg", "JPEG")
            Image.fromarray(label_arr).save(self.label_dir / f"{img_num}.png", "PNG")
        return True

    def __len__(self):
        return len(self.dataset)


def identity(l):
    return l


@hydra.main(config_path="configs", config_name="master_config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    seed_everything(seed=0, workers=True)

    dataset_names = cfg.crop_knn.dataset_names
    image_sets = cfg.crop_knn.image_sets
    crop_types = cfg.crop_knn.crop_types
    crop_ratios = cfg.crop_knn.crop_ratios

    for crop_ratio in crop_ratios:
        for crop_type in crop_types:
            for dataset_name in dataset_names:
                for image_set in image_sets:
                    print(
                        f"crop_ratio={crop_ratio}, crop_type={crop_type}, "
                        f"dataset_name={dataset_name}, image_set={image_set}"
                    )
                    dataset = RandomCropComputer(
                        cfg, dataset_name, image_set, crop_type, crop_ratio
                    )
                    loader = DataLoader(
                        dataset,
                        1,
                        shuffle=False,
                        num_workers=cfg.num_workers,
                        collate_fn=identity,
                    )
                    for _ in tqdm(loader):
                        pass


if __name__ == "__main__":
    prep_args()
    my_app()
