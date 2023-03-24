from pathlib import Path
from typing import Any

import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from lightning_fabric.utilities.seed import seed_everything
from matplotlib.colors import ListedColormap
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm

from data import ContrastiveSegDataset
from modules import DinoFeaturizer, FeaturePyramidNet, sample
from utils import get_transform, load_model, prep_args, prep_for_plot, remove_axes


def plot_heatmap(
    ax, image, heatmap, cmap="bwr", color=False, plot_img=True, symmetric=True
):
    vmax = np.abs(heatmap).max()
    if not color:
        bw = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
        image = np.ones_like(image) * np.expand_dims(bw, -1)

    if symmetric:
        kwargs = dict(vmax=vmax, vmin=-vmax)
    else:
        kwargs = {}

    if plot_img:
        return [ax.imshow(image), ax.imshow(heatmap, alpha=0.5, cmap=cmap, **kwargs)]
    else:
        return [ax.imshow(heatmap, alpha=0.5, cmap=cmap, **kwargs)]


def get_heatmaps(net, img, img_pos, query_points, cfg):
    if cfg.use_cuda:
        feats1, _ = net(img.cuda())
        feats2, _ = net(img_pos.cuda())
    else:
        feats1, _ = net(img)
        feats2, _ = net(img_pos)

    sfeats1 = sample(feats1, query_points)

    attn_intra = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1)
    )
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)

    attn_inter = torch.einsum(
        "nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats2, dim=1)
    )
    attn_inter -= attn_inter.mean([3, 4], keepdims=True)
    attn_inter = attn_inter.clamp(0).squeeze(0)

    heatmap_intra = (
        F.interpolate(attn_intra, img.shape[2:], mode="bilinear", align_corners=True)
        .squeeze(0)
        .detach()
        .cpu()
    )
    heatmap_inter = (
        F.interpolate(
            attn_inter, img_pos.shape[2:], mode="bilinear", align_corners=True
        )
        .squeeze(0)
        .detach()
        .cpu()
    )

    return heatmap_intra, heatmap_inter


@hydra.main(config_path="configs", config_name="master_config", version_base="1.1")
def my_app(cfg: DictConfig) -> None:
    net: Any
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = Path(cfg.pytorch_data_dir)
    data_dir = Path(cfg.output_root) / "data"
    log_dir = Path(cfg.output_root) / "logs"
    result_dir = Path(cfg.output_root) / "results" / "correspondence"
    data_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(seed=0, workers=True)
    high_res = 512

    transform = get_transform(high_res, False, "center")
    use_loader = True

    if use_loader:
        dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=cfg.plot.dataset_name,
            crop_type=None,
            image_set="train",
            transform=transform,
            target_transform=get_transform(high_res, True, "center"),
            cfg=cfg,
            aug_geometric_transform=None,
            aug_photometric_transform=None,
            num_neighbors=2,
            mask=True,
            pos_images=True,
            pos_labels=True,
        )
        loader = DataLoader(dataset, 16, shuffle=True, num_workers=cfg.num_workers)

    data_dir = Path(cfg.output_root) / "data"
    if cfg.plot.arch == "feature-pyramid":
        if cfg.use_cuda:
            cut_model = load_model(cfg.plot.model_type, data_dir).cuda()
        else:
            cut_model = load_model(cfg.plot.model_type, data_dir)
        net = FeaturePyramidNet(
            cfg.plot.granularity, cut_model, cfg.plot.dim, cfg.plot.continuous
        )
    elif cfg.plot.arch == "dino":
        net = DinoFeaturizer(cfg.plot.dim, cfg)
    else:
        raise ValueError(f"Unknown arch {cfg.plot.arch}")
    if cfg.use_cuda:
        net = net.cuda()

    for batch_val in loader:
        batch = batch_val
        break

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    cmaps = [
        ListedColormap([(1, 0, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 1, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 0, 1, i / 255) for i in range(255)]),
        ListedColormap([(1, 1, 0, i / 255) for i in range(255)]),
    ]

    with torch.no_grad():
        if cfg.plot.plot_correspondence:
            img_num = 6
            if cfg.use_cuda:
                query_points = (
                    torch.tensor([[-0.1, 0.0], [0.5, 0.8], [-0.7, -0.7]])
                    .reshape(1, 3, 1, 2)
                    .cuda()
                )
            else:
                query_points = torch.tensor(
                    [[-0.1, 0.0], [0.5, 0.8], [-0.7, -0.7]]
                ).reshape(1, 3, 1, 2)

            img = batch["img"][img_num : img_num + 1]
            img_pos = batch["img_pos"][img_num : img_num + 1]

            plt.style.use("dark_background")
            fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
            remove_axes(axes)
            axes[0].set_title("Image and Query Points", fontsize=20)
            axes[1].set_title("Self Correspondence", fontsize=20)
            axes[2].set_title("KNN Correspondence", fontsize=20)
            fig.tight_layout()

            heatmap_intra, heatmap_inter = get_heatmaps(
                net, img, img_pos, query_points, cfg
            )
            for point_num in range(query_points.shape[1]):
                point = ((query_points[0, point_num, 0] + 1) / 2 * high_res).cpu()
                img_point_h = point[0]
                img_point_w = point[1]

                plot_img = point_num == 0
                if plot_img:
                    axes[0].imshow(prep_for_plot(img[0]))
                axes[0].scatter(
                    img_point_h,
                    img_point_w,
                    c=colors[point_num],
                    marker="x",
                    s=500,
                    linewidths=5,
                )

                plot_heatmap(
                    axes[1],
                    prep_for_plot(img[0]) * 0.8,
                    heatmap_intra[point_num],
                    plot_img=plot_img,
                    cmap=cmaps[point_num],
                    symmetric=False,
                )
                plot_heatmap(
                    axes[2],
                    prep_for_plot(img_pos[0]) * 0.8,
                    heatmap_inter[point_num],
                    plot_img=plot_img,
                    cmap=cmaps[point_num],
                    symmetric=False,
                )
            plt.show()

        if cfg.plot.plot_movie:
            img_num = 6
            key_points = [[-0.7, -0.7], [-0.1, 0.0], [0.5, 0.8]]
            all_points = []
            for i in range(len(key_points)):
                all_points.extend([key_points[i]] * 60)

                if i < len(key_points) - 1:
                    all_points.extend(
                        np.stack(
                            [
                                np.linspace(key_points[i][0], key_points[i + 1][0], 50),
                                np.linspace(key_points[i][1], key_points[i + 1][1], 50),
                            ],
                            axis=1,
                        ).tolist()
                    )
            if cfg.use_cuda:
                query_points = (
                    torch.tensor(all_points).reshape(1, len(all_points), 1, 2).cuda()
                )
            else:
                query_points = torch.tensor(all_points).reshape(
                    1, len(all_points), 1, 2
                )

            plt.style.use("dark_background")
            fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
            remove_axes(axes)
            axes[0].set_title("Image and Query Points", fontsize=20)
            axes[1].set_title("Self Correspondence", fontsize=20)
            axes[2].set_title("KNN Correspondence", fontsize=20)

            fig.tight_layout()

            heatmap_intra, heatmap_inter = get_heatmaps(
                net, img, img_pos, query_points, cfg
            )

            frames = []  # for storing the generated images
            for point_num in range(query_points.shape[1]):
                point = ((query_points[0, point_num, 0] + 1) / 2 * high_res).cpu()
                img_point_h = point[0]
                img_point_w = point[1]

                frame = []

                frame.append(axes[0].imshow(prep_for_plot(img[0])))

                frame.extend(
                    [
                        axes[0].scatter(
                            img_point_h,
                            img_point_w,
                            c=colors[0],
                            marker="x",
                            s=400,
                            linewidths=4,
                        ),
                        *plot_heatmap(
                            axes[1],
                            prep_for_plot(img[0]) * 0.8,
                            heatmap_intra[point_num],
                            cmap=cmaps[0],
                            symmetric=False,
                        ),
                        *plot_heatmap(
                            axes[2],
                            prep_for_plot(img_pos[0]) * 0.8,
                            heatmap_inter[point_num],
                            cmap=cmaps[0],
                            symmetric=False,
                        ),
                    ]
                )

                frames.append(frame)

            result_dir.mkdir(parents=True, exist_ok=True)

            with tqdm(total=len(frames)) as pbar:
                animation.ArtistAnimation(fig, frames, blit=True).save(
                    result_dir / "attention_interp.mp4",
                    progress_callback=lambda i, n: pbar.update(),
                    writer=animation.FFMpegWriter(fps=30),
                )


if __name__ == "__main__":
    prep_args()
    my_app()
