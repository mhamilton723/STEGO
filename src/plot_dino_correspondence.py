import os
from os.path import join
from utils import get_transform, load_model, prep_for_plot, remove_axes, prep_args
from modules import FeaturePyramidNet, DinoFeaturizer, sample
from data import ContrastiveSegDataset
import hydra
import matplotlib.animation as animation
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.utilities.seed import seed_everything
from torch.utils.data import DataLoader
from tqdm import tqdm
from matplotlib.colors import ListedColormap


def plot_heatmap(ax, image, heatmap, cmap="bwr", color=False, plot_img=True, symmetric=True):
    vmax = np.abs(heatmap).max()
    if not color:
        bw = np.dot(np.array(image)[..., :3], [0.2989, 0.5870, 0.1140])
        image = np.ones_like(image) * np.expand_dims(bw, -1)

    if symmetric:
        kwargs = dict(vmax=vmax, vmin=-vmax)
    else:
        kwargs = {}

    if plot_img:
        return [
            ax.imshow(image),
            ax.imshow(heatmap, alpha=.5, cmap=cmap, **kwargs),
        ]
    else:
        return [ax.imshow(heatmap, alpha=.5, cmap=cmap, **kwargs)]


def get_heatmaps(net, img, img_pos, query_points):
    feats1, _ = net(img.cuda())
    feats2, _ = net(img_pos.cuda())

    sfeats1 = sample(feats1, query_points)

    attn_intra = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats1, dim=1))
    attn_intra -= attn_intra.mean([3, 4], keepdims=True)
    attn_intra = attn_intra.clamp(0).squeeze(0)

    attn_inter = torch.einsum("nchw,ncij->nhwij", F.normalize(sfeats1, dim=1), F.normalize(feats2, dim=1))
    attn_inter -= attn_inter.mean([3, 4], keepdims=True)
    attn_inter = attn_inter.clamp(0).squeeze(0)

    heatmap_intra = F.interpolate(
        attn_intra, img.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()
    heatmap_inter = F.interpolate(
        attn_inter, img_pos.shape[2:], mode="bilinear", align_corners=True).squeeze(0).detach().cpu()

    return heatmap_intra, heatmap_inter


@hydra.main(config_path="configs", config_name="plot_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    result_dir = join(cfg.output_root, "results", "correspondence")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)
    seed_everything(seed=0, workers=True)
    high_res = 512

    transform = get_transform(high_res, False, "center")
    use_loader = True

    if use_loader:
        dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=cfg.dataset_name,
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

    data_dir = join(cfg.output_root, "data")
    if cfg.arch == "feature-pyramid":
        cut_model = load_model(cfg.model_type, data_dir).cuda()
        net = FeaturePyramidNet(cfg.granularity, cut_model, cfg.dim, cfg.continuous)
    elif cfg.arch == "dino":
        net = DinoFeaturizer(cfg.dim, cfg)
    else:
        raise ValueError("Unknown arch {}".format(cfg.arch))
    net = net.cuda()

    for batch_val in loader:
        batch = batch_val
        break

    colors = [(1, 0, 0), (0, 1, 0), (0, 0, 1), (1, 1, 0)]
    cmaps = [
        ListedColormap([(1, 0, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 1, 0, i / 255) for i in range(255)]),
        ListedColormap([(0, 0, 1, i / 255) for i in range(255)]),
        ListedColormap([(1, 1, 0, i / 255) for i in range(255)])
    ]

    with torch.no_grad():
        if cfg.plot_correspondence:
            img_num = 6
            query_points = torch.tensor(
                [
                    [-.1, 0.0],
                    [.5, .8],
                    [-.7, -.7],
                ]
            ).reshape(1, 3, 1, 2).cuda()

            img = batch["img"][img_num:img_num + 1]
            img_pos = batch["img_pos"][img_num:img_num + 1]

            plt.style.use('dark_background')
            fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
            remove_axes(axes)
            axes[0].set_title("Image and Query Points", fontsize=20)
            axes[1].set_title("Self Correspondence", fontsize=20)
            axes[2].set_title("KNN Correspondence", fontsize=20)
            fig.tight_layout()

            heatmap_intra, heatmap_inter = get_heatmaps(net, img, img_pos, query_points)
            for point_num in range(query_points.shape[1]):
                point = ((query_points[0, point_num, 0] + 1) / 2 * high_res).cpu()
                img_point_h = point[0]
                img_point_w = point[1]

                plot_img = point_num == 0
                if plot_img:
                    axes[0].imshow(prep_for_plot(img[0]))
                axes[0].scatter(img_point_h, img_point_w,
                                c=colors[point_num], marker="x", s=500, linewidths=5)

                plot_heatmap(axes[1], prep_for_plot(img[0]) * .8, heatmap_intra[point_num],
                             plot_img=plot_img, cmap=cmaps[point_num], symmetric=False)
                plot_heatmap(axes[2], prep_for_plot(img_pos[0]) * .8, heatmap_inter[point_num],
                             plot_img=plot_img, cmap=cmaps[point_num], symmetric=False)
            plt.show()

        if cfg.plot_movie:
            img_num = 6
            key_points = [
                [-.7, -.7],
                [-.1, 0.0],
                [.5, .8],
            ]
            all_points = []
            for i in range(len(key_points)):
                all_points.extend([key_points[i]] * 60)

                if i < len(key_points) - 1:
                    all_points.extend(
                        np.stack([
                            np.linspace(key_points[i][0], key_points[i + 1][0], 50),
                            np.linspace(key_points[i][1], key_points[i + 1][1], 50),
                        ], axis=1).tolist())
            query_points = torch.tensor(all_points).reshape(1, len(all_points), 1, 2).cuda()


            plt.style.use('dark_background')
            fig, axes = plt.subplots(1, 3, figsize=(3 * 5, 1 * 5), dpi=100)
            remove_axes(axes)
            axes[0].set_title("Image and Query Points", fontsize=20)
            axes[1].set_title("Self Correspondence", fontsize=20)
            axes[2].set_title("KNN Correspondence", fontsize=20)

            fig.tight_layout()

            heatmap_intra, heatmap_inter = get_heatmaps(net, img, img_pos, query_points)

            frames = []  # for storing the generated images
            for point_num in range(query_points.shape[1]):
                point = ((query_points[0, point_num, 0] + 1) / 2 * high_res).cpu()
                img_point_h = point[0]
                img_point_w = point[1]

                frame = []

                frame.append(axes[0].imshow(prep_for_plot(img[0])))

                frame.extend([
                    axes[0].scatter(img_point_h, img_point_w,
                                    c=colors[0], marker="x", s=400, linewidths=4),
                    *plot_heatmap(axes[1], prep_for_plot(img[0]) * .8, heatmap_intra[point_num],
                                  cmap=cmaps[0], symmetric=False),
                    *plot_heatmap(axes[2], prep_for_plot(img_pos[0]) * .8, heatmap_inter[point_num],
                                  cmap=cmaps[0], symmetric=False)
                ])

                frames.append(frame)

            os.makedirs(result_dir, exist_ok=True)

            with tqdm(total=len(frames)) as pbar:
                animation.ArtistAnimation(fig, frames, blit=True).save(
                    join(result_dir, 'attention_interp.mp4'),
                    progress_callback=lambda i, n: pbar.update(),
                    writer=animation.FFMpegWriter(fps=30))


if __name__ == "__main__":
    prep_args()
    my_app()
