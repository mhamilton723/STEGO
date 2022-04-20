import io
from datetime import datetime

import PIL.Image
import hydra
import pytorch_lightning as pl
import seaborn as sns
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities.seed import seed_everything
from sklearn.metrics import auc, precision_recall_curve, average_precision_score
from torch.utils.tensorboard.summary import hparams
from torchvision.transforms import ToTensor
from data import *
from modules import *
from train_segmentation import get_class_labels



@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def prep_fd_coord(fd):
    fd -= fd.mean([3, 4], keepdim=True)
    fd /= fd.std([3, 4], keepdim=True)
    return fd.reshape(-1)


def prep_fd(fd):
    fd -= fd.min()
    fd /= fd.max()
    return fd.reshape(-1)


def prep_fd_2(fd):
    fd -= fd.mean([3, 4], keepdim=True)
    fd -= fd.min()
    fd /= fd.max()
    return fd


def plot_auc_raw(name, fpr, tpr):
    fpr, tpr = fpr.detach().cpu().squeeze(), tpr.detach().cpu().squeeze()
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, label=name + ' AUC = %0.2f' % roc_auc)


class CRFModule(nn.Module):

    def __init__(self):
        super().__init__()
        self.w1 = torch.nn.Parameter(torch.tensor(10.), requires_grad=True)
        self.w2 = torch.nn.Parameter(torch.tensor(3.), requires_grad=True)
        self.shift = torch.nn.Parameter(torch.tensor(-.3), requires_grad=True)
        self.alpha = torch.nn.Parameter(torch.tensor(.5), requires_grad=True)
        self.beta = torch.nn.Parameter(torch.tensor(.15), requires_grad=True)
        self.gamma = torch.nn.Parameter(torch.tensor(.05), requires_grad=True)

    def forward(self, coord_diff, img_diff):
        return torch.abs(self.w1) * torch.exp(- coord_diff / (2 * torch.exp(self.alpha))
                                              - img_diff / (2 * torch.exp(self.beta))) + \
               torch.abs(self.w2) * torch.exp(- coord_diff / (2 * torch.exp(self.gamma))) - self.shift


class LitRecalibrator(pl.LightningModule):
    def __init__(self, n_classes, cfg):
        super().__init__()
        self.cfg = cfg
        self.n_classes = n_classes

        if not cfg.continuous:
            dim = n_classes
        else:
            dim = cfg.dim

        data_dir = join(cfg.output_root, "data")
        self.moco = FeaturePyramidNet(cfg.granularity, load_model("mocov2", data_dir).cuda(), dim, cfg.continuous)
        # self.dino = DinoFeaturizer(dim, cfg)
        # self.dino = LitUnsupervisedSegmenter.load_from_checkpoint("../models/vit_base_cocostuff27.ckpt").net
        # self.crf = CRFModule()
        self.cm_metrics = UnsupervisedMetrics(
            "confusion_matrix/", n_classes, 0, False)
        self.automatic_optimization = False

        if self.cfg.dataset_name.startswith("cityscapes"):
            self.label_cmap = create_cityscapes_colormap()
        else:
            self.label_cmap = create_pascal_label_colormap()

    def get_crf_fd(self, img, coords1, coords2):
        with torch.no_grad():
            n = img.shape[0]
            [h1, w1, h2, w2] = [self.cfg.feature_samples] * 4
            img_samples_1 = sample(img, coords1).permute(0, 2, 3, 1).reshape(n, -1, 1, 3)
            img_samples_2 = sample(img, coords2).permute(0, 2, 3, 1).reshape(n, 1, -1, 3)
            coord_diff = (coords1.reshape(n, -1, 1, 2) - coords2.reshape(n, 1, -1, 2)) \
                .square().sum(-1).reshape(n, h1, w1, h2, w2)

            img_diff = (img_samples_1 - img_samples_2).square().sum(-1).reshape(n, h1, w1, h2, w2)

            return self.crf(coord_diff, img_diff)

    def get_net_fd(self, feats1, feats2, label1, label2, coords1, coords2):
        with torch.no_grad():
            feat_samples1 = sample(feats1, coords1)
            feat_samples2 = sample(feats2, coords2)

            label_samples1 = sample(F.one_hot(label1 + 1, self.n_classes + 1)
                                    .to(torch.float).permute(0, 3, 1, 2), coords1)
            label_samples2 = sample(F.one_hot(label2 + 1, self.n_classes + 1)
                                    .to(torch.float).permute(0, 3, 1, 2), coords2)

            fd = tensor_correlation(norm(feat_samples1), norm(feat_samples2))
            ld = tensor_correlation(label_samples1, label_samples2)

        return ld, fd, label_samples1.argmax(1), label_samples2.argmax(1)

    def training_step(self, batch, batch_idx):
        return None

    def validation_step(self, batch, batch_idx):
        with torch.no_grad():
            img = batch["img"]
            label = batch["label"]

            dino_feats, dino_code = self.dino(img)
            moco_feats, moco_code = self.moco(img)

            coord_shape = [img.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]
            coords1 = torch.rand(coord_shape, device=img.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=img.device) * 2 - 1

            crf_fd = self.get_crf_fd(img, coords1, coords2)

            ld, stego_fd, l1, l2 = self.get_net_fd(dino_code, dino_code, label, label, coords1, coords2)
            ld, dino_fd, l1, l2 = self.get_net_fd(dino_feats, dino_feats, label, label, coords1, coords2)
            ld, moco_fd, l1, l2 = self.get_net_fd(moco_feats, moco_feats, label, label, coords1, coords2)

            return dict(
                dino_fd=dino_fd,
                stego_fd=stego_fd,
                moco_fd=moco_fd,
                crf_fd=crf_fd,
                ld=ld
            )

    def validation_epoch_end(self, outputs) -> None:
        # self.cm_metrics.compute()

        all_outputs = {}
        for k in outputs[0].keys():
            t = torch.cat([o[k] for o in outputs], dim=0)
            all_outputs[k] = t

        def plot_pr(preds, targets, name):
            preds = preds.cpu().reshape(-1)
            preds -= preds.min()
            preds /= preds.max()
            targets = targets.to(torch.int64).cpu().reshape(-1)
            precisions, recalls, _ = precision_recall_curve(targets, preds)
            average_precision = average_precision_score(targets, preds)
            plt.plot(recalls, precisions, label="AP={}% {}".format(int(average_precision * 100), name))

        def plot_cm():
            histogram = self.cm_metrics.histogram
            fig = plt.figure(figsize=(10, 10))
            ax = fig.gca()
            hist = histogram.detach().cpu().to(torch.float32)
            hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
            sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
            ax.set_title('KNN Labels', fontsize=28)
            ax.set_ylabel('Image labels', fontsize=28)
            names = get_class_labels(self.cfg.dataset_name)
            if self.cfg.extra_clusters:
                names = names + ["Extra"]
            ax.set_xticks(np.arange(0, len(names)) + .5)
            ax.set_yticks(np.arange(0, len(names)) + .5)
            ax.xaxis.tick_top()
            ax.xaxis.set_ticklabels(names, fontsize=18)
            ax.yaxis.set_ticklabels(names, fontsize=18)
            colors = [self.label_cmap[i] / 255.0 for i in range(len(names))]
            [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
            [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
            plt.xticks(rotation=90)
            plt.yticks(rotation=0)
            ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
            ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
            plt.tight_layout()

        if self.trainer.is_global_zero:
            # plt.style.use('dark_background')
            print("Plotting")
            plt.figure(figsize=(5, 4), dpi=100)
            plot_cm()
            plt.tight_layout()
            plt.show()
            plt.clf()

            print("Plotting")
            # plt.style.use('dark_background')
            plt.figure(figsize=(5, 4), dpi=100)
            ld = all_outputs["ld"]
            plot_pr(prep_fd(all_outputs["stego_fd"]), ld, "STEGO (Ours)")
            plot_pr(prep_fd(all_outputs["dino_fd"]), ld, "DINO")
            plot_pr(prep_fd(all_outputs["moco_fd"]), ld, "MoCoV2")
            plot_pr(prep_fd(all_outputs["crf_fd"]), ld, "CRF")
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.legend(fontsize=12)
            plt.ylabel('Precision', fontsize=16)
            plt.xlabel('Recall', fontsize=16)
            plt.tight_layout()
            plt.show()

        return None

    def configure_optimizers(self):
        return None


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir
    data_dir = join(cfg.output_root, "data")
    log_dir = join(cfg.output_root, "logs")
    checkpoint_dir = join(cfg.output_root, "checkpoints")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(log_dir, exist_ok=True)

    seed_everything(seed=0, workers=True)

    train_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=cfg.crop_type,
        image_set="train",
        transform=get_transform(cfg.res, False, cfg.loader_crop_type),
        target_transform=get_transform(cfg.res, True, cfg.loader_crop_type),
        cfg=cfg,
        aug_geometric_transform=None,
        aug_photometric_transform=None,
        num_neighbors=cfg.num_neighbors,
        mask=True,
        pos_images=True,
        pos_labels=True
    )

    val_loader_crop = "center"
    val_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name=cfg.dataset_name,
        crop_type=None,
        image_set="val",
        transform=get_transform(320, False, val_loader_crop),
        target_transform=get_transform(320, True, val_loader_crop),
        mask=True,
        pos_images=True,
        pos_labels=True,
        cfg=cfg,
    )

    train_loader = DataLoader(train_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_loader = DataLoader(val_dataset, cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)

    model = LitRecalibrator(train_dataset.n_classes, cfg)

    prefix = "{}_{}".format(cfg.dataset_name, cfg.experiment_name)
    name = '{}_date_{}'.format(prefix, datetime.now().strftime('%b%d_%H-%M-%S'))
    tb_logger = TensorBoardLogger(
        join(log_dir, cfg.log_dir, name),
        default_hp_metric=False
    )
    steps = 1
    trainer = Trainer(
        log_every_n_steps=10,
        val_check_interval=steps,
        gpus=1,
        max_steps=steps,
        limit_val_batches=100,
        accelerator="ddp",
        num_sanity_val_steps=0,
        logger=tb_logger,
    )
    trainer.fit(model, train_loader, val_loader)
    os.makedirs(join(checkpoint_dir, cfg.log_dir), exist_ok=True)


if __name__ == "__main__":
    prep_args()
    my_app()
