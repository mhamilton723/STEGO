try:
    from .core import *
    from .modules import *
except (ModuleNotFoundError, ImportError):
    from core import *
    from modules import *
import hydra
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels
import seaborn as sns
from collections import defaultdict

torch.multiprocessing.set_sharing_strategy('file_system')


def plot_cm(histogram, label_cmap, cfg):
    fig = plt.figure(figsize=(10, 10))
    ax = fig.gca()
    hist = histogram.detach().cpu().to(torch.float32)
    hist /= torch.clamp_min(hist.sum(dim=0, keepdim=True), 1)
    sns.heatmap(hist.t(), annot=False, fmt='g', ax=ax, cmap="Blues", cbar=False)
    ax.set_title('Predicted labels', fontsize=28)
    ax.set_ylabel('True labels', fontsize=28)
    names = get_class_labels(cfg.dataset_name)
    if cfg.extra_clusters:
        names = names + ["Extra"]
    ax.set_xticks(np.arange(0, len(names)) + .5)
    ax.set_yticks(np.arange(0, len(names)) + .5)
    ax.xaxis.tick_top()
    ax.xaxis.set_ticklabels(names, fontsize=18)
    ax.yaxis.set_ticklabels(names, fontsize=18)
    colors = [label_cmap[i] / 255.0 for i in range(len(names))]
    [t.set_color(colors[i]) for i, t in enumerate(ax.xaxis.get_ticklabels())]
    [t.set_color(colors[i]) for i, t in enumerate(ax.yaxis.get_ticklabels())]
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    ax.vlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_xlim())
    ax.hlines(np.arange(0, len(names) + 1), color=[.5, .5, .5], *ax.get_ylim())
    plt.tight_layout()


def batch_list(iterable, n=1):
    l = len(iterable)
    for ndx in range(0, l, n):
        yield iterable[ndx:min(ndx + n, l)]


class CRFComputer(Dataset):

    def __init__(self, dataset, outputs, run_crf):
        self.dataset = dataset
        self.outputs = outputs
        self.run_crf = run_crf

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, item):
        batch = self.dataset[item]
        if "linear_probs" in self.outputs:
            if self.run_crf:
                batch["linear_probs"] = dense_crf(batch["img"].cpu().detach(),
                                                  self.outputs["linear_probs"][item].cpu().detach())
                batch["cluster_probs"] = dense_crf(batch["img"].cpu().detach(),
                                                   self.outputs["cluster_probs"][item].cpu().detach())
            else:
                batch["linear_probs"] = self.outputs["linear_probs"][item]
                batch["cluster_probs"] = self.outputs["cluster_probs"][item]
            batch["no_crf_linear_probs"] = self.outputs["linear_probs"][item]
            batch["no_crf_cluster_probs"] = self.outputs["cluster_probs"][item]
        if "picie_preds" in self.outputs:
            batch["picie_preds"] = self.outputs["picie_preds"][item]
        return batch


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir

    # Best
    model_names = ["../saved_models/cocostuff27_vit_base_5.ckpt"]
    # model_names = ["../saved_models/cityscapes_vit_base_1.ckpt"]
    # model_names = ["../saved_models/potsdam_test.ckpt"]

    for model_name in model_names:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_name)
        print(OmegaConf.to_yaml(model.cfg))

        run_picie = cfg.run_picie and model.cfg.dataset_name == "cocostuff27"
        if run_picie:
            picie_state = torch.load("../saved_models/picie_and_probes.pth")
            picie = picie_state["model"]
            picie_cluster_probe = picie_state["cluster_probe"]
            picie_cluster_metrics = picie_state["cluster_metrics"]

        loader_crop = "center"
        test_dataset = ContrastiveSegDataset(
            pytorch_data_dir=pytorch_data_dir,
            dataset_name=model.cfg.dataset_name,
            crop_type=None,
            image_set="val",
            transform=get_transform(cfg.res, False, loader_crop),
            target_transform=get_transform(cfg.res, True, loader_crop),
            cfg=model.cfg,
        )

        test_loader = DataLoader(test_dataset, cfg.batch_size * 2,
                                 shuffle=False, num_workers=cfg.num_workers,
                                 pin_memory=True, collate_fn=flexible_collate)

        outputs = defaultdict(list)
        model.eval().cuda()
        par_model = torch.nn.DataParallel(model.net)
        if run_picie:
            par_picie = torch.nn.DataParallel(picie)

        if cfg.run_prediction:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    feats, code1 = par_model(img)
                    feats, code2 = par_model(img.flip(dims=[3]))
                    code = (code1 + code2.flip(dims=[3])) / 2

                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    outputs["linear_probs"].append(torch.log_softmax(model.linear_probe(code), dim=1).cpu())
                    outputs["cluster_probs"].append(model.cluster_probe(code, 2, log_probs=True).cpu())

                    if run_picie:
                        outputs["picie_preds"].append(picie_cluster_metrics.map_clusters(
                            picie_cluster_probe(par_picie(img), None)[1].argmax(1).cpu()))
        outputs = {k: torch.cat(v, dim=0) for k, v in outputs.items()}

        if cfg.run_crf:
            crf_batch_size = 5
        else:
            crf_batch_size = cfg.batch_size * 2
        crf_dataset = CRFComputer(test_dataset, outputs, cfg.run_crf)
        crf_loader = DataLoader(crf_dataset, crf_batch_size,
                                shuffle=False, num_workers=cfg.num_workers + 5,
                                pin_memory=True, collate_fn=flexible_collate)

        crf_outputs = defaultdict(list)
        for i, batch in enumerate(tqdm(crf_loader)):
            with torch.no_grad():
                label = batch["label"].cuda(non_blocking=True)
                img = batch["img"]
                if cfg.run_prediction:
                    linear_preds = batch["linear_probs"].cuda(non_blocking=True).argmax(1)
                    cluster_preds = batch["cluster_probs"].cuda(non_blocking=True).argmax(1)
                    no_crf_linear_preds = batch["no_crf_linear_probs"].cuda(non_blocking=True).argmax(1)
                    no_crf_cluster_preds = batch["no_crf_cluster_probs"].cuda(non_blocking=True).argmax(1)
                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)
                    crf_outputs['linear_preds'].append(linear_preds[:model.cfg.n_images].detach().cpu())
                    crf_outputs["cluster_preds"].append(cluster_preds[:model.cfg.n_images].detach().cpu()),
                    crf_outputs['no_crf_linear_preds'].append(no_crf_linear_preds[:model.cfg.n_images].detach().cpu())
                    crf_outputs["no_crf_cluster_preds"].append(
                        no_crf_cluster_preds[:model.cfg.n_images].detach().cpu()),
                if run_picie:
                    crf_outputs["picie_preds"].append(batch["picie_preds"][:model.cfg.n_images].detach().cpu())

                crf_outputs["img"].append(img[:model.cfg.n_images].detach().cpu())
                crf_outputs["label"].append(label[:model.cfg.n_images].detach().cpu())
        crf_outputs = {k: torch.cat(v, dim=0) for k, v in crf_outputs.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        print("")
        print(model_name)
        print(tb_metrics)

        if model.cfg.dataset_name == "cocostuff27":
            # all_good_images = range(100, 250)
            all_good_images = [61, 60, 49, 44, 13, 70]
            # all_good_images = [19, 54, 67, 66, 65, 75, 77, 76, 124]
        elif model.cfg.dataset_name == "cityscapes":
            # all_good_images = range(80)
            # all_good_images = [ 5, 20, 56]
            all_good_images = [11, 32, 43, 52]
        else:
            raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))

        if cfg.run_prediction:
            n_rows = 4
        else:
            n_rows = 3

        if run_picie:
            n_rows += 1

        if cfg.dark_mode:
            plt.style.use('dark_background')

        for good_images in batch_list(all_good_images, 10):
            fig, ax = plt.subplots(n_rows, len(good_images), figsize=(len(good_images) * 3, n_rows * 3))
            for i, img_num in enumerate(good_images):
                ax[0, i].imshow(prep_for_plot(crf_outputs["img"][img_num]))
                ax[1, i].imshow(model.label_cmap[crf_outputs["label"][img_num]])
                if cfg.run_prediction:
                    ax[2, i].imshow(
                        model.label_cmap[
                            model.test_cluster_metrics.map_clusters(crf_outputs["cluster_preds"][img_num])])
                    ax[3, i].imshow(
                        model.label_cmap[
                            model.test_cluster_metrics.map_clusters(crf_outputs["no_crf_cluster_preds"][img_num])])
                if run_picie:
                    ax[4, i].imshow(model.label_cmap[crf_outputs["picie_preds"][img_num]])

            ax[0, 0].set_ylabel("Image", fontsize=26)
            ax[1, 0].set_ylabel("Label", fontsize=26)
            if cfg.run_prediction:
                ax[2, 0].set_ylabel("STEGO\n(Ours)", fontsize=26)
                ax[3, 0].set_ylabel("STEGO\nno CRF", fontsize=26)
            if run_picie:
                ax[4, 0].set_ylabel("PiCIE\n(Baseline)", fontsize=26)

            remove_axes(ax)
            plt.tight_layout()
            plt.show()
            plt.clf()

        plot_cm(model.test_cluster_metrics.histogram, model.label_cmap, model.cfg)
        plt.show()
        plt.clf()


if __name__ == "__main__":
    prep_args()
    my_app()
