from modules import *
from data import *
from collections import defaultdict
from multiprocessing import Pool
import hydra
import seaborn as sns
import torch.multiprocessing
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from tqdm import tqdm
from train_segmentation import LitUnsupervisedSegmenter, prep_for_plot, get_class_labels

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


def _apply_crf(tup):
    return dense_crf(tup[0], tup[1])


def batched_crf(pool, img_tensor, prob_tensor):
    outputs = pool.map(_apply_crf, zip(img_tensor.detach().cpu(), prob_tensor.detach().cpu()))
    return torch.cat([torch.from_numpy(arr).unsqueeze(0) for arr in outputs], dim=0)


@hydra.main(config_path="configs", config_name="eval_config.yml")
def my_app(cfg: DictConfig) -> None:
    pytorch_data_dir = cfg.pytorch_data_dir
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "picie"), exist_ok=True)

    for model_path in cfg.model_paths:
        model = LitUnsupervisedSegmenter.load_from_checkpoint(model_path)
        print(OmegaConf.to_yaml(model.cfg))

        run_picie = cfg.run_picie and model.cfg.dataset_name == "cocostuff27"
        if run_picie:
            picie_state = torch.load("../saved_models/picie_and_probes.pth")
            picie = picie_state["model"].cuda()
            picie_cluster_probe = picie_state["cluster_probe"].module.cuda()
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

        model.eval().cuda()

        if cfg.use_ddp:
            par_model = torch.nn.DataParallel(model.net)
            if run_picie:
                par_picie = torch.nn.DataParallel(picie)
        else:
            par_model = model.net
            if run_picie:
                par_picie = picie

        if model.cfg.dataset_name == "cocostuff27":
            # all_good_images = range(10)
            # all_good_images = range(250)
            # all_good_images = [61, 60, 49, 44, 13, 70] #Failure cases
            all_good_images = [19, 54, 67, 66, 65, 75, 77, 76, 124]  # Main figure
        elif model.cfg.dataset_name == "cityscapes":
            # all_good_images = range(80)
            # all_good_images = [ 5, 20, 56]
            all_good_images = [11, 32, 43, 52]
        else:
            raise ValueError("Unknown Dataset {}".format(model.cfg.dataset_name))
        batch_nums = torch.tensor([n // (cfg.batch_size * 2) for n in all_good_images])
        batch_offsets = torch.tensor([n % (cfg.batch_size * 2) for n in all_good_images])

        saved_data = defaultdict(list)
        with Pool(cfg.num_workers + 5) as pool:
            for i, batch in enumerate(tqdm(test_loader)):
                with torch.no_grad():
                    img = batch["img"].cuda()
                    label = batch["label"].cuda()

                    feats, code1 = par_model(img)
                    feats, code2 = par_model(img.flip(dims=[3]))
                    code = (code1 + code2.flip(dims=[3])) / 2

                    code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)

                    linear_probs = torch.log_softmax(model.linear_probe(code), dim=1)
                    cluster_probs = model.cluster_probe(code, 2, log_probs=True)

                    if cfg.run_crf:
                        linear_preds = batched_crf(pool, img, linear_probs).argmax(1).cuda()
                        cluster_preds = batched_crf(pool, img, cluster_probs).argmax(1).cuda()
                    else:
                        linear_preds = linear_probs.argmax(1)
                        cluster_preds = cluster_probs.argmax(1)

                    model.test_linear_metrics.update(linear_preds, label)
                    model.test_cluster_metrics.update(cluster_preds, label)

                    if run_picie:
                        picie_preds = picie_cluster_metrics.map_clusters(
                            picie_cluster_probe(par_picie(img), None)[1].argmax(1).cpu())

                    if i in batch_nums:
                        matching_offsets = batch_offsets[torch.where(batch_nums == i)]
                        for offset in matching_offsets:
                            saved_data["linear_preds"].append(linear_preds.cpu()[offset].unsqueeze(0))
                            saved_data["cluster_preds"].append(cluster_preds.cpu()[offset].unsqueeze(0))
                            saved_data["label"].append(label.cpu()[offset].unsqueeze(0))
                            saved_data["img"].append(img.cpu()[offset].unsqueeze(0))
                            if run_picie:
                                saved_data["picie_preds"].append(picie_preds.cpu()[offset].unsqueeze(0))
        saved_data = {k: torch.cat(v, dim=0) for k, v in saved_data.items()}

        tb_metrics = {
            **model.test_linear_metrics.compute(),
            **model.test_cluster_metrics.compute(),
        }

        print("")
        print(model_path)
        print(tb_metrics)

        if cfg.run_prediction:
            n_rows = 3
        else:
            n_rows = 2

        if run_picie:
            n_rows += 1

        if cfg.dark_mode:
            plt.style.use('dark_background')

        for good_images in batch_list(range(len(all_good_images)), 10):
            fig, ax = plt.subplots(n_rows, len(good_images), figsize=(len(good_images) * 3, n_rows * 3))
            for i, img_num in enumerate(good_images):
                plot_img = (prep_for_plot(saved_data["img"][img_num]) * 255).numpy().astype(np.uint8)
                plot_label = (model.label_cmap[saved_data["label"][img_num]]).astype(np.uint8)
                Image.fromarray(plot_img).save(join(join(result_dir, "img", str(img_num) + ".jpg")))
                Image.fromarray(plot_label).save(join(join(result_dir, "label", str(img_num) + ".png")))

                ax[0, i].imshow(plot_img)
                ax[1, i].imshow(plot_label)
                if cfg.run_prediction:
                    plot_cluster = (model.label_cmap[
                        model.test_cluster_metrics.map_clusters(
                            saved_data["cluster_preds"][img_num])]) \
                        .astype(np.uint8)
                    Image.fromarray(plot_cluster).save(join(join(result_dir, "cluster", str(img_num) + ".png")))
                    ax[2, i].imshow(plot_cluster)
                if run_picie:
                    picie_img = model.label_cmap[saved_data["picie_preds"][img_num]].astype(np.uint8)
                    ax[3, i].imshow(picie_img)
                    Image.fromarray(picie_img).save(join(join(result_dir, "picie", str(img_num) + ".png")))

            ax[0, 0].set_ylabel("Image", fontsize=26)
            ax[1, 0].set_ylabel("Label", fontsize=26)
            if cfg.run_prediction:
                ax[2, 0].set_ylabel("STEGO\n(Ours)", fontsize=26)
            if run_picie:
                ax[3, 0].set_ylabel("PiCIE\n(Baseline)", fontsize=26)

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
