from collections import defaultdict
import hydra
import torch.multiprocessing
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader
from data import *
from modules import *
from train_segmentation import LitUnsupervisedSegmenter


@hydra.main(config_path="configs", config_name="train_config.yml")
def my_app(cfg: DictConfig) -> None:
    # print(OmegaConf.to_yaml(cfg))
    pytorch_data_dir = cfg.pytorch_data_dir

    result_dir = "../results/predictions/potsdam"
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "img"), exist_ok=True)
    os.makedirs(join(result_dir, "label"), exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)

    full_dataset = ContrastiveSegDataset(
        pytorch_data_dir=pytorch_data_dir,
        dataset_name="potsdamraw",
        crop_type=None,
        image_set="all",
        transform=get_transform(320, False, "center"),
        target_transform=get_transform(320, True, "center"),
        cfg=cfg,
    )

    test_loader = DataLoader(full_dataset, 64,
                             shuffle=False, num_workers=cfg.num_workers,
                             pin_memory=True, collate_fn=flexible_collate)

    model = LitUnsupervisedSegmenter.load_from_checkpoint("../saved_models/potsdam_test.ckpt")
    print(OmegaConf.to_yaml(model.cfg))
    model.eval().cuda()
    par_model = torch.nn.DataParallel(model.net)

    outputs = defaultdict(list)
    for i, batch in enumerate(tqdm(test_loader)):
        with torch.no_grad():
            if i > 100:
                break

            img = batch["img"].cuda()
            label = batch["label"].cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, label.shape[-2:], mode='bilinear', align_corners=False)
            cluster_prob = model.cluster_probe(code, 2, log_probs=True)
            cluster_pred = cluster_prob.argmax(1)

            model.test_cluster_metrics.update(cluster_pred, label)

            outputs['img'].append(img.cpu())
            outputs['label'].append(label.cpu())
            outputs['cluster_pred'].append(cluster_pred.cpu())
            outputs['cluster_prob'].append(cluster_prob.cpu())
    model.test_cluster_metrics.compute()

    img_num = 6
    outputs = {k: torch.cat(v, dim=0)[15 * 15 * img_num:15 * 15 * (img_num + 1)] for k, v in outputs.items()}

    full_image = outputs['img'].reshape(15, 15, 3, 320, 320) \
        .permute(2, 0, 3, 1, 4) \
        .reshape(3, 320 * 15, 320 * 15)

    full_cluster_prob = outputs['cluster_prob'].reshape(15, 15, 3, 320, 320) \
        .permute(2, 0, 3, 1, 4) \
        .reshape(3, 320 * 15, 320 * 15)

    # crf_probs = dense_crf(full_image.cpu().detach(),
    #                       full_cluster_prob.cpu().detach())
    crf_probs = full_cluster_prob.numpy()
    print(crf_probs.shape)

    reshaped_label = outputs['label'].reshape(15, 15, 320, 320).permute(0, 2, 1, 3).reshape(320 * 15, 320 * 15)
    reshaped_img = unnorm(full_image).permute(1, 2, 0)
    reshaped_preds = model.test_cluster_metrics.map_clusters(np.expand_dims(crf_probs.argmax(0), 0))

    fig, ax = plt.subplots(1, 3, figsize=(4 * 3, 4))
    ax[0].imshow(reshaped_img)
    ax[1].imshow(reshaped_preds)
    ax[2].imshow(reshaped_label)

    Image.fromarray(reshaped_img.cuda()).save(join(join(result_dir, "img", str(img_num) + ".png")))
    Image.fromarray(reshaped_preds).save(join(join(result_dir, "cluster", str(img_num) + ".png")))

    remove_axes(ax)
    plt.show()

if __name__ == "__main__":
    prep_args()
    my_app()
