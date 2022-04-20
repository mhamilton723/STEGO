from modules import *
import hydra
import torch.multiprocessing
from PIL import Image
from crf import dense_crf
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader, Dataset
from train_segmentation import LitUnsupervisedSegmenter
from tqdm import tqdm
import random
torch.multiprocessing.set_sharing_strategy('file_system')


class UnlabeledImageFolder(Dataset):
    def __init__(self, root, transform):
        super(UnlabeledImageFolder, self).__init__()
        self.root = join(root)
        self.transform = transform
        self.images = os.listdir(self.root)

    def __getitem__(self, index):
        image = Image.open(join(self.root, self.images[index])).convert('RGB')
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)

        return image, self.images[index]

    def __len__(self):
        return len(self.images)


@hydra.main(config_path="configs", config_name="demo_config.yml")
def my_app(cfg: DictConfig) -> None:
    result_dir = "../results/predictions/{}".format(cfg.experiment_name)
    os.makedirs(result_dir, exist_ok=True)
    os.makedirs(join(result_dir, "cluster"), exist_ok=True)
    os.makedirs(join(result_dir, "linear"), exist_ok=True)

    model = LitUnsupervisedSegmenter.load_from_checkpoint(cfg.model_path)
    print(OmegaConf.to_yaml(model.cfg))

    dataset = UnlabeledImageFolder(
        root=cfg.image_dir,
        transform=get_transform(cfg.res, False, "center"),
    )

    loader = DataLoader(dataset, cfg.batch_size * 2,
                        shuffle=False, num_workers=cfg.num_workers,
                        pin_memory=True, collate_fn=flexible_collate)

    model.eval().cuda()
    if cfg.use_ddp:
        par_model = torch.nn.DataParallel(model.net)
    else:
        par_model = model.net

    for i, (img, name) in enumerate(tqdm(loader)):
        with torch.no_grad():
            img = img.cuda()
            feats, code1 = par_model(img)
            feats, code2 = par_model(img.flip(dims=[3]))
            code = (code1 + code2.flip(dims=[3])) / 2

            code = F.interpolate(code, img.shape[-2:], mode='bilinear', align_corners=False)

            linear_probs = torch.log_softmax(model.linear_probe(code), dim=1).cpu()
            cluster_probs = model.cluster_probe(code, 2, log_probs=True).cpu()

            for j in range(img.shape[0]):
                single_img = img[j].cpu()
                linear_crf = dense_crf(single_img, linear_probs[j]).argmax(0)
                cluster_crf = dense_crf(single_img, cluster_probs[j]).argmax(0)

                new_name = ".".join(name[j].split(".")[:-1]) + ".png"
                Image.fromarray(linear_crf.astype(np.uint8)).save(join(result_dir, "linear", new_name))
                Image.fromarray(cluster_crf.astype(np.uint8)).save(join(result_dir, "cluster", new_name))


if __name__ == "__main__":
    prep_args()
    my_app()
