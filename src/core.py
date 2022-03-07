import os
import random
from os.path import join

import matplotlib.pyplot as plt
import numpy as np
import torch.multiprocessing
import torch.nn as nn
import torch.nn.functional as F
import wget
from PIL import Image
from scipy.io import loadmat
from scipy.optimize import linear_sum_assignment
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchmetrics import Metric
from torchvision import models
from torchvision import transforms as T
from torchvision.transforms.functional import resize, pad, to_pil_image
from tqdm import tqdm
from torchvision.datasets.cityscapes import Cityscapes
from torch.utils.data._utils.collate import np_str_obj_array_pattern, default_collate_err_msg_format
from torch._six import string_classes
import collections


def prep_for_plot(img, rescale=True, resize=None):
    if resize is not None:
        img = F.interpolate(img.unsqueeze(0), resize, mode="bilinear")
    else:
        img = img.unsqueeze(0)

    plot_img = unnorm(img).squeeze(0).cpu().permute(1, 2, 0)
    if rescale:
        plot_img = (plot_img - plot_img.min()) / (plot_img.max() - plot_img.min())
    return plot_img


def load_model(model_type, data_dir):
    if model_type == "robust_resnet50":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'imagenet_l2_3_0.pt')
        if not os.path.exists(model_file):
            wget.download("http://6.869.csail.mit.edu/fa19/psets19/pset6/imagenet_l2_3_0.pt",
                          model_file)
        model_weights = torch.load(model_file)
        model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
                                  'model' in name}
        model.load_state_dict(model_weights_modified)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densecl":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'densecl_r50_coco_1600ep.pth')
        if not os.path.exists(model_file):
            wget.download("https://cloudstor.aarnet.edu.au/plus/s/3GapXiWuVAzdKwJ/download",
                          model_file)
        model_weights = torch.load(model_file)
        # model_weights_modified = {name.split('model.')[1]: value for name, value in model_weights['model'].items() if
        #                          'model' in name}
        model.load_state_dict(model_weights['state_dict'], strict=False)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "resnet50":
        model = models.resnet50(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "mocov2":
        model = models.resnet50(pretrained=False)
        model_file = join(data_dir, 'moco_v2_800ep_pretrain.pth.tar')
        if not os.path.exists(model_file):
            wget.download("https://dl.fbaipublicfiles.com/moco/moco_checkpoints/"
                          "moco_v2_800ep/moco_v2_800ep_pretrain.pth.tar", model_file)
        checkpoint = torch.load(model_file)
        # rename moco pre-trained keys
        state_dict = checkpoint['state_dict']
        for k in list(state_dict.keys()):
            # retain only encoder_q up to before the embedding layer
            if k.startswith('module.encoder_q') and not k.startswith('module.encoder_q.fc'):
                # remove prefix
                state_dict[k[len("module.encoder_q."):]] = state_dict[k]
            # delete renamed or unused k
            del state_dict[k]
        msg = model.load_state_dict(state_dict, strict=False)
        assert set(msg.missing_keys) == {"fc.weight", "fc.bias"}
        model = nn.Sequential(*list(model.children())[:-1])
    elif model_type == "densenet121":
        model = models.densenet121(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    elif model_type == "vgg11":
        model = models.vgg11(pretrained=True)
        model = nn.Sequential(*list(model.children())[:-1] + [nn.AdaptiveAvgPool2d((1, 1))])
    else:
        raise ValueError("No model: {} found".format(model_type))

    model.eval()
    model.cuda()
    return model


class UnNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, image):
        image2 = torch.clone(image)
        for t, m, s in zip(image2, self.mean, self.std):
            t.mul_(s).add_(m)
        return image2


class MultiOptimizer(object):
    def __init__(self, *op):
        self.optimizers = op

    def zero_grad(self):
        for op in self.optimizers:
            op.zero_grad()

    def step(self):
        for op in self.optimizers:
            op.step()


normalize = T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
unnorm = UnNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])


class ToTargetTensor(object):
    def __call__(self, target):
        return torch.as_tensor(np.array(target), dtype=torch.int64).unsqueeze(0)


def prep_args():
    import sys

    old_args = sys.argv
    new_args = [old_args.pop(0)]
    while len(old_args) > 0:
        arg = old_args.pop(0)
        if len(arg.split("=")) == 2:
            new_args.append(arg)
        elif arg.startswith("--"):
            new_args.append(arg[2:] + "=" + old_args.pop(0))
        else:
            raise ValueError("Unexpected arg style {}".format(arg))
    sys.argv = new_args


def get_transform(res, is_label, crop_type):
    if crop_type == "center":
        cropper = T.CenterCrop(res)
    elif crop_type == "random":
        cropper = T.RandomCrop(res)
    elif crop_type is None:
        cropper = T.Lambda(lambda x: x)
        res = (res, res)
    else:
        raise ValueError("Unknown Cropper {}".format(crop_type))
    if is_label:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          ToTargetTensor()])
    else:
        return T.Compose([T.Resize(res, Image.NEAREST),
                          cropper,
                          T.ToTensor(),
                          normalize])


def _remove_axes(ax):
    ax.xaxis.set_major_formatter(plt.NullFormatter())
    ax.yaxis.set_major_formatter(plt.NullFormatter())
    ax.set_xticks([])
    ax.set_yticks([])


def remove_axes(axes):
    if len(axes.shape) == 2:
        for ax1 in axes:
            for ax in ax1:
                _remove_axes(ax)
    else:
        for ax in axes:
            _remove_axes(ax)


def bit_get(val, idx):
    """Gets the bit value.
    Args:
      val: Input value, int or numpy int array.
      idx: Which bit of the input val.
    Returns:
      The "idx"-th bit of input val.
    """
    return (val >> idx) & 1


def create_pascal_label_colormap():
    """Creates a label colormap used in PASCAL VOC segmentation benchmark.
    Returns:
      A colormap for visualizing segmentation results.
    """
    colormap = np.zeros((512, 3), dtype=int)
    ind = np.arange(512, dtype=int)

    for shift in reversed(list(range(8))):
        for channel in range(3):
            colormap[:, channel] |= bit_get(ind, channel) << shift
        ind >>= 3

    return colormap


def create_cityscapes_colormap():
    colors = [(128, 64, 128),
              (244, 35, 232),
              (250, 170, 160),
              (230, 150, 140),
              (70, 70, 70),
              (102, 102, 156),
              (190, 153, 153),
              (180, 165, 180),
              (150, 100, 100),
              (150, 120, 90),
              (153, 153, 153),
              (153, 153, 153),
              (250, 170, 30),
              (220, 220, 0),
              (107, 142, 35),
              (152, 251, 152),
              (70, 130, 180),
              (220, 20, 60),
              (255, 0, 0),
              (0, 0, 142),
              (0, 0, 70),
              (0, 60, 100),
              (0, 0, 90),
              (0, 0, 110),
              (0, 80, 100),
              (0, 0, 230),
              (119, 11, 32),
              (0, 0, 0)]
    return np.array(colors)


def load_im(path):
    with open(path, "rb") as f:
        image = Image.open(f).convert("RGB")
    return image


def confusion_matrix(preds, targets):
    n_classes = (torch.max(targets) + 1).item()
    assert (n_classes == torch.max(preds) + 1)
    counts = torch.zeros(n_classes, n_classes, dtype=torch.int64)
    for ci in range(n_classes):
        for cj in range(n_classes):
            counts[ci, cj] = torch.sum((preds == ci) * (targets == cj))
    return counts


def unsupervised_accuracy(preds, targets, masks):
    valid_areas = torch.where(masks)
    valid_targets = targets[valid_areas]
    valid_preds = preds[valid_areas]
    conf = confusion_matrix(valid_preds, valid_targets)
    assignments = linear_sum_assignment(conf, maximize=True)
    return conf[assignments[:, 0], assignments[:, 1]].sum() / conf.sum()


class UnsupervisedMetrics(Metric):
    def __init__(self, prefix: str, n_classes: int, extra_clusters: int, compute_hungarian: bool,
                 dist_sync_on_step=True):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.n_classes = n_classes
        self.extra_clusters = extra_clusters
        self.compute_hungarian = compute_hungarian
        self.prefix = prefix
        self.add_state("stats",
                       default=torch.zeros(n_classes + self.extra_clusters, n_classes, dtype=torch.int64),
                       dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        with torch.no_grad():
            actual = target.reshape(-1)
            preds = preds.reshape(-1)
            mask = (actual >= 0) & (actual < self.n_classes) & (preds >= 0) & (preds < self.n_classes)
            actual = actual[mask]
            preds = preds[mask]
            self.stats += torch.bincount(
                (self.n_classes + self.extra_clusters) * actual + preds,
                minlength=self.n_classes * (self.n_classes + self.extra_clusters)) \
                .reshape(self.n_classes, self.n_classes + self.extra_clusters).t().to(self.stats.device)

    def map_clusters(self, clusters):
        if self.extra_clusters == 0:
            return torch.tensor(self.assignments[1])[clusters]
        else:
            missing = sorted(list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0])))
            cluster_to_class = self.assignments[1]
            for missing_entry in missing:
                if missing_entry == cluster_to_class.shape[0]:
                    cluster_to_class = np.append(cluster_to_class, -1)
                else:
                    cluster_to_class = np.insert(cluster_to_class, missing_entry + 1, -1)
            cluster_to_class = torch.tensor(cluster_to_class)
            return cluster_to_class[clusters]

    def compute(self):
        if self.compute_hungarian:
            self.assignments = linear_sum_assignment(self.stats.detach().cpu(), maximize=True)
            # print(self.assignments)
            if self.extra_clusters == 0:
                self.histogram = self.stats[np.argsort(self.assignments[1]), :]
            if self.extra_clusters > 0:
                self.assignments_t = linear_sum_assignment(self.stats.detach().cpu().t(), maximize=True)
                histogram = self.stats[self.assignments_t[1], :]
                missing = list(set(range(self.n_classes + self.extra_clusters)) - set(self.assignments[0]))
                new_row = self.stats[missing, :].sum(0, keepdim=True)
                histogram = torch.cat([histogram, new_row], axis=0)
                new_col = torch.zeros(self.n_classes + 1, 1, device=histogram.device)
                self.histogram = torch.cat([histogram, new_col], axis=1)
        else:
            self.assignments = (torch.arange(self.n_classes).unsqueeze(1),
                                torch.arange(self.n_classes).unsqueeze(1))
            self.histogram = self.stats

        tp = torch.diag(self.histogram)
        fp = torch.sum(self.histogram, dim=0) - tp
        fn = torch.sum(self.histogram, dim=1) - tp

        iou = tp / (tp + fp + fn)
        prc = tp / (tp + fn)
        opc = torch.sum(tp) / torch.sum(self.histogram)

        metric_dict = {self.prefix + "mIoU": iou[~torch.isnan(iou)].mean().item(),
                       self.prefix + "Accuracy": opc.item()}
        return {k: 100 * v for k, v in metric_dict.items()}


def rescale(img, scale_factor, interpolation):
    target_size = [int(img.shape[1] * scale_factor), int(img.shape[2] * scale_factor)]
    return resize(img, target_size, interpolation=interpolation)


def pad_to_size(img, size):
    if img.shape[1] < size:
        p = (size - img.shape[1]) // 2 + 1
        img = pad(img, [0, p, 0, p])
    if img.shape[2] < size:
        p = (size - img.shape[2]) // 2 + 1
        img = pad(img, [p, 0, p, 0])
    return img


class Potsdam(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(Potsdam, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdam")
        self.transform = transform
        self.target_transform = target_transform
        split_files = {
            "train": ["labelled_train.txt"],
            "unlabelled_train": ["unlabelled_train.txt"],
            # "train": ["unlabelled_train.txt"],
            "val": ["labelled_test.txt"],
            "train+val": ["labelled_train.txt", "labelled_test.txt"],
            "all": ["all.txt"]
        }
        assert self.split in split_files.keys()

        self.files = []
        for split_file in split_files[self.split]:
            with open(join(self.root, split_file), "r") as f:
                self.files.extend(fn.rstrip() for fn in f.readlines())

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id + ".mat"))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id + ".mat"))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class PotsdamRaw(Dataset):
    def __init__(self, root, image_set, transform, target_transform, coarse_labels):
        super(PotsdamRaw, self).__init__()
        self.split = image_set
        self.root = os.path.join(root, "potsdamraw", "processed")
        self.transform = transform
        self.target_transform = target_transform
        self.files = []
        for im_num in range(38):
            for i_h in range(15):
                for i_w in range(15):
                    self.files.append("{}_{}_{}.mat".format(im_num, i_h, i_w))

        self.coarse_labels = coarse_labels
        self.fine_to_coarse = {0: 0, 4: 0,  # roads and cars
                               1: 1, 5: 1,  # buildings and clutter
                               2: 2, 3: 2,  # vegetation and trees
                               255: -1
                               }

    def __getitem__(self, index):
        image_id = self.files[index]
        img = loadmat(join(self.root, "imgs", image_id))["img"]
        img = to_pil_image(torch.from_numpy(img).permute(2, 0, 1)[:3])  # TODO add ir channel back
        try:
            label = loadmat(join(self.root, "gt", image_id))["gt"]
            label = to_pil_image(torch.from_numpy(label).unsqueeze(-1).permute(2, 0, 1))
        except FileNotFoundError:
            label = to_pil_image(torch.ones(1, img.height, img.width))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(img)

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.target_transform(label).squeeze(0)
        if self.coarse_labels:
            new_label_map = torch.zeros_like(label)
            for fine, coarse in self.fine_to_coarse.items():
                new_label_map[label == fine] = coarse
            label = new_label_map

        mask = (label > 0).to(torch.float32)
        return img, label, mask

    def __len__(self):
        return len(self.files)


class Coco(Dataset):
    def __init__(self, root, image_set, transform, target_transform,
                 coarse_labels, exclude_things, subset=None):
        super(Coco, self).__init__()
        self.split = image_set
        self.root = join(root, "cocostuff")
        self.coarse_labels = coarse_labels
        self.transform = transform
        self.label_transform = target_transform
        self.subset = subset
        self.exclude_things = exclude_things

        if self.subset is None:
            self.image_list = "Coco164kFull_Stuff_Coarse.txt"
        elif self.subset == 6:  # IIC Coarse
            self.image_list = "Coco164kFew_Stuff_6.txt"
        elif self.subset == 7:  # IIC Fine
            self.image_list = "Coco164kFull_Stuff_Coarse_7.txt"

        assert self.split in ["train", "val", "train+val"]
        split_dirs = {
            "train": ["train2017"],
            "val": ["val2017"],
            "train+val": ["train2017", "val2017"]
        }

        self.image_files = []
        self.label_files = []
        for split_dir in split_dirs[self.split]:
            with open(join(self.root, "curated", split_dir, self.image_list), "r") as f:
                img_ids = [fn.rstrip() for fn in f.readlines()]
                for img_id in img_ids:
                    self.image_files.append(join(self.root, "images", split_dir, img_id + ".jpg"))
                    self.label_files.append(join(self.root, "annotations", split_dir, img_id + ".png"))

        self.fine_to_coarse = {0: 9, 1: 11, 2: 11, 3: 11, 4: 11, 5: 11, 6: 11, 7: 11, 8: 11, 9: 8, 10: 8, 11: 8, 12: 8,
                               13: 8, 14: 8, 15: 7, 16: 7, 17: 7, 18: 7, 19: 7, 20: 7, 21: 7, 22: 7, 23: 7, 24: 7,
                               25: 6, 26: 6, 27: 6, 28: 6, 29: 6, 30: 6, 31: 6, 32: 6, 33: 10, 34: 10, 35: 10, 36: 10,
                               37: 10, 38: 10, 39: 10, 40: 10, 41: 10, 42: 10, 43: 5, 44: 5, 45: 5, 46: 5, 47: 5, 48: 5,
                               49: 5, 50: 5, 51: 2, 52: 2, 53: 2, 54: 2, 55: 2, 56: 2, 57: 2, 58: 2, 59: 2, 60: 2,
                               61: 3, 62: 3, 63: 3, 64: 3, 65: 3, 66: 3, 67: 3, 68: 3, 69: 3, 70: 3, 71: 0, 72: 0,
                               73: 0, 74: 0, 75: 0, 76: 0, 77: 1, 78: 1, 79: 1, 80: 1, 81: 1, 82: 1, 83: 4, 84: 4,
                               85: 4, 86: 4, 87: 4, 88: 4, 89: 4, 90: 4, 91: 17, 92: 17, 93: 22, 94: 20, 95: 20, 96: 22,
                               97: 15, 98: 25, 99: 16, 100: 13, 101: 12, 102: 12, 103: 17, 104: 17, 105: 23, 106: 15,
                               107: 15, 108: 17, 109: 15, 110: 21, 111: 15, 112: 25, 113: 13, 114: 13, 115: 13, 116: 13,
                               117: 13, 118: 22, 119: 26, 120: 14, 121: 14, 122: 15, 123: 22, 124: 21, 125: 21, 126: 24,
                               127: 20, 128: 22, 129: 15, 130: 17, 131: 16, 132: 15, 133: 22, 134: 24, 135: 21, 136: 17,
                               137: 25, 138: 16, 139: 21, 140: 17, 141: 22, 142: 16, 143: 21, 144: 21, 145: 25, 146: 21,
                               147: 26, 148: 21, 149: 24, 150: 20, 151: 17, 152: 14, 153: 21, 154: 26, 155: 15, 156: 23,
                               157: 20, 158: 21, 159: 24, 160: 15, 161: 24, 162: 22, 163: 25, 164: 15, 165: 20, 166: 17,
                               167: 17, 168: 22, 169: 14, 170: 18, 171: 18, 172: 18, 173: 18, 174: 18, 175: 18, 176: 18,
                               177: 26, 178: 26, 179: 19, 180: 19, 181: 24}

        self._label_names = [
            "ground-stuff",
            "plant-stuff",
            "sky-stuff",
        ]
        self.cocostuff3_coarse_classes = [23, 22, 21]
        self.first_stuff_index = 12

    def __getitem__(self, index):
        image_path = self.image_files[index]
        label_path = self.label_files[index]
        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        img = self.transform(Image.open(image_path).convert("RGB"))

        random.seed(seed)
        torch.manual_seed(seed)
        label = self.label_transform(Image.open(label_path)).squeeze(0)
        label[label == 255] = -1  # to be consistent with 10k
        coarse_label = torch.zeros_like(label)
        for fine, coarse in self.fine_to_coarse.items():
            coarse_label[label == fine] = coarse
        coarse_label[label == -1] = -1

        if self.coarse_labels:
            coarser_labels = -torch.ones_like(label)
            for i, c in enumerate(self.cocostuff3_coarse_classes):
                coarser_labels[coarse_label == c] = i
            return img, coarser_labels, coarser_labels >= 0
        else:
            if self.exclude_things:
                return img, coarse_label - self.first_stuff_index, (coarse_label >= self.first_stuff_index)
            else:
                return img, coarse_label, coarse_label >= 0

    def __len__(self):
        return len(self.image_files)


class CityscapesSeg(Dataset):
    def __init__(self, root, image_set, transform, target_transform):
        super(CityscapesSeg, self).__init__()
        self.split = image_set
        self.root = join(root, "cityscapes")
        if image_set == "train":
            # our_image_set = "train_extra"
            # mode = "coarse"
            our_image_set = "train"
            mode = "fine"
        else:
            our_image_set = image_set
            mode = "fine"
        self.inner_loader = Cityscapes(self.root, our_image_set,
                                       mode=mode,
                                       target_type="semantic",
                                       transform=None,
                                       target_transform=None)
        self.transform = transform
        self.target_transform = target_transform
        self.first_nonvoid = 7

    def __getitem__(self, index):
        if self.transform is not None:
            image, target = self.inner_loader[index]

            seed = np.random.randint(2147483647)
            random.seed(seed)
            torch.manual_seed(seed)
            image = self.transform(image)
            random.seed(seed)
            torch.manual_seed(seed)
            target = self.target_transform(target)

            target = target - self.first_nonvoid
            target[target < 0] = -1
            mask = target == -1
            return image, target.squeeze(0), mask
        else:
            return self.inner_loader[index]

    def __len__(self):
        return len(self.inner_loader)


class CroppedDataset(Dataset):
    def __init__(self, root, dataset_name, crop_type, crop_ratio, image_set, transform, target_transform):
        super(CroppedDataset, self).__init__()
        self.dataset_name = dataset_name
        self.split = image_set
        self.root = join(root, "cropped", "{}_{}_crop_{}".format(dataset_name, crop_type, crop_ratio))
        self.transform = transform
        self.target_transform = target_transform
        self.img_dir = join(self.root, "img", self.split)
        self.label_dir = join(self.root, "label", self.split)
        self.num_images = len(os.listdir(self.img_dir))
        assert self.num_images == len(os.listdir(self.label_dir))

    def __getitem__(self, index):
        image = Image.open(join(self.img_dir, "{}.jpg".format(index))).convert('RGB')
        target = Image.open(join(self.label_dir, "{}.png".format(index)))

        seed = np.random.randint(2147483647)
        random.seed(seed)
        torch.manual_seed(seed)
        image = self.transform(image)
        random.seed(seed)
        torch.manual_seed(seed)
        target = self.target_transform(target)

        target = target - 1
        mask = target == -1
        return image, target.squeeze(0), mask

    def __len__(self):
        return self.num_images


class MaterializedDataset(Dataset):

    def __init__(self, ds):
        self.ds = ds
        self.materialized = []
        loader = DataLoader(ds, num_workers=12, collate_fn=lambda l: l[0])
        for batch in tqdm(loader):
            self.materialized.append(batch)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, ind):
        return self.materialized[ind]


class ContrastiveSegDataset(Dataset):
    def __init__(self,
                 pytorch_data_dir,
                 dataset_name,
                 crop_type,
                 image_set,
                 transform,
                 target_transform,
                 cfg,
                 aug_geometric_transform=None,
                 aug_photometric_transform=None,
                 num_neighbors=5,
                 compute_knns=False,
                 mask=False,
                 pos_labels=False,
                 pos_images=False,
                 extra_transform=None,
                 model_type_override=None
                 ):
        super(ContrastiveSegDataset).__init__()
        self.num_neighbors = num_neighbors
        self.image_set = image_set
        self.dataset_name = dataset_name
        self.mask = mask
        self.pos_labels = pos_labels
        self.pos_images = pos_images
        self.extra_transform = extra_transform

        if dataset_name == "potsdam":
            self.n_classes = 3
            dataset_class = Potsdam
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "potsdamraw":
            self.n_classes = 3
            dataset_class = PotsdamRaw
            extra_args = dict(coarse_labels=True)
        elif dataset_name == "cityscapes" and crop_type is None:
            self.n_classes = 27
            dataset_class = CityscapesSeg
            extra_args = dict()
        elif dataset_name == "cityscapes" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cityscapes", crop_type=crop_type, crop_ratio=cfg.crop_ratio)
        elif dataset_name == "cocostuff3":
            self.n_classes = 3
            dataset_class = Coco
            extra_args = dict(coarse_labels=True, subset=6, exclude_things=True)
        elif dataset_name == "cocostuff15":
            self.n_classes = 15
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=7, exclude_things=True)
        elif dataset_name == "cocostuff27" and crop_type is not None:
            self.n_classes = 27
            dataset_class = CroppedDataset
            extra_args = dict(dataset_name="cocostuff27", crop_type=cfg.crop_type, crop_ratio=cfg.crop_ratio)
        elif dataset_name == "cocostuff27" and crop_type is None:
            self.n_classes = 27
            dataset_class = Coco
            extra_args = dict(coarse_labels=False, subset=None, exclude_things=False)
            if image_set == "val":
                extra_args["subset"] = 7
        else:
            raise ValueError("Unknown dataset: {}".format(dataset_name))

        self.aug_geometric_transform = aug_geometric_transform
        self.aug_photometric_transform = aug_photometric_transform

        self.dataset = dataset_class(
            root=pytorch_data_dir,
            image_set=self.image_set,
            transform=transform,
            target_transform=target_transform, **extra_args)

        if model_type_override is not None:
            model_type = model_type_override
        else:
            model_type = cfg.model_type

        feature_cache_file = join(pytorch_data_dir, "nns", "nns_{}_{}_{}_{}_{}.npz".format(
            model_type, dataset_name, image_set, crop_type, cfg.res))
        if pos_labels or pos_images:
            if not os.path.exists(feature_cache_file) or compute_knns:
                raise ValueError("could not find nn file {} please run precompute_knns".format(feature_cache_file))
            else:
                loaded = np.load(feature_cache_file)
                self.nns = loaded["nns"]
            assert len(self.dataset) == self.nns.shape[0]

    def __len__(self):
        return len(self.dataset)

    def _set_seed(self, seed):
        random.seed(seed)  # apply this seed to img tranfsorms
        torch.manual_seed(seed)  # needed for torchvision 0.7

    def __getitem__(self, ind):
        pack = self.dataset[ind]

        if self.pos_images or self.pos_labels:
            ind_pos = self.nns[ind][torch.randint(low=1, high=self.num_neighbors + 1, size=[]).item()]
            pack_pos = self.dataset[ind_pos]

        seed = np.random.randint(2147483647)  # make a seed with numpy generator

        self._set_seed(seed)
        coord_entries = torch.meshgrid([torch.linspace(-1, 1, pack[0].shape[1]),
                                        torch.linspace(-1, 1, pack[0].shape[2])])
        coord = torch.cat([t.unsqueeze(0) for t in coord_entries], 0)

        if self.extra_transform is not None:
            extra_trans = self.extra_transform
        else:
            extra_trans = lambda i, x: x

        ret = {
            "ind": ind,
            "img": extra_trans(ind, pack[0]),
            "label": extra_trans(ind, pack[1]),
        }

        if self.pos_images:
            ret["img_pos"] = extra_trans(ind, pack_pos[0])
            ret["ind_pos"] = ind_pos

        if self.mask:
            ret["mask"] = pack[2]

        if self.pos_labels:
            ret["label_pos"] = extra_trans(ind, pack_pos[1])
            ret["mask_pos"] = pack_pos[2]

        if self.aug_photometric_transform is not None:
            img_aug = self.aug_photometric_transform(self.aug_geometric_transform(pack[0]))

            self._set_seed(seed)
            coord_aug = self.aug_geometric_transform(coord)

            ret["img_aug"] = img_aug
            ret["coord_aug"] = coord_aug.permute(1, 2, 0)

        return ret


def yuv_to_rgb(image: torch.Tensor) -> torch.Tensor:
    if len(image.shape) < 3 or image.shape[-3] != 3:
        raise ValueError("Input size must have a shape of (*, 3, H, W). Got {}"
                         .format(image.shape))

    t = torch.tensor([
        [1, 0, 1.13983],
        [1, -0.39465, -0.58060],
        [1, 2.03211, 0]
    ])

    return torch.einsum("ry,byhw->brhw", t, image)


def coord_view(coords):
    yuv = F.pad(input=coords.unsqueeze(0), pad=(0, 0, 0, 0, 1, 0), mode='constant', value=1) \
              .detach().cpu() / 2
    return yuv_to_rgb(yuv).squeeze(0).permute(1, 2, 0)


def entropy(p):
    p = torch.clamp_min(p, .0000001)
    return -(p * torch.log(p)).sum(dim=1)




def flexible_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        try:
            return torch.stack(batch, 0, out=out)
        except RuntimeError:
            return batch
    elif elem_type.__module__ == 'numpy' and elem_type.__name__ != 'str_' \
            and elem_type.__name__ != 'string_':
        if elem_type.__name__ == 'ndarray' or elem_type.__name__ == 'memmap':
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return flexible_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: flexible_collate([d[key] for d in batch]) for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, '_fields'):  # namedtuple
        return elem_type(*(flexible_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError('each element in list of batch should be of equal size')
        transposed = zip(*batch)
        return [flexible_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
