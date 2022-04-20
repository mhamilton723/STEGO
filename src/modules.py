import torch

from utils import *
import torch.nn.functional as F
import dino.vision_transformer as vits


class LambdaLayer(nn.Module):
    def __init__(self, lambd):
        super(LambdaLayer, self).__init__()
        self.lambd = lambd

    def forward(self, x):
        return self.lambd(x)


class DinoFeaturizer(nn.Module):

    def __init__(self, dim, cfg):
        super().__init__()
        self.cfg = cfg
        self.dim = dim
        patch_size = self.cfg.dino_patch_size
        self.patch_size = patch_size
        self.feat_type = self.cfg.dino_feat_type
        arch = self.cfg.model_type
        self.model = vits.__dict__[arch](
            patch_size=patch_size,
            num_classes=0)
        for p in self.model.parameters():
            p.requires_grad = False
        self.model.eval().cuda()
        self.dropout = torch.nn.Dropout2d(p=.1)

        if arch == "vit_small" and patch_size == 16:
            url = "dino_deitsmall16_pretrain/dino_deitsmall16_pretrain.pth"
        elif arch == "vit_small" and patch_size == 8:
            url = "dino_deitsmall8_300ep_pretrain/dino_deitsmall8_300ep_pretrain.pth"
        elif arch == "vit_base" and patch_size == 16:
            url = "dino_vitbase16_pretrain/dino_vitbase16_pretrain.pth"
        elif arch == "vit_base" and patch_size == 8:
            url = "dino_vitbase8_pretrain/dino_vitbase8_pretrain.pth"
        else:
            raise ValueError("Unknown arch and patch size")

        if cfg.pretrained_weights is not None:
            state_dict = torch.load(cfg.pretrained_weights, map_location="cpu")
            state_dict = state_dict["teacher"]
            # remove `module.` prefix
            state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
            # remove `backbone.` prefix induced by multicrop wrapper
            state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

            # state_dict = {k.replace("projection_head", "mlp"): v for k, v in state_dict.items()}
            # state_dict = {k.replace("prototypes", "last_layer"): v for k, v in state_dict.items()}

            msg = self.model.load_state_dict(state_dict, strict=False)
            print('Pretrained weights found at {} and loaded with msg: {}'.format(cfg.pretrained_weights, msg))
        else:
            print("Since no pretrained weights have been provided, we load the reference pretrained DINO weights.")
            state_dict = torch.hub.load_state_dict_from_url(url="https://dl.fbaipublicfiles.com/dino/" + url)
            self.model.load_state_dict(state_dict, strict=True)

        if arch == "vit_small":
            self.n_feats = 384
        else:
            self.n_feats = 768
        self.cluster1 = self.make_clusterer(self.n_feats)
        self.proj_type = cfg.projection_type
        if self.proj_type == "nonlinear":
            self.cluster2 = self.make_nonlinear_clusterer(self.n_feats)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))  # ,

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)))

    def forward(self, img, n=1, return_class_feat=False):
        self.model.eval()
        with torch.no_grad():
            assert (img.shape[2] % self.patch_size == 0)
            assert (img.shape[3] % self.patch_size == 0)

            # get selected layer activations
            feat, attn, qkv = self.model.get_intermediate_feat(img, n=n)
            feat, attn, qkv = feat[0], attn[0], qkv[0]

            feat_h = img.shape[2] // self.patch_size
            feat_w = img.shape[3] // self.patch_size

            if self.feat_type == "feat":
                image_feat = feat[:, 1:, :].reshape(feat.shape[0], feat_h, feat_w, -1).permute(0, 3, 1, 2)
            elif self.feat_type == "KK":
                image_k = qkv[1, :, :, 1:, :].reshape(feat.shape[0], 6, feat_h, feat_w, -1)
                B, H, I, J, D = image_k.shape
                image_feat = image_k.permute(0, 1, 4, 2, 3).reshape(B, H * D, I, J)
            else:
                raise ValueError("Unknown feat type:{}".format(self.feat_type))

            if return_class_feat:
                return feat[:, :1, :].reshape(feat.shape[0], 1, 1, -1).permute(0, 3, 1, 2)

        if self.proj_type is not None:
            code = self.cluster1(self.dropout(image_feat))
            if self.proj_type == "nonlinear":
                code += self.cluster2(self.dropout(image_feat))
        else:
            code = image_feat

        if self.cfg.dropout:
            return self.dropout(image_feat), code
        else:
            return image_feat, code


class ResizeAndClassify(nn.Module):

    def __init__(self, dim: int, size: int, n_classes: int):
        super(ResizeAndClassify, self).__init__()
        self.size = size
        self.predictor = torch.nn.Sequential(
            torch.nn.Conv2d(dim, n_classes, (1, 1)),
            torch.nn.LogSoftmax(1))

    def forward(self, x):
        return F.interpolate(self.predictor.forward(x), self.size, mode="bilinear", align_corners=False)


class ClusterLookup(nn.Module):

    def __init__(self, dim: int, n_classes: int):
        super(ClusterLookup, self).__init__()
        self.n_classes = n_classes
        self.dim = dim
        self.clusters = torch.nn.Parameter(torch.randn(n_classes, dim))

    def reset_parameters(self):
        with torch.no_grad():
            self.clusters.copy_(torch.randn(self.n_classes, self.dim))

    def forward(self, x, alpha, log_probs=False):
        normed_clusters = F.normalize(self.clusters, dim=1)
        normed_features = F.normalize(x, dim=1)
        inner_products = torch.einsum("bchw,nc->bnhw", normed_features, normed_clusters)

        if alpha is None:
            cluster_probs = F.one_hot(torch.argmax(inner_products, dim=1), self.clusters.shape[0]) \
                .permute(0, 3, 1, 2).to(torch.float32)
        else:
            cluster_probs = nn.functional.softmax(inner_products * alpha, dim=1)

        cluster_loss = -(cluster_probs * inner_products).sum(1).mean()
        if log_probs:
            return nn.functional.log_softmax(inner_products * alpha, dim=1)
        else:
            return cluster_loss, cluster_probs


class FeaturePyramidNet(nn.Module):

    @staticmethod
    def _helper(x):
        # TODO remove this hard coded 56
        return F.interpolate(x, 56, mode="bilinear", align_corners=False).unsqueeze(-1)

    def make_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def make_nonlinear_clusterer(self, in_channels):
        return torch.nn.Sequential(
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, in_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(in_channels, self.dim, (1, 1)),
            LambdaLayer(FeaturePyramidNet._helper))

    def __init__(self, granularity, cut_model, dim, continuous):
        super(FeaturePyramidNet, self).__init__()
        self.layer_nums = [5, 6, 7]
        self.spatial_resolutions = [7, 14, 28, 56]
        self.feat_channels = [2048, 1024, 512, 3]
        self.extra_channels = [128, 64, 32, 32]
        self.granularity = granularity
        self.encoder = NetWithActivations(cut_model, self.layer_nums)
        self.dim = dim
        self.continuous = continuous
        self.n_feats = self.dim

        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False)

        assert granularity in {1, 2, 3, 4}
        self.cluster1 = self.make_clusterer(self.feat_channels[0])
        self.cluster1_nl = self.make_nonlinear_clusterer(self.feat_channels[0])

        if granularity >= 2:
            # self.conv1 = DoubleConv(self.feat_channels[0], self.extra_channels[0])
            # self.conv2 = DoubleConv(self.extra_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.conv2 = DoubleConv(self.feat_channels[0] + self.feat_channels[1], self.extra_channels[1])
            self.cluster2 = self.make_clusterer(self.extra_channels[1])
        if granularity >= 3:
            self.conv3 = DoubleConv(self.extra_channels[1] + self.feat_channels[2], self.extra_channels[2])
            self.cluster3 = self.make_clusterer(self.extra_channels[2])
        if granularity >= 4:
            self.conv4 = DoubleConv(self.extra_channels[2] + self.feat_channels[3], self.extra_channels[3])
            self.cluster4 = self.make_clusterer(self.extra_channels[3])

    def c(self, x, y):
        return torch.cat([x, y], dim=1)

    def forward(self, x):
        with torch.no_grad():
            feats = self.encoder(x)
        low_res_feats = feats[self.layer_nums[-1]]

        all_clusters = []

        # all_clusters.append(self.cluster1(low_res_feats) + self.cluster1_nl(low_res_feats))
        all_clusters.append(self.cluster1(low_res_feats))

        if self.granularity >= 2:
            # f1 = self.conv1(low_res_feats)
            # f1_up = self.up(f1)
            f1_up = self.up(low_res_feats)
            f2 = self.conv2(self.c(f1_up, feats[self.layer_nums[-2]]))
            all_clusters.append(self.cluster2(f2))
        if self.granularity >= 3:
            f2_up = self.up(f2)
            f3 = self.conv3(self.c(f2_up, feats[self.layer_nums[-3]]))
            all_clusters.append(self.cluster3(f3))
        if self.granularity >= 4:
            f3_up = self.up(f3)
            final_size = self.spatial_resolutions[-1]
            f4 = self.conv4(self.c(f3_up, F.interpolate(
                x, (final_size, final_size), mode="bilinear", align_corners=False)))
            all_clusters.append(self.cluster4(f4))

        avg_code = torch.cat(all_clusters, 4).mean(4)

        if self.continuous:
            clusters = avg_code
        else:
            clusters = torch.log_softmax(avg_code, 1)

        return low_res_feats, clusters


class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, x):
        return self.double_conv(x)


def norm(t):
    return F.normalize(t, dim=1, eps=1e-10)


def average_norm(t):
    return t / t.square().sum(1, keepdim=True).sqrt().mean()


def tensor_correlation(a, b):
    return torch.einsum("nchw,ncij->nhwij", a, b)


def sample(t: torch.Tensor, coords: torch.Tensor):
    return F.grid_sample(t, coords.permute(0, 2, 1, 3), padding_mode='border', align_corners=True)


@torch.jit.script
def super_perm(size: int, device: torch.device):
    perm = torch.randperm(size, device=device, dtype=torch.long)
    perm[perm == torch.arange(size, device=device)] += 1
    return perm % size


def sample_nonzero_locations(t, target_size):
    nonzeros = torch.nonzero(t)
    coords = torch.zeros(target_size, dtype=nonzeros.dtype, device=nonzeros.device)
    n = target_size[1] * target_size[2]
    for i in range(t.shape[0]):
        selected_nonzeros = nonzeros[nonzeros[:, 0] == i]
        if selected_nonzeros.shape[0] == 0:
            selected_coords = torch.randint(t.shape[1], size=(n, 2), device=nonzeros.device)
        else:
            selected_coords = selected_nonzeros[torch.randint(len(selected_nonzeros), size=(n,)), 1:]
        coords[i, :, :, :] = selected_coords.reshape(target_size[1], target_size[2], 2)
    coords = coords.to(torch.float32) / t.shape[1]
    coords = coords * 2 - 1
    return torch.flip(coords, dims=[-1])


class ContrastiveCorrelationLoss(nn.Module):

    def __init__(self, cfg, ):
        super(ContrastiveCorrelationLoss, self).__init__()
        self.cfg = cfg

    def standard_scale(self, t):
        t1 = t - t.mean()
        t2 = t1 / t1.std()
        return t2

    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))

        if self.cfg.zero_clamp:
            min_val = 0.0
        else:
            min_val = -9999.0

        if self.cfg.stabalize:
            loss = - cd.clamp(min_val, .8) * (fd - shift)
        else:
            loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def forward(self,
                orig_feats: torch.Tensor, orig_feats_pos: torch.Tensor,
                orig_salience: torch.Tensor, orig_salience_pos: torch.Tensor,
                orig_code: torch.Tensor, orig_code_pos: torch.Tensor,
                ):

        coord_shape = [orig_feats.shape[0], self.cfg.feature_samples, self.cfg.feature_samples, 2]

        if self.cfg.use_salience:
            coords1_nonzero = sample_nonzero_locations(orig_salience, coord_shape)
            coords2_nonzero = sample_nonzero_locations(orig_salience_pos, coord_shape)
            coords1_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2_reg = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            mask = (torch.rand(coord_shape[:-1], device=orig_feats.device) > .1).unsqueeze(-1).to(torch.float32)
            coords1 = coords1_nonzero * mask + coords1_reg * (1 - mask)
            coords2 = coords2_nonzero * mask + coords2_reg * (1 - mask)
        else:
            coords1 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1
            coords2 = torch.rand(coord_shape, device=orig_feats.device) * 2 - 1

        feats = sample(orig_feats, coords1)
        code = sample(orig_code, coords1)

        feats_pos = sample(orig_feats_pos, coords2)
        code_pos = sample(orig_code_pos, coords2)

        pos_intra_loss, pos_intra_cd = self.helper(
            feats, feats, code, code, self.cfg.pos_intra_shift)
        pos_inter_loss, pos_inter_cd = self.helper(
            feats, feats_pos, code, code_pos, self.cfg.pos_inter_shift)

        neg_losses = []
        neg_cds = []
        for i in range(self.cfg.neg_samples):
            perm_neg = super_perm(orig_feats.shape[0], orig_feats.device)
            feats_neg = sample(orig_feats[perm_neg], coords2)
            code_neg = sample(orig_code[perm_neg], coords2)
            neg_inter_loss, neg_inter_cd = self.helper(
                feats, feats_neg, code, code_neg, self.cfg.neg_inter_shift)
            neg_losses.append(neg_inter_loss)
            neg_cds.append(neg_inter_cd)
        neg_inter_loss = torch.cat(neg_losses, axis=0)
        neg_inter_cd = torch.cat(neg_cds, axis=0)

        return (pos_intra_loss.mean(),
                pos_intra_cd,
                pos_inter_loss.mean(),
                pos_inter_cd,
                neg_inter_loss,
                neg_inter_cd)


class Decoder(nn.Module):
    def __init__(self, code_channels, feat_channels):
        super().__init__()
        self.linear = torch.nn.Conv2d(code_channels, feat_channels, (1, 1))
        self.nonlinear = torch.nn.Sequential(
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, code_channels, (1, 1)),
            torch.nn.ReLU(),
            torch.nn.Conv2d(code_channels, feat_channels, (1, 1)))

    def forward(self, x):
        return self.linear(x) + self.nonlinear(x)


class NetWithActivations(torch.nn.Module):
    def __init__(self, model, layer_nums):
        super(NetWithActivations, self).__init__()
        self.layers = nn.ModuleList(model.children())
        self.layer_nums = []
        for l in layer_nums:
            if l < 0:
                self.layer_nums.append(len(self.layers) + l)
            else:
                self.layer_nums.append(l)
        self.layer_nums = set(sorted(self.layer_nums))

    def forward(self, x):
        activations = {}
        for ln, l in enumerate(self.layers):
            x = l(x)
            if ln in self.layer_nums:
                activations[ln] = x
        return activations


class ContrastiveCRFLoss(nn.Module):

    def __init__(self, n_samples, alpha, beta, gamma, w1, w2, shift):
        super(ContrastiveCRFLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.w1 = w1
        self.w2 = w2
        self.n_samples = n_samples
        self.shift = shift

    def forward(self, guidance, clusters):
        device = clusters.device
        assert (guidance.shape[0] == clusters.shape[0])
        assert (guidance.shape[2:] == clusters.shape[2:])
        h = guidance.shape[2]
        w = guidance.shape[3]

        coords = torch.cat([
            torch.randint(0, h, size=[1, self.n_samples], device=device),
            torch.randint(0, w, size=[1, self.n_samples], device=device)], 0)

        selected_guidance = guidance[:, :, coords[0, :], coords[1, :]]
        coord_diff = (coords.unsqueeze(-1) - coords.unsqueeze(1)).square().sum(0).unsqueeze(0)
        guidance_diff = (selected_guidance.unsqueeze(-1) - selected_guidance.unsqueeze(2)).square().sum(1)

        sim_kernel = self.w1 * torch.exp(- coord_diff / (2 * self.alpha) - guidance_diff / (2 * self.beta)) + \
                     self.w2 * torch.exp(- coord_diff / (2 * self.gamma)) - self.shift

        selected_clusters = clusters[:, :, coords[0, :], coords[1, :]]
        cluster_sims = torch.einsum("nka,nkb->nab", selected_clusters, selected_clusters)
        return -(cluster_sims * sim_kernel)
