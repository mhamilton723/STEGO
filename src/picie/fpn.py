import torch 
import torch.nn as nn 
import torch.nn.functional as F 
from . import backbone


class PanopticFPN(nn.Module):
    def __init__(self, args):
        super(PanopticFPN, self).__init__()
        self.backbone = backbone.__dict__[args.arch](pretrained=args.pretrain)
        self.decoder  = FPNDecoder(args)

    def forward(self, x):
        feats = self.backbone(x)
        outs  = self.decoder(feats) 

        return outs 

class FPNDecoder(nn.Module):
    def __init__(self, args):
        super(FPNDecoder, self).__init__()
        if args.arch == 'resnet18':
            mfactor = 1
            out_dim = 128 
        else:
            mfactor = 4
            out_dim = 256

        self.layer4 = nn.Conv2d(512*mfactor//8, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer3 = nn.Conv2d(512*mfactor//4, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer2 = nn.Conv2d(512*mfactor//2, out_dim, kernel_size=1, stride=1, padding=0)
        self.layer1 = nn.Conv2d(512*mfactor, out_dim, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        o1 = self.layer1(x['res5'])
        o2 = self.upsample_add(o1, self.layer2(x['res4']))
        o3 = self.upsample_add(o2, self.layer3(x['res3']))
        o4 = self.upsample_add(o3, self.layer4(x['res2']))

        return o4

    def upsample_add(self, x, y):
        _, _, H, W = y.size()

        return F.interpolate(x, size=(H, W), mode='bilinear', align_corners=False) + y 



