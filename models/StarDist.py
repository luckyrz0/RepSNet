import torch
import torch.nn as nn
import torch.nn.functional as F
from stardist import dist_to_coord, non_maximum_suppression, polygons_to_label
import numpy as np

class outconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(outconv, self).__init__()
        self.conv = nn.Conv2d(in_ch, out_ch, 1)

    def forward(self, x):
        x = self.conv(x)
        return x
    
class down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(down, self).__init__()
        self.mpconv = nn.Sequential(
            nn.MaxPool2d(2),
            double_conv(in_ch, out_ch)
        )

    def forward(self, x):
        x = self.mpconv(x)
        return x


class up(nn.Module):
    def __init__(self, in_ch, out_ch, bilinear=True):
        super(up, self).__init__()

    
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        else:
            self.up = nn.ConvTranspose2d(in_ch//2, in_ch//2, 2, stride=2)

        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # input is CHW
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]

        x1 = F.pad(x1, (diffX // 2, diffX - diffX//2,
                        diffY // 2, diffY - diffY//2))

        x = torch.cat([x2, x1], dim=1)
        x = self.conv(x)
        return x

class double_conv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_ch, out_ch):
        super(double_conv, self).__init__()
        num_groups = out_ch // 8
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch,num_groups=num_groups),
            nn.ELU(inplace=True),
            #nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.GroupNorm(num_channels=out_ch,num_groups=num_groups),
            nn.ELU(inplace=True)
            #nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x

class inconv(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(inconv, self).__init__()
        self.conv = double_conv(in_ch, out_ch)

    def forward(self, x):
        x = self.conv(x)
        return x

class StarDist(nn.Module):

    def __init__(self, n_channels, n_rays=32, erosion_factor_list=[0.2, 0.4, 0.6, 0.8, 1.0], return_conf=False, with_seg=False, n_seg_cls=1):
        super(StarDist, self).__init__()
        self.inc = inconv(n_channels, 128)
        self.down1 = down(128, 256)
        self.down2 = down(256, 512)
        self.down3 = down(512, 512)
        self.up1 = up(1024, 256, bilinear=True)
        self.up2 = up(512, 128, bilinear=True)
        self.up3 = up(256, 128, bilinear=True)
        self.features = nn.Conv2d(128, 128, 3, padding=1)
        self.out_prob = outconv(128, 1)
        self.out_ray = outconv(128, n_rays)


    def forward(self, img, gt_dist=None):
        x1 = self.inc(img)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)

        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        x = self.features(x)

        out_ray = self.out_ray(x)
        out_prob = self.out_prob(x)

        return [out_ray, out_prob]
    
    def init_weight(self):
        for m in self.modules():        
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if hasattr(m, 'bias'):
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
        nn.init.constant_(self.conv_1_confidence.conv.bias, 1.0)


def get_ann(preds,center_prob_thres=0.4):
    h, w, _ = preds[1].shape
    dist_cuda = preds[0][-1][:, :, :h, :w]
    dist = dist_cuda.data.cpu()
    prob = preds[1][-1].data.cpu()[:, :, :h, :w]
    dist_numpy = dist.numpy().squeeze()
    prob_numpy = prob.numpy().squeeze()
    prob_numpy = prob_numpy
    
    dist_numpy = np.transpose(dist_numpy,(1,2,0))
    coord = dist_to_coord(dist_numpy)
    points = non_maximum_suppression(coord, prob_numpy, prob_thresh=center_prob_thres)
    star_label = polygons_to_label(coord, prob_numpy, points)
    return star_label