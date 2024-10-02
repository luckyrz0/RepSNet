import numpy as np
import torch
import torch.nn as nn
from torchvision.models.resnet import Bottleneck as ResNetBottleneck
from torchvision.models.resnet import ResNet

from collections import OrderedDict
import torch.nn.functional as F

import cv2
import math
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects
from skimage.segmentation import watershed
from utils.utils import get_bounding_box

class UpSample2x(nn.Module):
    """A layer to scale input by a factor of 2.
    This layer uses Kronecker product underneath rather than the default
    pytorch interpolation.
    """

    def __init__(self):
        super().__init__()
        # correct way to create constant within module
        self.register_buffer(
            "unpool_mat", torch.from_numpy(np.ones((2, 2), dtype="float32"))
        )
        self.unpool_mat.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        """Logic for using layers defined in init.
        Args:
            x (torch.Tensor):
                Input images, the tensor is in the shape of NCHW.
        Returns:
            torch.Tensor:
                Input images upsampled by a factor of 2 via nearest
                neighbour interpolation. The tensor is the shape as
                NCHW.
        """
        input_shape = list(x.shape)
        # un-squeeze is the same as expand_dims
        # permute is the same as transpose
        # view is the same as reshape
        x = x.unsqueeze(-1)  # bchwx1
        mat = self.unpool_mat.unsqueeze(0)  # 1xshxsw
        ret = torch.tensordot(x, mat, dims=1)  # bxcxhxwxshxsw
        ret = ret.permute(0, 1, 2, 4, 3, 5)
        return ret.reshape((-1, input_shape[1], input_shape[2] * 2, input_shape[3] * 2))

class ResNetExt(ResNet):
    def _forward_impl(self, x, freeze):
        # See note [TorchScript super()]
        if self.training:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            with torch.set_grad_enabled(not freeze):
                x1 = x = self.layer1(x)
                x2 = x = self.layer2(x)
                x3 = x = self.layer3(x)
                x4 = x = self.layer4(x)
        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)
            x1 = x = self.layer1(x)
            x2 = x = self.layer2(x)
            x3 = x = self.layer3(x)
            x4 = x = self.layer4(x)
        return x1, x2, x3, x4

    def forward(self, x: torch.Tensor, freeze: bool = False) -> torch.Tensor:
        return self._forward_impl(x, freeze)

    @staticmethod
    def resnet50(num_input_channels, pretrained=None):
        model = ResNetExt(ResNetBottleneck, [3, 4, 6, 3])
        model.conv1 = nn.Conv2d(
            num_input_channels, 64, 7, stride=1, padding=3)
        if pretrained is not None:
            pretrained = torch.load(pretrained)
            (
                missing_keys, unexpected_keys
            ) = model.load_state_dict(pretrained, strict=False)
        return model


class DenseBlock(nn.Module):
    """Dense Block as defined in:
    Huang, Gao, Zhuang Liu, Laurens Van Der Maaten, and Kilian Q. Weinberger. 
    "Densely connected convolutional networks." In Proceedings of the IEEE conference 
    on computer vision and pattern recognition, pp. 4700-4708. 2017.
    Only performs `valid` convolution.
    """

    def __init__(self, in_ch, unit_ksize, unit_ch, unit_count, split=1):
        super().__init__()
        assert len(unit_ksize) == len(unit_ch), "Unbalance Unit Info"

        self.nr_unit = unit_count
        self.in_ch = in_ch
        self.unit_ch = unit_ch

        # ! For inference only so init values for batchnorm may not match tensorflow
        unit_in_ch = in_ch
        pad_vals = [v // 2 for v in unit_ksize]
        self.units = nn.ModuleList()
        for idx in range(unit_count):
            self.units.append(
                nn.Sequential(
                    nn.BatchNorm2d(unit_in_ch, eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_in_ch, unit_ch[0], unit_ksize[0],
                        stride=1, padding=pad_vals[0], bias=False,
                    ),
                    nn.BatchNorm2d(unit_ch[0], eps=1e-5),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(
                        unit_ch[0], unit_ch[1], unit_ksize[1],
                        stride=1, padding=pad_vals[1], bias=False,
                        groups=split,
                    ),
                )
            )
            unit_in_ch += unit_ch[1]

        self.blk_bna = nn.Sequential(
            nn.BatchNorm2d(unit_in_ch, eps=1e-5),
            nn.ReLU(inplace=True)
        )

    def out_ch(self):
        return self.in_ch + self.nr_unit * self.unit_ch[-1]

    def forward(self, prev_feat):
        for idx in range(self.nr_unit):
            new_feat = self.units[idx](prev_feat)
            prev_feat = torch.cat([prev_feat, new_feat], dim=1)
        prev_feat = self.blk_bna(prev_feat)

        return prev_feat


class HoVerNetConic(nn.Module):
    """Initialise HoVer-Net."""

    def __init__(
            self,
            num_types=None,
            freeze=False,
            pretrained_backbone=None,
            ):
        super().__init__()
        self.freeze = freeze
        self.num_types = num_types
        self.output_ch = 3 if num_types is None else 4

        self.backbone = ResNetExt.resnet50(
            3, pretrained=pretrained_backbone)
        self.conv_bot = nn.Conv2d(
            2048, 1024, 1, stride=1, padding=0, bias=False)

        def create_decoder_branch(out_ch=2, ksize=5):
            pad = ksize // 2
            module_list = [
                nn.Conv2d(1024, 256, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(256, [1, ksize], [128, 32], 8, split=4),
                nn.Conv2d(512, 512, 1, stride=1, padding=0, bias=False),
            ]
            u3 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(512, 128, ksize, stride=1, padding=pad, bias=False),
                DenseBlock(128, [1, ksize], [128, 32], 4, split=4),
                nn.Conv2d(256, 256, 1, stride=1, padding=0, bias=False),
            ]
            u2 = nn.Sequential(*module_list)

            module_list = [
                nn.Conv2d(256, 64, ksize, stride=1, padding=pad, bias=False),
            ]
            u1 = nn.Sequential(*module_list)

            module_list = [
                nn.BatchNorm2d(64, eps=1e-5),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1, stride=1, padding=0, bias=True),
            ]
            u0 = nn.Sequential(*module_list)

            decoder = nn.Sequential(
                OrderedDict([("u3", u3), ("u2", u2), ("u1", u1), ("u0", u0)])
            )
            return decoder

        ksize = 3
        if num_types is None:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("nucleus_map", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv_map", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )
        else:
            self.decoder = nn.ModuleDict(
                OrderedDict(
                    [
                        ("type_map", create_decoder_branch(ksize=ksize, out_ch=num_types)),
                        ("nucleus_map", create_decoder_branch(ksize=ksize, out_ch=2)),
                        ("hv_map", create_decoder_branch(ksize=ksize, out_ch=2)),
                    ]
                )
            )

        self.upsample2x = UpSample2x()

    def forward(self, imgs):
        imgs = imgs / 255.0  # to 0-1 range to match XY

        d0, d1, d2, d3 = self.backbone(imgs, self.freeze)
        d3 = self.conv_bot(d3)
        d = [d0, d1, d2, d3]

        out_dict = OrderedDict()
        for branch_name, branch_desc in self.decoder.items():
            u3 = self.upsample2x(d[-1]) + d[-2]
            u3 = branch_desc[0](u3)

            u2 = self.upsample2x(u3) + d[-3]
            u2 = branch_desc[1](u2)

            u1 = self.upsample2x(u2) + d[-4]
            u1 = branch_desc[2](u1)

            u0 = branch_desc[3](u1)
            out_dict[branch_name] = u0

        return out_dict


    def get_loss(self, out, feed_dict):
        "计算损失"
        device = out["type_map"].device


        loss_weight={
            "loss_nucleus_ce": 1,
            "loss_nucleus_dice": 1,
            "loss_type_ce": 1,
            "loss_type_dice": 1,
            "loss_hv_mse": 1,
            "loss_hv_msge": 1,
        }

        def dice_loss(pred, true, focus=None):
            """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
            inse = torch.sum(pred * true, (0, 1, 2))
            l = torch.sum(pred, (0, 1, 2))
            r = torch.sum(true, (0, 1, 2))
            loss = 1.0 - (2.0 * inse + 1e-3) / (l + r + 1e-3)
            loss = torch.mean(loss)

            return loss

        def msge_loss(pred, true, focus):
            """Calculate the mean squared error of the gradients of 
            horizontal and vertical map predictions. Assumes 
            channel 0 is Vertical and channel 1 is Horizontal.
            Args:
                true:  ground truth of combined horizontal
                    and vertical maps
                pred:  prediction of combined horizontal
                    and vertical maps 
                focus: area where to apply loss (we only calculate
                        the loss within the nuclei)
            
            Returns:
                loss:  mean squared error of gradients
            """


            def get_sobel_kernel(size):
                """Get sobel kernel with a given size."""
                assert size % 2 == 1, "Must be odd, get size=%d" % size

                h_range = torch.arange(
                    -size // 2 + 1,
                    size // 2 + 1,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=False,
                )
                v_range = torch.arange(
                    -size // 2 + 1,
                    size // 2 + 1,
                    dtype=torch.float32,
                    device=device,
                    requires_grad=False,
                )
                h, v = torch.meshgrid(h_range, v_range)
                kernel_h = h / (h * h + v * v + 1.0e-15)
                kernel_v = v / (h * h + v * v + 1.0e-15)
                return kernel_h, kernel_v

            ####
            def get_gradient_hv(hv):
                """For calculating gradient."""
                kernel_h, kernel_v = get_sobel_kernel(5)
                kernel_h = kernel_h.view(1, 1, 5, 5)  # constant
                kernel_v = kernel_v.view(1, 1, 5, 5)  # constant

                h_ch = hv[..., 0].unsqueeze(1)  # Nx1xHxW
                v_ch = hv[..., 1].unsqueeze(1)  # Nx1xHxW

                # can only apply in NCHW mode
                h_dh_ch = F.conv2d(h_ch, kernel_h, padding=2)
                v_dv_ch = F.conv2d(v_ch, kernel_v, padding=2)
                dhv = torch.cat([h_dh_ch, v_dv_ch], dim=1)
                dhv = dhv.permute(0, 2, 3, 1).contiguous()  # to NHWC

                return dhv

            focus = (focus[..., None]).float()  # assume input NHW
            focus = torch.cat([focus, focus], axis=-1)
            true_grad = get_gradient_hv(true)
            pred_grad = get_gradient_hv(pred)
            loss = pred_grad - true_grad
            loss = focus * (loss * loss)
            # artificial reduce_mean with focused region
            loss = loss.sum() / (focus.sum() + 1.0e-8)
            return loss

        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()

        

        nucleus_pred, type_pred, hv_pred  = out["nucleus_map"], out["type_map"], out["hv_map"],
        type_true, hv_true = feed_dict["type_map"], feed_dict["hv_map"]

        type_true = type_true.type(torch.long).to(device)
        nucleus_true = (type_true > 0).type(torch.long)
        hv_true = hv_true.type(torch.float32).to(device)

        loss_dict = {}

        # nucleus_loss
        loss_dict["loss_nucleus_ce"] = ce_loss(nucleus_pred, nucleus_true)
        loss_dict["loss_nucleus_dice"] = dice_loss(F.softmax(nucleus_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(nucleus_true, num_classes=nucleus_pred.shape[1]).type(torch.float32))

        # type_loss
        loss_dict["loss_type_ce"] = ce_loss(type_pred, type_true)
        loss_dict["loss_type_dice"] = dice_loss(F.softmax(type_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(type_true, num_classes=type_pred.shape[1]).type(torch.float32))
        

        loss_dict["loss_hv_mse"] = mse_loss(hv_pred.permute(0, 2, 3, 1), hv_true)
        loss_dict["loss_hv_msge"] = msge_loss(hv_pred.permute(0, 2, 3, 1), hv_true, nucleus_true)


        overall_loss = 0
        for key in loss_dict:
            loss_dict[key] *= loss_weight[key]
            overall_loss += loss_dict[key]

        return overall_loss, loss_dict


    @staticmethod
    def _proc_np_hv(np_map: np.ndarray, hv_map: np.ndarray, scale_factor: float = 1):
        """Extract Nuclei Instance with NP and HV Map.
        Sobel will be applied on horizontal and vertical channel in
        `hv_map` to derive an energy landscape which highlight possible
        nuclei instance boundaries. Afterward, watershed with markers is
        applied on the above energy map using the `np_map` as filter to
        remove background regions.
        Args:
            np_map (np.ndarray):
                An image of shape (height, width, 1) which contains the
                probabilities of a pixel being a nucleus.
            hv_map (np.ndarray):
                An array of shape (height, width, 2) which contains the
                horizontal (channel 0) and vertical (channel 1) maps of
                possible instances within the image.
            scale_factor (float):
                The scale factor for processing nuclei. The scale
                assumes an image of resolution 0.25 microns per pixel.
                Default is therefore 1 for HoVer-Net.
        Returns:
            :class:`numpy.ndarray`:
                An np.ndarray of shape (height, width) where each
                non-zero values within the array correspond to one
                detected nuclei instances.
        """
        blb_raw = np_map[..., 0] #是否为细胞的分割概率图像
        h_dir_raw = hv_map[..., 0] #水平距离图
        v_dir_raw = hv_map[..., 1] #垂直距离图

        # processing
        blb = np.array(blb_raw >= 0.5, dtype=np.int32) #大于0.5认为是细胞像素

        blb = measurements.label(blb)[0]  #将不同的区域标上不一样的标签（粗略的实例分割） 返回 粗略的实例分割图 和 多少个实例 
        blb = remove_small_objects(blb, min_size=10)  #将粗略的实例分割图中像素少于10的实例去除 
        blb[blb > 0] = 1  # background is 0 already  #将其再转为 二值图
        
        #归一化 到（0,1）
        h_dir = cv2.normalize(  
            h_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        v_dir = cv2.normalize(
            v_dir_raw,
            None,
            alpha=0,
            beta=1,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        ksize = int((20 * scale_factor) + 1)
        # Get resolution specific filters etc.

        sobelh = cv2.Sobel(h_dir, cv2.CV_64F, 1, 0, ksize=ksize)
        sobelv = cv2.Sobel(v_dir, cv2.CV_64F, 0, 1, ksize=ksize)

        sobelh = 1 - (
            cv2.normalize(
                sobelh,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )
        sobelv = 1 - (
            cv2.normalize(
                sobelv,
                None,
                alpha=0,
                beta=1,
                norm_type=cv2.NORM_MINMAX,
                dtype=cv2.CV_32F,
            )
        )

        overall = np.maximum(sobelh, sobelv) #返回 水平和垂直梯度的最大值
        overall = overall - (1 - blb)
        overall[overall < 0] = 0  #只保留细胞部分的梯度

        dist = (1.0 - overall) * blb 
        # * nuclei values form mountains so inverse to get basins
        dist = -cv2.GaussianBlur(dist, (3, 3), 0)

        overall = np.array(overall >= 0.4, dtype=np.int32)

        
        marker = blb - overall
        marker[marker < 0] = 0
        marker = binary_fill_holes(marker).astype("uint8")

        # 开运算和剔除小细胞会影响精度
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        # marker = cv2.morphologyEx(marker, cv2.MORPH_OPEN, kernel)
        marker = measurements.label(marker)[0]
        obj_size = math.ceil(16 * (scale_factor**2))
        marker = remove_small_objects(marker, min_size=obj_size)

        return watershed(dist, markers=marker, mask=blb), marker  

    @staticmethod
    def get_markers(pred_dict, net_args={}):
        with torch.no_grad():
            pred_dict = OrderedDict(
                    [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
                )
            np_map = F.softmax(pred_dict["nucleus_map"], dim=-1)[..., 1:]
            np_map = np_map.cpu().numpy()
            hv_map = pred_dict["hv_map"].cpu().numpy()
            tp_map = None
            if "type_map" in pred_dict:
                tp_map = F.softmax(pred_dict["type_map"], dim=-1)
                tp_map = torch.argmax(tp_map, dim=-1, keepdim=True)
                tp_map = tp_map.type(torch.float32).cpu().numpy()
            
            B, H, W = np_map.shape[:3]
            ann_list = []
            marker_map = np.zeros((B, H, W))
            for b in range(B):
                inst_map, marker = HoVerNetConic._proc_np_hv(np_map[b], hv_map[b])
                marker_map[b] = marker
                
            return marker_map.astype("int32")


    @staticmethod
    def get_ann(pred_dict, net_args={}):
        with torch.no_grad():
            pred_dict = OrderedDict(
                    [[k, v.permute(0, 2, 3, 1).contiguous()] for k, v in pred_dict.items()]
                )
            np_map = F.softmax(pred_dict["nucleus_map"], dim=-1)[..., 1:]
            np_map = np_map.cpu().numpy()
            hv_map = pred_dict["hv_map"].cpu().numpy()
            tp_map = None
            if "type_map" in pred_dict:
                tp_map = F.softmax(pred_dict["type_map"], dim=-1)
                tp_map = torch.argmax(tp_map, dim=-1, keepdim=True)
                tp_map = tp_map.type(torch.float32).cpu().numpy()
            
            B, H, W = np_map.shape[:3]
            ann_list = []
            for b in range(B):
                ann_map = np.zeros((H, W, 2))
                inst_map, marker = HoVerNetConic._proc_np_hv(np_map[b], hv_map[b])
                ann_map[..., 0] = inst_map

                for i in np.unique(inst_map)[1:]:
                    type_list = tp_map[b, inst_map==i]
                    # 去除背景0
                    type_list = type_list[type_list > 0].astype(np.long)
                    if type_list.shape[0]>0:
                        # 以概率的形式选择细胞类型
                        # ann_map[instance_map==i, 1] = np.random.choice(type_list)
                        # 直接取众数
                        ann_map[inst_map==i, 1] = np.argmax(np.bincount(type_list)) 
                    else:
                        # 如果没有类别，则表示实例识别错误
                        ann_map[inst_map==i, 0] = 0
                
                ann_list.append(ann_map)

            ann_list = np.array(ann_list)

            return ann_list.astype("int32")




if __name__ == "__main__":
    HoVerNetConic()
    pass