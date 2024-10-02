import torch
import torch.nn as nn
from scipy import ndimage
import torch.nn.functional as F
import numpy as np

# this code from: https://github.com/rishikksh20/ResUnet

class ResidualConv(nn.Module):
    def __init__(self, input_dim, output_dim, stride, padding):
        super(ResidualConv, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


class DoubleConv(nn.Module):
    
    def __init__(self, input_dim, output_dim, stride, padding):
        super(DoubleConv, self).__init__()

        assert stride == 2 or stride == 1

        self.max_pool = None
        if stride == 2:
            self.max_pool = nn.MaxPool2d(kernel_size=stride)
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=padding, bias=False),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(inplace=True)
        )
       

    def forward(self, x):
        if self.max_pool is not None:
            x = self.max_pool(x)

        return self.double_conv(x)



class Upsample(nn.Module):
    def __init__(self, input_dim, output_dim, kernel, stride):
        super(Upsample, self).__init__()

        self.upsample = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size=kernel, stride=stride
        )

    def forward(self, x):
        return self.upsample(x)



class UNet(nn.Module):
    def __init__(self, channel, num_classes, filters=[128, 256, 512, 1024], ResUNet=False):  
        super(UNet, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )

        ConvBock = ResidualConv if ResUNet else DoubleConv


        self.residual_conv_1 = ConvBock(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = ConvBock(filters[1], filters[2], 2, 1)

        self.bridge = ConvBock(filters[2], filters[3], 2, 1)

        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = ConvBock(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = ConvBock(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = ConvBock(filters[1] + filters[0], filters[0], 1, 1)

        self.type_map = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1, stride=1, padding=0, bias=True),
        )


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)       # 64, 256, 256
        x2 = self.residual_conv_1(x1)                       # 128, 128, 128
        x3 = self.residual_conv_2(x2)                       # 256, 64, 64
        
        # Bridge
        x4 = self.bridge(x3)                                # 512, 32, 32

        # Decode
        x4 = self.upsample_1(x4)                            # 512, 64, 64

        x5 = torch.cat([x4, x3], dim=1)                     # 768, 64, 64
        x6 = self.up_residual_conv1(x5)                     # 256, 64, 64

        x6 = self.upsample_2(x6)                            # 256, 128, 128
        x7 = torch.cat([x6, x2], dim=1)                     # 384, 128, 128        

        x8 = self.up_residual_conv2(x7)                     # 128, 128, 128

        x8 = self.upsample_3(x8)                            # 128, 256, 256
        x9 = torch.cat([x8, x1], dim=1)                     # 192, 256, 256

        x10 = self.up_residual_conv3(x9)                    # 64, 256, 256

        type_map = self.type_map(x10)
        out = {
            "type_map": type_map,
        }

        return out


    def get_loss(self, out, feed_dict):
        "计算损失"
        loss_weight={
            "loss_type_ce": 1,
            "loss_type_dice": 1,
        }

        def dice_loss(pred, true):
            """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
            inse = torch.sum(pred * true, (0, 1, 2))
            l = torch.sum(pred, (0, 1, 2))
            r = torch.sum(true, (0, 1, 2))
            loss = 1.0 - (2.0 * inse + 1e-3) / (l + r + 1e-3)
            loss = torch.mean(loss)

            return loss


        type_pred = out["type_map"]
        device = type_pred.device

        type_true = feed_dict["type_map"]
        type_true = type_true.type(torch.long).to(device)

        ce_loss = nn.CrossEntropyLoss()

        loss_dict = {}
        loss_dict["loss_type_ce"] = ce_loss(type_pred, type_true)
        loss_dict["loss_type_dice"] = dice_loss(F.softmax(type_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(type_true, num_classes=type_pred.shape[1]).type(torch.float32))
           
        
        overall_loss = 0
        for key in loss_dict:
            loss_dict[key] *= loss_weight[key]
            overall_loss += loss_dict[key]

        return overall_loss, loss_dict

    def get_ann(self, out, net_args={}):
        
        instance_threshold=net_args["instance_threshold"]
        type_map = out["type_map"]
        type_map = type_map.to("cpu").data.numpy()
        type_map = np.argmax(type_map, axis=1)


        B, H, W = type_map.shape

        ann_list = []
        for b in range(B):

            ann_map = np.zeros((H, W, 2))
            # 连通域分割的作为Unet的分割结果
            instance_map, _ = ndimage.label(type_map[b]>0)

            # 去除连通较小和较大的
            instance_map_unique, instance_map_count, = np.unique(instance_map, return_counts=True)
            for invalid_instance in instance_map_unique[instance_map_count<=instance_threshold[0]]:
                instance_map[instance_map==invalid_instance] = 0
            for invalid_instance in instance_map_unique[instance_map_count>instance_threshold[1]]:
                instance_map[instance_map==invalid_instance] = 0

            ann_map[..., 0] = instance_map
            ann_map[..., 1] = type_map[b]

            ann_list.append(ann_map)
        

        ann_list = np.array(ann_list)

        return ann_list.astype("int32")

        

        
