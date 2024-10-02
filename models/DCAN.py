import torch
import torch.nn as nn
from scipy import ndimage
import torch.nn.functional as F
import numpy as np
from scipy.ndimage import measurements
from scipy.ndimage.morphology import binary_fill_holes
from skimage.morphology import remove_small_objects

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


class Decoder(nn.Module):
    
    def __init__(self, filters):  
        super(Decoder, self).__init__()
        
        self.upsample_1 = Upsample(filters[3], filters[3], 2, 2)
        self.up_residual_conv1 = DoubleConv(filters[3] + filters[2], filters[2], 1, 1)

        self.upsample_2 = Upsample(filters[2], filters[2], 2, 2)
        self.up_residual_conv2 = DoubleConv(filters[2] + filters[1], filters[1], 1, 1)

        self.upsample_3 = Upsample(filters[1], filters[1], 2, 2)
        self.up_residual_conv3 = DoubleConv(filters[1] + filters[0], filters[0], 1, 1)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x4 = self.upsample_1(x4)                            # 512, 64, 64
        x5 = torch.cat([x4, x3], dim=1)                     # 768, 64, 64
        x6 = self.up_residual_conv1(x5)                            # 256, 64, 64
        x6 = self.upsample_2(x6)                            # 256, 128, 128
        x7 = torch.cat([x6, x2], dim=1)                     # 384, 128, 128        
        x8 = self.up_residual_conv2(x7)                             # 128, 128, 128
        x8 = self.upsample_3(x8)                            # 128, 256, 256
        x9 = torch.cat([x8, x1], dim=1)                     # 192, 256, 256
        x10 = self.up_residual_conv3(x9)                    # 64, 256, 256
        
        return x10


class DCAN(nn.Module):
    def __init__(self, channel, num_classes, filters=[128, 256, 512, 1024]):  
        super(DCAN, self).__init__()

        self.input_layer = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1),
            nn.BatchNorm2d(filters[0]),
            nn.ReLU(),
            nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1),
        )
        self.input_skip = nn.Sequential(
            nn.Conv2d(channel, filters[0], kernel_size=3, padding=1)
        )


        self.residual_conv_1 = DoubleConv(filters[0], filters[1], 2, 1)
        self.residual_conv_2 = DoubleConv(filters[1], filters[2], 2, 1)

        self.bridge = DoubleConv(filters[2], filters[3], 2, 1)

        self.decoder1 = Decoder(filters)
        self.decoder2 = Decoder(filters)

        self.type_map = nn.Sequential(
            nn.Conv2d(filters[0], num_classes, 1, stride=1, padding=0, bias=True),
        )

        self.boundary_probability_map = nn.Sequential(
            nn.Conv2d(filters[0], 2, 1, stride=1, padding=0, bias=True),
        )

        self.dilate = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)


    def forward(self, x):
        # Encode
        x1 = self.input_layer(x) + self.input_skip(x)       # 64, 256, 256
        x2 = self.residual_conv_1(x1)                       # 128, 128, 128
        x3 = self.residual_conv_2(x2)                       # 256, 64, 64
        
        # Bridge
        x4 = self.bridge(x3)                                # 512, 32, 32

        # Decode
        d1 = self.decoder1((x1,x2,x3,x4))
        d2 = self.decoder2((x1,x2,x3,x4))

        type_map = self.type_map(d1)
        boundary_probability_map = self.boundary_probability_map(d2)

        out = {
            "type_map": type_map,
            "boundary_probability_map": boundary_probability_map,
        }

        return out


    def get_loss(self, out, feed_dict):
        "计算损失"
        loss_weight={
            "loss_o_ce": 1,
            "loss_o_dice": 1,
            "loss_c_ce": 1,
            "loss_c_dice": 1,
        }

        def dice_loss(pred, true, focus=None):
            """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
            inse = torch.sum(pred * true, (0, 1, 2))
            l = torch.sum(pred, (0, 1, 2))
            r = torch.sum(true, (0, 1, 2))
            loss = 1.0 - (2.0 * inse + 1e-3) / (l + r + 1e-3)
            loss = torch.mean(loss)

            return loss
        ce_loss = nn.CrossEntropyLoss()


        type_pred = out["type_map"]
        B, C, H, W = type_pred.size()

        boundary_pred = out["boundary_probability_map"]
        device = type_pred.device

        type_true = feed_dict["type_map"]
        type_true = type_true.type(torch.long).to(device)

        boundary_true = feed_dict["boundary_map"]
        boundary_true = (boundary_true==1).type(torch.long).to(device)

        loss_dict = {}
        loss_dict["loss_o_ce"] = ce_loss(type_pred, type_true)
        loss_dict["loss_o_dice"] = dice_loss(F.softmax(type_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(type_true, num_classes=type_pred.shape[1]).type(torch.float32))
         
        loss_dict["loss_c_ce"] = ce_loss(boundary_pred, boundary_true)
        loss_dict["loss_c_dice"] = dice_loss(F.softmax(boundary_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(boundary_true, num_classes=boundary_pred.shape[1]).type(torch.float32))
        


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

        boundary_map = out["boundary_probability_map"].to("cpu").data.numpy()
        boundary_map = np.argmax(boundary_map, axis=1)
        
        # 类别减去边界
        type_map[boundary_map>0] = 0

        B, H, W = type_map.shape

        ann_list = []
        for b in range(B):

            ann_map = np.zeros((H, W, 2))
            # 连通域分割的作为Unet的分割结果
            instance_map = measurements.label(type_map[b]>0)[0]
            instance_map = remove_small_objects(instance_map, min_size=instance_threshold[0])

            count = 1
            for i in np.unique(instance_map)[1:]:
                inst_mask = instance_map == i
                # 填充中间缺失的洞
                # inst_mask = binary_fill_holes(inst_mask)
                
                type_list = type_map[b, inst_mask]
                # 去除背景0
                type_list = type_list[type_list > 0]
                if type_list.shape[0]>0:
                    # 以概率的形式选择细胞类型
                    # ann_map[inst_mask, 1] = np.random.choice(type_list)
                    # 直接取众数
                    ann_map[inst_mask, 1] = np.argmax(np.bincount(type_list)) 
                    ann_map[inst_mask, 0] = count
                    count += 1

            ann_list.append(ann_map)
        

        ann_list = np.array(ann_list)

        return ann_list.astype("int32")
       
        