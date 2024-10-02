import torch.nn as nn
import torch
import numpy as np
from scipy import ndimage
from sklearn.neighbors import NearestNeighbors
import torch.nn.functional as F


# https://nenadmarkus.com/p/fusing-batchnorm-and-conv/
def conv_bn(in_channels, out_channels, kernel_size, stride, padding):
    result = nn.Sequential()
    result.add_module(
        'conv', 
        nn.Conv2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding, 
            bias=False)
    )
    result.add_module(
        'bn', 
        nn.BatchNorm2d(num_features=out_channels)
    )
    return result

def dconv_bn(in_channels, out_channels, kernel_size, stride, padding, output_padding=1):
    result = nn.Sequential()
    result.add_module(
        'dconv', 
        nn.ConvTranspose2d(
            in_channels=in_channels, 
            out_channels=out_channels,
            kernel_size=kernel_size, 
            stride=stride, 
            padding=padding,
            bias=False,
            output_padding=output_padding
        )
    )
    result.add_module(
        'bn', 
        nn.BatchNorm2d(num_features=out_channels)
    )
    return result



class RepVggUnit(nn.Module):

    def __init__(self, 
                in_channels, 
                out_channels, 
                stride=1,
                deploy=False):
        super(RepVggUnit, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels


        if deploy:
            self.conv_reparam = nn.Conv2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=stride,
                padding=1,
                bias=True
            )
        else:
            self.conv3_bn = conv_bn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
            self.conv1_bn = conv_bn(in_channels, out_channels, kernel_size=1, stride=stride, padding=0)
            self.bn = nn.BatchNorm2d(num_features=in_channels) if out_channels == in_channels and stride == 1 else None

    def forward(self, x):
        if hasattr(self, 'conv_reparam'):
            return torch.relu((self.conv_reparam(x)))

        return torch.relu(self.conv3_bn(x) + self.conv1_bn(x) + (self.bn(x) if self.bn is not None else 0))


    def switch_to_deploy(self):
        if hasattr(self, 'conv_reparam'):
            return

        k3_b, b3_b = self.__fuse_bn_tensor(self.conv3_bn)
        k1_b, b1_b = self.__fuse_bn_tensor(self.conv1_bn)
        k_b, b_b = self.__fuse_bn_tensor(self.bn)

        kernel = k3_b + nn.functional.pad(k1_b, [1,1,1,1]) + k_b
        bias =  b3_b + b1_b + b_b

        self.conv_reparam = nn.Conv2d(
            in_channels=self.conv3_bn.conv.in_channels, 
            out_channels=self.conv3_bn.conv.out_channels,
            kernel_size=3, 
            stride=self.conv3_bn.conv.stride,
            padding=1,
            bias=True
        )
        self.conv_reparam.weight.data = kernel
        self.conv_reparam.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('conv3_bn')
        self.__delattr__('conv1_bn')
        if hasattr(self, 'bn'):
            self.__delattr__('bn')
        if hasattr(self, 'bn_tensor'):
            self.__delattr__('bn_tensor')
        self.deploy = True


    def __fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
        if isinstance(branch, nn.Sequential):
            kernel = branch.conv.weight
            running_mean = branch.bn.running_mean
            running_var = branch.bn.running_var
            gamma = branch.bn.weight
            beta = branch.bn.bias
            eps = branch.bn.eps
        else:
            assert isinstance(branch, nn.BatchNorm2d)
            if not hasattr(self, 'bn_tensor'):
                kernel_value = np.zeros((self.in_channels, self.in_channels, 3, 3), dtype=np.float32)
                for i in range(self.in_channels):
                    kernel_value[i, i % self.in_channels, 1, 1] = 1
                self.bn_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
            kernel = self.bn_tensor
            running_mean = branch.running_mean
            running_var = branch.running_var
            gamma = branch.weight
            beta = branch.bias
            eps = branch.eps
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(-1, 1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std

        
class RepUpsample(nn.Module):

    def __init__(self, 
                in_channels, 
                out_channels, 
                deploy=False):
        super(RepUpsample, self).__init__()
        self.deploy = deploy
        self.in_channels = in_channels
        self.bilinearSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)


        if deploy:
            self.dconv_reparam = nn.ConvTranspose2d(
                in_channels=in_channels, 
                out_channels=out_channels, 
                kernel_size=3, 
                stride=2,
                padding=1,
                bias=True,
                output_padding=1,
            )
        else:
            self.dconv3_bn = dconv_bn(in_channels, out_channels, kernel_size=3, stride=2, padding=1)
            self.dconv1_bn = dconv_bn(in_channels, out_channels, kernel_size=1, stride=2, padding=0)


    def forward(self, x):
        if hasattr(self, 'dconv_reparam'):
            return torch.relu((self.dconv_reparam(x) + self.bilinearSample(x)))

        return torch.relu(self.dconv3_bn(x) + self.dconv1_bn(x) + self.bilinearSample(x))


    def switch_to_deploy(self):
        if hasattr(self, 'dconv_reparam'):
            return

        k3_b, b3_b = self.__fuse_bn_tensor(self.dconv3_bn)
        k1_b, b1_b = self.__fuse_bn_tensor(self.dconv1_bn)

        kernel = k3_b + nn.functional.pad(k1_b, [1,1,1,1])
        bias =  b3_b + b1_b

        self.dconv_reparam = nn.ConvTranspose2d(
            in_channels=self.dconv3_bn.dconv.in_channels, 
            out_channels=self.dconv3_bn.dconv.out_channels,
            kernel_size=3, 
            stride=2,
            padding=1,
            bias=True,
            output_padding=1,
        )

        self.dconv_reparam.weight.data = kernel
        self.dconv_reparam.bias.data = bias

        for para in self.parameters():
            para.detach_()
        self.__delattr__('dconv3_bn')
        self.__delattr__('dconv1_bn')
        self.deploy = True


    def __fuse_bn_tensor(self, branch):
        if branch is None:
            return 0, 0
       
        kernel = branch.dconv.weight
        running_mean = branch.bn.running_mean
        running_var = branch.bn.running_var
        gamma = branch.bn.weight
        beta = branch.bn.bias
        eps = branch.bn.eps
        
        std = (running_var + eps).sqrt()
        t = (gamma / std).reshape(1, -1, 1, 1)
        return kernel * t, beta - running_mean * gamma / std



class Encoder(nn.Module):

    def __init__(self, input_channel, filters, deploy=False):
        super(Encoder, self).__init__()

        self.downsample_1 = nn.Sequential(
            RepVggUnit(input_channel, filters[0], stride=1, deploy=deploy),
            RepVggUnit(filters[0], filters[0], stride=1, deploy=deploy)
        )
        self.downsample_2 = nn.Sequential(
            RepVggUnit(filters[0], filters[1], stride=1, deploy=deploy),
            RepVggUnit(filters[1], filters[1], stride=2, deploy=deploy),
            RepVggUnit(filters[1], filters[1], stride=1, deploy=deploy)
        )
        self.downsample_3 = nn.Sequential(
            RepVggUnit(filters[1], filters[2], stride=1, deploy=deploy),
            RepVggUnit(filters[2], filters[2], stride=2, deploy=deploy),
            RepVggUnit(filters[2], filters[2], stride=1, deploy=deploy)
        )
        self.downsample_4 = nn.Sequential(
            RepVggUnit(filters[2], filters[3], stride=1, deploy=deploy),
            RepVggUnit(filters[3], filters[3], stride=2, deploy=deploy),
            RepVggUnit(filters[3], filters[3], stride=1, deploy=deploy)

        )

    def forward(self, x):
        x1 = self.downsample_1(x)                       # 64, 256, 256
        x2 = self.downsample_2(x1)                       # 128, 128, 128
        x3 = self.downsample_3(x2)                       # 256, 64, 64
        x4 = self.downsample_4(x3)                       # 512, 32, 32

        return (x1, x2, x3, x4)


class Decoder(nn.Module):
    
    def __init__(self, filters , deploy=False):  
        super(Decoder, self).__init__()
        
        self.upsample_1 = RepUpsample(filters[3], filters[3], deploy=deploy)
        self.up_repvgg_1 = RepVggUnit(filters[3] + filters[2], filters[2], stride=1, deploy=deploy)

        self.upsample_2 = RepUpsample(filters[2], filters[2], deploy=deploy)
        self.up_repvgg_2 = RepVggUnit(filters[2] + filters[1], filters[1], stride=1, deploy=deploy)

        self.upsample_3 = RepUpsample(filters[1], filters[1], deploy=deploy)
        self.up_repvgg_3 = RepVggUnit(filters[1] + filters[0], filters[0], stride=1, deploy=deploy)

    def forward(self, x):
        x1, x2, x3, x4 = x
        x4 = self.upsample_1(x4)                            # 512, 64, 64
        x5 = torch.cat([x4, x3], dim=1)                     # 768, 64, 64
        x6 = self.up_repvgg_1(x5)                            # 256, 64, 64
        x6 = self.upsample_2(x6)                            # 256, 128, 128
        x7 = torch.cat([x6, x2], dim=1)                     # 384, 128, 128        
        x8 = self.up_repvgg_2(x7)                             # 128, 128, 128
        x8 = self.upsample_3(x8)                            # 128, 256, 256
        x9 = torch.cat([x8, x1], dim=1)                     # 192, 256, 256
        x10 = self.up_repvgg_3(x9)                           # 64, 256, 256
        
        return x10


class RepSNet(nn.Module):

    def __init__(
            self, 
            img_channel=3,
            filters=[64, 128, 256, 512],
            num_classes=7,
            deploy=False,
            pretrained=False,
        ):

        super().__init__()

        self.num_classes = num_classes
        self.pretrained = pretrained
        self.dilate = nn.MaxPool2d(kernel_size=(3,3), stride=1, padding=1)

        self.encoder = Encoder(input_channel=img_channel, filters=filters, deploy=deploy)
        if not self.pretrained:
            self.decoder1 = Decoder(filters=filters, deploy=deploy)
        self.decoder2 = Decoder(filters=filters, deploy=deploy)
       
        if not self.pretrained:
            self.nucleus_map = nn.Sequential(
                nn.Conv2d(filters[0], 2, 1, stride=1, padding=0, bias=True),
            )
            self.type_map = nn.Sequential(
                nn.Conv2d(filters[0], self.num_classes, 1, stride=1, padding=0, bias=True),
            )
        self.boundary_distance_map = nn.Sequential(
            nn.Conv2d(filters[0], 4, 1, stride=1, padding=0, bias=True),
            nn.ReLU()
        )
        
    def forward(self, x):
        x = x / 255.0

        x = self.encoder(x)
        if self.pretrained:
            x2 = self.decoder2(x)
            boundary_distance_map = self.boundary_distance_map(x2) 
            out = {
                "boundary_distance_map": boundary_distance_map,
            }
        else:
            x1 = self.decoder1(x)
            x2 = self.decoder2(x)
            
            nucleus_map = self.nucleus_map(x1)
            type_map = self.type_map(x1) 
            boundary_distance_map = self.boundary_distance_map(x2) 
            
            # 屏蔽非细胞核的距离回归，效果一般，但是能够提高收敛速度
            temp_map = torch.argmax(nucleus_map, dim=1).type(torch.long).detach()
            boundary_distance_map = boundary_distance_map * temp_map.unsqueeze(1)  


            boundary_map = self.__get_boundary_map(nucleus_map, boundary_distance_map)

            out = {
                "nucleus_map": nucleus_map,
                "type_map": type_map,
                "boundary_distance_map": boundary_distance_map,
                "boundary_map": boundary_map,
            }

        return out


    def __get_boundary_map(self, nucleus_map, boundary_distance_map):
        "获取核边界"
        B, N, H, W = boundary_distance_map.size()
        device = boundary_distance_map.device

        # 将细胞核的边界加入了边界特征图中，并且进行训练
        nucleus_map = torch.argmax(nucleus_map, dim=1)
        nucleus_map_erode = -self.dilate(-nucleus_map.type(torch.float).unsqueeze(1)).reshape(B, H, W)
        boundary_map = (nucleus_map - nucleus_map_erode)

        # 核像素索引
        nucleus_mask_index = torch.where(nucleus_map > 0)

        # 开根号
        boundary_distance_mask_value = boundary_distance_map[nucleus_mask_index[0], :, nucleus_mask_index[1], nucleus_mask_index[2]].T ** 2
        # 四舍五入
        boundary_distance_mask_value = (boundary_distance_mask_value+0.5).type(torch.long)

        # 边界的绝对位置
        nucleus_mask_index_l = nucleus_mask_index[2] - boundary_distance_mask_value[0] + 1
        nucleus_mask_index_r = nucleus_mask_index[2] + boundary_distance_mask_value[1] - 1
        nucleus_mask_index_u = nucleus_mask_index[1] - boundary_distance_mask_value[2] + 1
        nucleus_mask_index_d = nucleus_mask_index[1] + boundary_distance_mask_value[3] - 1

        # 限制范围
        nucleus_mask_index_l = torch.clamp(nucleus_mask_index_l, min=0, max=W-1).type(torch.long)
        nucleus_mask_index_r = torch.clamp(nucleus_mask_index_r, min=0, max=W-1).type(torch.long)
        nucleus_mask_index_u = torch.clamp(nucleus_mask_index_u, min=0, max=H-1).type(torch.long)
        nucleus_mask_index_d = torch.clamp(nucleus_mask_index_d, min=0, max=H-1).type(torch.long)

        for b in range(B):
            # 坐标拼接
            uc = torch.cat([nucleus_mask_index_u[nucleus_mask_index[0]==b].reshape(-1,1), 
                         nucleus_mask_index[2][nucleus_mask_index[0]==b].reshape(-1,1)], dim=1)
            dc = torch.cat([nucleus_mask_index_d[nucleus_mask_index[0]==b].reshape(-1,1), 
                                nucleus_mask_index[2][nucleus_mask_index[0]==b].reshape(-1,1)], dim=1)
            cl = torch.cat([nucleus_mask_index[1][nucleus_mask_index[0]==b].reshape(-1,1), 
                                nucleus_mask_index_l[nucleus_mask_index[0]==b].reshape(-1,1)], dim=1)
            cr = torch.cat([nucleus_mask_index[1][nucleus_mask_index[0]==b].reshape(-1,1), 
                                nucleus_mask_index_r[nucleus_mask_index[0]==b].reshape(-1,1)], dim=1)
            
            # 计数，避免重复计算
            uc_unique = torch.unique(uc, dim=0, return_counts=True)
            dc_unique = torch.unique(dc, dim=0, return_counts=True)
            cl_unique = torch.unique(cl, dim=0, return_counts=True)
            cr_unique = torch.unique(cr, dim=0, return_counts=True)

            # 统计边界的次数
            if uc_unique[1].shape[0] > 0:
                boundary_map[b, uc_unique[0][:,0].type(torch.long), uc_unique[0][:,1].type(torch.long)] += uc_unique[1]
            if dc_unique[1].shape[0] > 0:
                boundary_map[b, dc_unique[0][:,0].type(torch.long), dc_unique[0][:,1].type(torch.long)] += dc_unique[1]
            if cl_unique[1].shape[0] > 0:
                boundary_map[b, cl_unique[0][:,0].type(torch.long), cl_unique[0][:,1].type(torch.long)] += cl_unique[1]
            if cr_unique[1].shape[0] > 0:
                boundary_map[b, cr_unique[0][:,0].type(torch.long), cr_unique[0][:,1].type(torch.long)] += cr_unique[1]



        # 边界非极大抑制
        maxpool1d = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        m_boundary_map = maxpool1d(boundary_map.reshape(-1, H*W)).reshape(-1,H, W)
        m_boundary_map[m_boundary_map==0] = -1;
        mT_boundary_map = maxpool1d(boundary_map.permute(0,2,1).reshape(-1, H*W)).reshape(-1, W, H).permute(0, 2, 1)
        mT_boundary_map[mT_boundary_map==0] = -1;
        m_boundary_map = maxpool1d == m_boundary_map
        mT_boundary_map = maxpool1d == mT_boundary_map
        nms_mask = m_boundary_map + mT_boundary_map
        boundary_map[~nms_mask] = 0

        return boundary_map



    def get_loss(self, out, feed_dict):
        "计算损失"
        loss_weight={
            "loss_nucleus_ce": 1,
            "loss_nucleus_dice": 1,
            "loss_type_ce": 1,
            "loss_type_dice": 1,
            "loss_boundary_distance": 1,
            "loss_boundary": 1,
        }
        def dice_loss(pred, true, focus=None):
            """`pred` and `true` must be of torch.float32. Assuming of shape NxHxWxC."""
            inse = torch.sum(pred * true, (0, 1, 2))
            l = torch.sum(pred, (0, 1, 2))
            r = torch.sum(true, (0, 1, 2))
            loss = 1.0 - (2.0 * inse + 1e-3) / (l + r + 1e-3)
            loss = torch.mean(loss)

            return loss

        def boundary_loss(pred, true):
            pred = pred > 0
            B, H, W = pred.size()
            
            max_radius = true.max()

            loss_sum = 0
            for i in range(2, max_radius+1):
                loss_sum += pred[true==i].sum()*(i-1)
            loss_sum += pred[true==0].sum()*max_radius
            return (loss_sum + 1e-3) / (pred.sum()*max_radius + 1e-3) 


        # def min_boundary_dist_loss(pred, true, boundary_radius=5):
            
        #     B, H, W = pred.size()

        #     boundary_index = list(torch.where(pred))
        #     boundary_total = len(boundary_index[0])

        #     match_index = true[boundary_index[0], boundary_index[1], boundary_index[2]]
            
        #     loss_sum = 0
        #     for radius in range(1, boundary_radius):
        #         boundary_index[0] = boundary_index[0][~match_index]
        #         boundary_index[1] = boundary_index[1][~match_index]
        #         boundary_index[2] = boundary_index[2][~match_index]

        #         if boundary_index[0].shape[0] == 0:
        #             break;

        #         index_x1, index_x2 = boundary_index[2]-radius, boundary_index[2]+radius
        #         index_y1, index_y2 = boundary_index[1]-radius, boundary_index[1]+radius
        #         index_x1 = torch.clamp(index_x1, min=0, max=W-1).type(torch.long)
        #         index_x2 = torch.clamp(index_x2, min=0, max=W-1).type(torch.long)
        #         index_y1 = torch.clamp(index_y1, min=0, max=H-1).type(torch.long)
        #         index_y2 = torch.clamp(index_y2, min=0, max=H-1).type(torch.long)

        #         match_index = torch.zeros_like(boundary_index[0]).type(torch.bool)
        #         for i in range(boundary_index[0].shape[0]):
        #             tmp = true[boundary_index[0][i], index_y1[i]:index_y2[i]+1, index_x1[i]:index_x2[i]+1].sum() > 0
        #             loss_sum += tmp * radius
        #             match_index[i] = tmp

        #     if boundary_index[0].shape[0] > 0:
        #         loss_sum += len(boundary_index[0][~match_index])*boundary_radius

        #     return (loss_sum + 1e-3)/(boundary_total * boundary_radius + 1e-3)


        device = out["boundary_distance_map"].device

        if not self.pretrained:
            nucleus_pred = out["nucleus_map"]
            type_pred = out["type_map"]
            boundary_pred = out["boundary_map"]
        boundary_distance_pred = out["boundary_distance_map"]

    
        type_true, boundary_distance_true, boundary_true = feed_dict["type_map"], feed_dict["boundary_distance_map"], feed_dict["boundary_map"]
        type_true = type_true.type(torch.long).to(device)
        nucleus_true = (type_true > 0).type(torch.long)
        boundary_distance_true = boundary_distance_true.type(torch.float32).to(device)
        boundary_true = boundary_true.type(torch.long).to(device)


        ce_loss = nn.CrossEntropyLoss()
        mse_loss = nn.MSELoss()
        sl1_loss = nn.SmoothL1Loss()


        loss_dict = {}
        if not self.pretrained:
            # nucleus_loss
            loss_dict["loss_nucleus_ce"] = ce_loss(nucleus_pred, nucleus_true)
            loss_dict["loss_nucleus_dice"] = dice_loss(F.softmax(nucleus_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(nucleus_true, num_classes=nucleus_pred.shape[1]).type(torch.float32))

            # type_loss
            loss_dict["loss_type_ce"] = ce_loss(type_pred, type_true)
            loss_dict["loss_type_dice"] = dice_loss(F.softmax(type_pred.permute(0, 2, 3, 1), dim=-1), F.one_hot(type_true, num_classes=type_pred.shape[1]).type(torch.float32))
            
            # boundary_loss
            # loss_dict["loss_boundary_dice"] = dice_loss(boundary_pred>=1, boundary_true==3)
            loss_dict["loss_boundary"] = boundary_loss(boundary_pred, boundary_true)
            # loss_dict["loss_boundary_dice"] = min_boundary_dist_loss(boundary_pred>=1, boundary_true==3, boundary_radius=4)



        # boundary_distance_loss
        # loss_dict["loss_boundary_distance"] = mse_loss(boundary_distance_pred.permute(0, 2, 3, 1)[type_true>0,:], boundary_distance_true[type_true>0,:]) + (torch.pow(boundary_distance_pred.permute(0, 2, 3, 1)[type_true==0,:], 2)).mean()
        loss_dict["loss_boundary_distance"] = sl1_loss(boundary_distance_pred.permute(0, 2, 3, 1)[type_true>0,:], boundary_distance_true[type_true>0,:]) + (boundary_distance_pred.permute(0, 2, 3, 1)[type_true==0,:]).mean()


        overall_loss = 0
        for key in loss_dict:
            loss_dict[key] *= loss_weight[key]
            overall_loss += loss_dict[key]

        return overall_loss, loss_dict

        
   


    def get_ann(self, out, instance_threshold=[8, 10000], neigh_dist_threshold=3, boundary_threshold=3):
        "获取ann"
        type_map = out["type_map"]
        boundary_map = out["boundary_map"]

        B, H, W = boundary_map.size()

        type_map = type_map.to("cpu").data.numpy()
        type_map = np.argmax(type_map, axis=1)

        # 根据阈值选择边界
        boundary_map = boundary_map.to("cpu").data.numpy()
        boundary_map = boundary_map >= boundary_threshold

        # 利用最近邻将边界分配到核实例中
        Neigh = NearestNeighbors(n_neighbors=1)

        ann_list = []

        for b in range(B):

            ann_map = np.zeros((H, W, 2))

            instance_map, _ = ndimage.label(1-boundary_map[b])
            
            # 全除边缘和背景
            instance_map -= 1
            instance_map[instance_map==-1] = 0
          
            # 去除连通较小和较大的
            instance_map_unique, instance_map_count, = np.unique(instance_map, return_counts=True)
            for invalid_instance in instance_map_unique[instance_map_count<=instance_threshold[0]]:
                instance_map[instance_map==invalid_instance] = 0
            for invalid_instance in instance_map_unique[instance_map_count>instance_threshold[1]]:
                instance_map[instance_map==invalid_instance] = 0

            X_instance_index = np.vstack(np.where(instance_map!=0)).T
            Y_instance = instance_map[X_instance_index[:,0], X_instance_index[:,1]]
            
            if X_instance_index.shape[0] > 0:

                # 通过最近邻将边界加入连通实例中，但是如果边界离连通实例太远则剔除
                Neigh.fit(X_instance_index)
                X_boundary_index = np.vstack(np.where(boundary_map[b]==1)).T
                neigh_dist, neigh_ind = Neigh.kneighbors(X_boundary_index, return_distance=True)
                vlaid_boundary_index = np.where(neigh_dist <= neigh_dist_threshold)[0]
                instance_map[X_boundary_index[vlaid_boundary_index, 0], X_boundary_index[vlaid_boundary_index, 1]] = Y_instance[neigh_ind[vlaid_boundary_index]].reshape(-1)

                # 6.10 去除不连通(4)的点, 有一点提升
                temp = np.zeros((4, H, W))
                temp[0, :-1, :] = instance_map[1:, :]
                temp[1, 1:, :] = instance_map[:-1, :]
                temp[2, :, :-1] = instance_map[:, 1:]
                temp[3, :, 1:] = instance_map[:, :-1]
                temp = temp.sum(axis=0)
                instance_map[temp==0] = 0

                ann_map[..., 0] = instance_map

                for i in np.unique(instance_map)[1:]:
                    type_list = type_map[b, instance_map==i]
                    # 去除背景0
                    type_list = type_list[type_list > 0]
                    if type_list.shape[0]>0:
                        # 以概率的形式选择细胞类型
                        # ann_map[instance_map==i, 1] = np.random.choice(type_list)
                        # 直接取众数
                        ann_map[instance_map==i, 1] = np.argmax(np.bincount(type_list)) 
                    else:
                        # 如果没有类别，则表示实例识别错误
                        ann_map[instance_map==i, 0] = 0

            ann_list.append(ann_map)


        ann_list = np.array(ann_list)

        return ann_list.astype("int32")
            

