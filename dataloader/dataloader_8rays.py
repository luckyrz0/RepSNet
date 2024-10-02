import numpy as np
import cv2
import pickle

from dataloader.img_augs import img_augs
from utils.utils import get_bounding_box
import torch.utils.data as Data
import json
from scipy.ndimage import measurements
import warnings
import albumentations as A
from .conic import *
from imgaug import augmenters as iaa
from augmend import Augmend
from stardist import star_dist,edt_prob

transforms = A.Compose([
    A.GaussianBlur(p=0.3), 
    # A.MotionBlur(p=0.3),
    # A.AdditiveNoise(limit=(-0.01, 0.01), per_channel=True, p=0.5),
    A.ColorJitter(brightness=0.1, saturation=(1,1), hue=0, p=0.3),
    # A.OneOf([A.CLAHE(clip_limit=2), A.IAAEmboss()], p=0.3)
])

#########################2023年8月19号
aug = Augmend()

aug.add([HEStaining(amount_matrix=0.15, amount_stains=0.4), Identity()], probability=0.3)
 
aug.add([Elastic(grid=5, amount=10, order=1, axis=(0,1), use_gpu=False),
         Elastic(grid=5, amount=10, order=0, axis=(0,1), use_gpu=False)], probability=0.3)

# aug.add([GaussianBlur(amount=(0,2), axis=(0,1), use_gpu=False), Identity()], probability=0.3)  

# aug.add([AdditiveNoise(0.01), Identity()], probability=0.3)

# aug.add([HueBrightnessSaturation(hue=0, brightness=0.1, saturation=(1,1)), Identity()], probability=0.9)
#########################

class dataset(Data.Dataset):

    def __init__(
        self,
        img_path,
        ann_path,
        run_mode="train",
        indices=None,
        with_augs=False,
        with_instances_aug=None,
        boundary_mode='inside',
        max_dist =15,
    ):
        self.max_dist =max_dist
        self.run_mode = run_mode
        self.with_augs = with_augs
        self.with_instances_aug = with_instances_aug is not None
        self.boundary_mode = boundary_mode
        self.imgs = np.load(img_path, mmap_mode='r')
        self.anns = np.load(ann_path, mmap_mode='r')
        if self.anns.shape[-1] > 1:
            self.with_type = True
        else:
            self.with_type = False
        if self.with_instances_aug:
            self.synthesize_data = pickle.load(open(with_instances_aug, 'rb'))
        if indices is not None:
            self.imgs = self.imgs[indices]
            self.anns = self.anns[indices]

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):

        img = np.array(self.imgs[idx]).astype("uint8")
        ann = np.array(self.anns[idx]).astype("int32")

        # if self.resize:
        #     H, W, C = img.shape
        #     size_decrease = (int(W/2), int(H/2))
        #     img = cv2.resize(img, size_decrease)
        #     ann = ann[::2, ::2]

        if self.run_mode == "train":
            if self.with_instances_aug:
                img, ann = self.__add_instance_aug(img, ann)

            img, ann = self.__img_augmentation(img, ann)
        ################ 2023 7 25 拉普拉斯增强
        # img = np.concatenate([img, np.array([cv2.Laplacian(img[:, :, i], cv2.CV_64F, ksize=3) for i in range(3)]).transpose(1, 2, 0)], axis=-1)
        ################
        feed_dict = {
            "img": img.copy(),
            "ann": ann.copy(),
        }

        type_map, boundary_map = self.__get_target(ann, dir_mode=4)
        boundary_distance_map = star_dist(ann[...,0],8)

        if self.max_dist:
            boundary_distance_map[boundary_distance_map>self.max_dist] = self.max_dist

        # boundary_distance_map = np.transpose(boundary_distance_map,(2,0,1))


        feed_dict["type_map"] = type_map.copy()

        feed_dict["boundary_distance_map"] = np.sqrt(boundary_distance_map.copy())

        feed_dict["boundary_map"] = boundary_map.copy()


        return feed_dict

    def __add_instance_aug(self, img, ann):
        # 添加实例，进行数据增强
        assert self.with_instances_aug

        inst_num = ann[..., 0].max() + 1
        sample_num = 20
        sample_index = np.random.choice(range(len(self.synthesize_data)), sample_num)
        for idx in sample_index:
            inst = self.synthesize_data[idx]
            rot_index = np.random.randint(0, 5)
            inst = np.rot90(inst, rot_index)
            while_count = 0
            while while_count < 100:
                y_start = np.random.randint(0, img.shape[0] - inst.shape[0])
                y_end = y_start + inst.shape[0]
                x_start = np.random.randint(0, img.shape[1] - inst.shape[1])
                x_end = x_start + inst.shape[1]

                # 保证不接触
                if (ann[y_start:y_end, x_start:x_end, 0]).max() == 0:
                    temp = img[y_start:y_end, x_start:x_end]
                    temp[inst[..., 3] > 0, :] = inst[inst[..., 3] > 0, :3]
                    img[y_start:y_end, x_start:x_end] = temp
                    ann[y_start:y_end, x_start:x_end, 0] = inst[..., 3] * inst_num
                    if self.with_type:
                        ann[y_start:y_end, x_start:x_end, 1] = inst[..., 4]
                    inst_num += 1
                    break

                # 超过最大上限跳出
                while_count += 1

        return img, ann

    def __img_augmentation(self, img, ann):
        # 改变形状
        if np.random.rand() < 0.5:
            k = np.random.randint(1, 4)
            img = np.rot90(img, k=k, axes=(0, 1))
            ann = np.rot90(ann, k=k, axes=(0, 1))

        if np.random.rand() < 0.5:
            img = np.fliplr(img)
            ann = np.fliplr(ann)

        if np.random.rand() < 0.5:
            img = np.flipud(img)
            ann = np.flipud(ann)

        # 目前来说数据增强效果不明显
        if self.with_augs:
            img,ann = aug([img,ann])
            # augs = img_augs.to_deterministic()
            # img = augs.augment_image(img)
            img = transforms(image = img)["image"]
           

        

        return img, ann


    # 原来的方法
    def __get_target(self, ann, dir_mode=4):
        inst_map = ann[..., 0]

        if self.with_type:
            type_map = ann[..., 1]
        else:
            type_map = (inst_map > 0).astype("int32")

        # boundary_distance_map的mask是跟inst_map重合的
        # boundary_distance_map = np.zeros((inst_map.shape[0], inst_map.shape[0], 8)) # 0 45 90 135 180 225 270 315 

        # for i in range(inst_map.shape[0]):
        #         for j in range(inst_map.shape[1]):
        #             value = inst_map[i,j]
        #             if value == 0:
        #                 continue
        #             else:
        #                 st_rays = np.float32((2*np.pi) / 8)
        #                 for k in range(8):
        #                     phi = np.float32(k*st_rays)
        #                     dy = np.cos(phi)
        #                     dx = np.sin(phi)
        #                     x, y = np.float32(0), np.float32(0)
        #                     while True:
        #                         x += dx
        #                         y += dy
        #                         ii = int(round(i+x))
        #                         jj = int(round(j+y))
        #                         if (ii < 0 or ii >= inst_map.shape[0] or
        #                             jj < 0 or jj >= inst_map.shape[1] or
        #                             value != inst_map[ii,jj]):
        #                             # small correction as we overshoot the boundary
        #                             t_corr = 1-.5/max(np.abs(dx),np.abs(dy))
        #                             x -= t_corr*dx
        #                             y -= t_corr*dy
        #                             dist = np.sqrt(x**2+y**2)
        #                             boundary_distance_map[i,j,k] = dist
        #                             break


       
        boundary_map = self.__get_boundary_map(inst_map, boundary_mode=self.boundary_mode)

        return type_map, boundary_map

    # 2022.10.20
    def __get_boundary_map(self, inst_map, boundary_radius=5, boundary_mode='outside'):
        """
        获取边界map
        每个像素的值表示离边界的距离+1
        """
        inst_list = list(np.unique(inst_map))
        inst_list.remove(0)

        boundary_map = np.zeros_like(inst_map, dtype=np.uint8).copy()

        kernel = np.ones((3, 3), np.uint8)

        for inst_idx, inst_id in enumerate(inst_list):
            inst_map_mask = np.array(inst_map == inst_id, np.uint8)

            if boundary_mode == 'inside':
                # 内边界
                erode = cv2.erode(inst_map_mask, kernel, iterations=1)
                boundary = inst_map_mask - erode
            else:
                # 外边界
                dilate = cv2.dilate(inst_map_mask, kernel, iterations=1)
                boundary = dilate - inst_map_mask

            boundary_map |= boundary

        temp = boundary_map.copy()
        for i in range(boundary_radius - 1):
            temp = cv2.dilate(temp, kernel, iterations=1)
            boundary_map += temp

        boundary_map[boundary_map == 0] = boundary_radius + 1
        boundary_map = boundary_radius + 1 - boundary_map
        
        return boundary_map

class dataloader(object):

    def __init__(
        self,
        dataset_name="CoNIC",  # ['kumar', 'cpm17', 'consep', 'CoNIC', 'PanNuke', 'dsb18]
        dataset_path="./dataset/unify_dataset/",
        batch_size=8,
        with_augs=False,
        with_instances_aug=False,
        boundary_mode='inside',
    ):
        self.dataset_name = dataset_name
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.with_augs = with_augs
        self.with_instances_aug = with_instances_aug
        self.boundary_mode = boundary_mode

    def getDataloader(self, seed=0):
        
            seed %= 5
            with open('/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/splits.json', 'r') as f:
                splits = json.load(f)
            split = splits[seed]

            train_dataset = dataset(
                 img_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/images.npy",
                ann_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/labels.npy",
                run_mode="train", indices=split["train"], 
                with_augs=self.with_augs, 
                with_instances_aug=(self.dataset_path + "CoNIC/instances.pkl") if self.with_instances_aug else None)

            valid_dataset = dataset(
                img_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/images.npy",
                ann_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/labels.npy",
                run_mode="test",
                indices=split["valid"],
                with_augs=False,
            )

            test_dataset = dataset(
                img_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/images.npy",
                ann_path="/home/DM22/workspace/CoNIC/RepSNet/dataset/unify_dataset/CoNIC/labels.npy",
                run_mode="test",
                indices=split["test"],
                with_augs=False,
            )

            train_dataloader = Data.DataLoader(
                dataset=train_dataset,
                batch_size=self.batch_size,
                shuffle=True,
                num_workers=2,
            )

            valid_dataloader = Data.DataLoader(
                dataset=valid_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
            )

            test_dataloader = Data.DataLoader(
                dataset=test_dataset,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=2,
            )

            dataloader = {
                "train": train_dataloader,
                "valid": valid_dataloader,
                "test": test_dataloader,
            }

            return dataloader

