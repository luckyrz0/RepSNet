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
        max_dist =None,
        n_rays =32
    ):
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
        self.max_dist = max_dist
        self.n_rays =n_rays

    def __len__(self):
        return self.imgs.shape[0]

    def __getitem__(self, idx):

        img = np.array(self.imgs[idx]).astype("uint8")
        ann = np.array(self.anns[idx]).astype("int32") #(256,256,2)
        distances = star_dist(ann[...,0],self.n_rays)

        if self.max_dist:
            distances[distances>self.max_dist] = self.max_dist

        distances = np.transpose(distances,(2,0,1))

        obj_probabilities = edt_prob(ann[...,0])

        obj_probabilities = np.expand_dims(obj_probabilities,0)

        # if self.run_mode == "train":
        #     if self.with_instances_aug:
        #         img, ann = self.__add_instance_aug(img, ann)

        #     img, ann = self.__img_augmentation(img, ann)
        
        type_map = self.__get_target(ann)
       
        feed_dict = {
            "img": img.copy(),
            "prob": obj_probabilities,
            "dist":distances,
            "ann" :ann.copy(),
            "type_map":type_map
        }

        return feed_dict
        # return img,obj_probabilities,distances

    def __get_target(self, ann):
        inst_map = ann[..., 0]

        if self.with_type:
            type_map = ann[..., 1]
        else:
            type_map = (inst_map > 0).astype("int32")

        return type_map


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

            self.dataset_name == "CoNIC"
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