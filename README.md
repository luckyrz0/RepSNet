# RepSNet

**Paper**: RepSNet: A Nucleus Instance Segmentation model based on Boundary Regression and Structural Re-parameterization<!-- (https://arxiv.org/abs/2004.01888) -->

## Abstract


## Model


## 训练表现

| Dataset      | MOTA | IDF1 | IDs | FPS  |
|--------------|------|------|-----|------|
| MOT17        | 74.9 | 72.8 | 312 | 30   |
| MOT20        | 74.5 | 71.9 | 322 | 25   |

- **MOTA**: 多目标跟踪准确率  
- **IDF1**: 识别 F1 得分  
- **IDs**: 身份转换次数  
- **FPS**: 每秒帧数

## 安装步骤

1. 克隆仓库并安装依赖：
    ```bash
    git clone https://github.com/ifzhang/FairMOT.git
    cd FairMOT
    pip install -r requirements.txt
    ```

2. 编译 DLA 模型：
    ```bash
    cd src/lib/models/networks
    sh make.sh
    cd ../../../
    ```

## 数据准备

1. 下载 MOT 数据集 (如 [MOT17](https://motchallenge.net/data/MOT17/)) 并放置在 `datasets` 目录下:
    ```
    FairMOT/
      └── datasets/
            └── MOT17/
    ```

2. 转换数据集格式以适应训练：
    ```bash
    python src/tools/convert_mot17_to_coco.py
    ```

## 预训练模型

下载以下预训练模型，并将其放置在 `models/` 文件夹下：

- [DLA-34 模型](https://github.com/ifzhang/FairMOT/releases/download/v1.0/dla34.pth)
- [FairMOT 模型](https://github.com/ifzhang/FairMOT/releases/download/v1.0/fairmot.pth)

## 训练

1. 使用默认参数训练 FairMOT 模型：
    ```bash
    python src/train.py mot --exp_id fairmot_dla34 --gpus 0
    ```

2. 可通过以下命令查看更多训练参数：
    ```bash
    python src/train.py --help
    ```

## 测试

1. 在 MOT17 数据集上进行跟踪测试：
    ```bash
    python src/track.py mot --load_model ../models/fairmot.pth --conf_thres 0.4
    ```

2. 结果将保存到 `results/` 文件夹中。

## Demo 演示

1. 使用预训练模型进行视频跟踪：
    ```bash
    python demo.py --input_video path_to_video.mp4 --load_model ../models/fairmot.pth
    ```

2. 结果将输出到 `results/` 文件夹中。

## 自定义数据集训练

1. 将自定义数据集转换为 COCO 格式：
    ```bash
    python src/tools/convert_to_coco.py --dataset custom
    ```

2. 使用与 MOT 数据集相同的步骤进行训练。

## 引用

如果你在研究中使用了我们的工作，请引用以下论文：

```bibtex
@article{zhang2020fairmot,
  title={FairMOT: On the Fairness of Detection and Re-Identification in Multiple Object Tracking},
  author={Zhang, Yifu and Wang, Changhong and Wang, Xiaogang and Liu, Wei},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
