# RepSNet

**Paper**: RepSNet: A Nucleus Instance Segmentation model based on Boundary Regression and Structural Re-parameterization<!-- (https://arxiv.org/abs/2004.01888) -->

RepSNet 提出了一种简单有效的细胞核实例分割方法，缓解了密集粘连实例分割的挑战。通过使用共享的特征提取网络，RepSNet在保持高效分割的同时，实现了细胞核分类的高精度。

## Pipeline

![RepSNet Pipeline](/assets/image.png)


<!-- FairMOT 使用了 DLA-34 作为 backbone，同时为检测和 Re-ID 任务提供了统一的特征表示。网络结构如下图所示：
![RepSNet Pipeline](/assets/model_structure.png)
1. DLA-34 backbone
2. Shared head for detection and Re-ID
3. Multi-scale feature fusion for fairness -->

<!-- ## 更新记录

- 2020/04/22: 发布了 FairMOT 初版代码
- 2020/06/10: 更新了自定义挑战数据集上的性能表现 -->

## Test performance

| Dataset      | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| local_test   | 0.672 | 0.837 | 0.641 | 0.539 |
| online_test  | - | - | 0.635 | 0.478  |

- **AJI**: 聚合 Jaccard 指数，评估分割与真实分割的重叠度。  
- **DICE**: Dice 系数，衡量分割与真实分割的相似度。 
- **PQ**: 全景质量，结合分割和检测的准确性。  
- **mPQ**: 平均全景质量，评估多个类别的平均分割性能。

## 安装步骤

1. 克隆仓库并安装依赖：
    ```bash
    git clone https://github.com/luckyrz0/RepSNet.git
    cd RepSNet
    pip install -r requirements.txt
    ```

<!-- 2. 编译 DLA 模型：
    ```bash
    cd src/lib/models/networks
    sh make.sh
    cd ../../../
    ``` -->

## 数据准备

1. 下载 CoNIC 数据集 ([CoNIC](https://github.com/TissueImageAnalytics/CoNIC)) 并放置在 `datasets` 目录下:
    ```
   dataset/
            └── CoNIC/
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
