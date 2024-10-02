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

| Test set      | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| local test set  | 0.672 | 0.837 | 0.641 | 0.539 |
| online test set | - | - | 0.635 | 0.478  |

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

下载以下预训练模型，并将其放置在 `model_log/` 文件夹下：

<!-- | Test         | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| local_test   | 0.672 | 0.837 | 0.641 | 0.539 |
| local_test   | - | - | 0.635 | 0.478  | -->

The RepSNet_lager model 'best_mpq_lager.pth' can be downloaded here: [RepSNet_L](https://drive.google.com/file/d/1082dGUDeGQQwiOxylXmgU5ueArpGs2Ib/view?usp=sharing)

The RepSNet_lager model 'best_mpq_samll.pth' can be downloaded here:  [RepSNet_S](https://github.com/ifzhang/FairMOT/releases/download/v1.0/dla34.pth)

## 训练

1. 使用默认参数训练 RepSNet 模型：
    ```bash
    python train.py
    ```

2. 可通过以下命令查看更多训练参数：
    ```bash
    python train.py --help
    ```

## 测试

1. 在 CoNIC 数据集上进行测试：
    ```bash
    python test.py
    ```

<!-- ## Quantitative Results

| Method     | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| U-Net      | 0.518 | 0.800 | 0.505 | 0.411 |
| DCAN       | 0.636 | 0.815 | 0.590 | 0.480 |
| Hover-Net       | 0.663 | 0.830 | 0.628 | 0.531 |
| StarDist       | 0.671 | 0.837 | 0.634 | 0.547 |
| RepSNet       | 0.683 | 0.841 | 0.641 | 0.563 | -->

## Qualitative Results

![RepSNet Qualitative Results1](/results/Qualitative_analysis1.png)

![RepSNet Qualitative Results1](/results/Qualitative_analysis2.png)

## 引用

如果你在研究中使用了我们的工作，请引用以下论文：

```bibtex
@article{zhang2020fairmot,
  title={RepSNet: A Nucleus Instance Segmentation model based on
Boundary Regression and Structural Re-parameterization},
  author={Zhang, Yifu and Wang, Changhong and Wang, Xiaogang and Liu, Wei},
  journal={arXiv preprint arXiv:2004.01888},
  year={2020}
}
