# RepSNet

**Paper**: RepSNet: A Nucleus Instance Segmentation model based on Boundary Regression and Structural Re-parameterization<!-- (https://arxiv.org/abs/2004.01888) -->

<!-- RepSNet 提出了一种简单有效的细胞核实例分割方法，缓解了密集粘连实例分割的挑战。通过使用共享的特征提取网络，RepSNet在保持高效分割的同时，实现了细胞核分类的高精度。 -->
RepSNet proposes a simple and effective method for nucleus instance segmentation, addressing the challenges of dense and overlapping instance segmentation. By utilizing a shared encoder, RepSNet achieves high segmentation efficiency while maintaining high accuracy in nucleus classification.
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

## Test the performance on CoNIC([download CoNIC](https://drive.google.com/file/d/1fiUjPj3lS1UjKgzTwJ5A8iG1NNKJ6VG8/view?usp=drive_link)) test set

| Test set      | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| local test set  | 0.672 | 0.837 | 0.641 | 0.539 |
| online test set | - | - | 0.635 | 0.478  |

**AJI**: Aggregated Jaccard Index, assesses the overlap between the segmentation and ground truth.  
**DICE**: Dice Coefficient, measures the similarity between the segmentation and ground truth.  
**PQ**: Panoptic Quality, combines segmentation and detection accuracy.  
**mPQ**: Mean Panoptic Quality, evaluates the average segmentation performance across multiple categories.
<!-- - **AJI**: 聚合 Jaccard 指数，评估分割与真实分割的重叠度。  
- **DICE**: Dice 系数，衡量分割与真实分割的相似度。 
- **PQ**: 全景质量，结合分割和检测的准确性。  
- **mPQ**: 平均全景质量，评估多个类别的平均分割性能。 -->

## Installation

1. Clone the repository and install dependencies:
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

## Data preparation

<!-- 1. 下载 CoNIC 数据集 ([CoNIC](https://github.com/TissueImageAnalytics/CoNIC)) 并放置在 `datasets` 目录下: -->
1. Download the CoNIC([download CoNIC](https://drive.google.com/file/d/1fiUjPj3lS1UjKgzTwJ5A8iG1NNKJ6VG8/view?usp=drive_link)) dataset and place it in the datasets directory.
    ```
   dataset/
            └── CoNIC/
    ```
2. Use the methods in the augmend folder for data augmentation, including oversampling, contrast enhancement, elastic deformation, etc.
    ```
   augmend/
            └── oversampling.py
            └── transforms/
    ```


## Pretrained models

<!-- 下载以下预训练模型，并将其放置在 `model_log/` 文件夹下： -->
Download the following pre-trained models and place them in the `model_log/` folder.
<!-- | Test         | AJI | DICE | PQ | mPQ  |
|--------------|------|------|-----|------|
| local_test   | 0.672 | 0.837 | 0.641 | 0.539 |
| local_test   | - | - | 0.635 | 0.478  | -->

The RepSNet_lager model 'RepSNet_L.pth' can be downloaded here: [RepSNet_L](https://drive.google.com/file/d/1082dGUDeGQQwiOxylXmgU5ueArpGs2Ib/view?usp=sharing)

The RepSNet_small model 'RepSNet_S.pth' can be downloaded here:  [RepSNet_S](https://drive.google.com/file/d/18lhFa5bRXi3WFBwHJap2Q9m365FX93Wu/view?usp=drive_link)

## Train

<!-- 1. 使用默认参数训练 RepSNet 模型： -->
1. Train the RepSNet model using default parameters：
    ```bash
    python train.py
    ```

<!-- 2. 可通过以下命令查看更多训练参数： -->
2. You can view more training parameters with the following command：
    ```bash
    python train.py --help
    ```

## Test

<!-- 1. 在 CoNIC 数据集上进行测试： -->
1. Test the RepSNet model using default parameters：
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

## Qualitative results

![RepSNet Qualitative Results1](/results/Qualitative_analysis1.png)

<!-- ![RepSNet Qualitative Results1](/results/Qualitative_analysis2.png) -->


## Acknowledgement
The code is based on [StarDist](https://github.com/stardist/augmend) and [HoVer-Net](https://github.com/vqdang/hover_net). Thanks for their wonderful works.
## Citation

<!-- 如果你在研究中使用了我们的工作，请引用以下论文： -->
If you use our work in your research, please cite the following paper:

```bibtex
@article{,
  title={RepSNet: A Nucleus Instance Segmentation model based on
Boundary Regression and Structural Re-parameterization},
  author={},
  journal={},
  year={2024}
}
