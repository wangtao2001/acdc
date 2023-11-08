# ACDC 心脏数据集 语义分割

## 1.背景

ACDC (Automatic Cardiac Diagnosis Challenge) 是 MICCAI 2017 中的的挑战赛之一，旨在对心脏动态磁共振成像 (cine-MRI) 中的舒张期 (ED) 和收缩期 (ES) 帧进行左心室 (LV) 、右心室 (RV) 和心肌 (Myo) 分割。精确分割心脏图像对于评估心脏功能，如射血分数（EF）、每次搏动的血量（SV）、左心室质量和心肌厚度，这些进而为诊断和治疗心脏疾病提供关键信息。


> MICCAI 是 **医学图像计算与计算机辅助干预国际会议** （International Conference on Medical Image Computing and Computer Assisted Intervention）的缩写。该会议自 1998 年开始，每年举办一次，是医学图像处理领域中最具有权威性和影响力的会议之一，被广泛认为是医学图像处理领域内的顶级会议之一。而 MICCAI CHALLENGES 是 MICCAI 的一个重要组成部分。它是一个国际性的竞赛平台，面向医学图像计算和计算机辅助干预领域的研究人员和开发者，旨在鼓励和推动该领域的技术发展和应用。

## 2.数据集

该数据集涵盖 150 个病例，分为 5 个子类：NOR (正常)、MINF (心肌梗死伴随收缩性心力衰竭)、DCM (扩张型心肌病)、HCM (肥厚型心肌病) 和 ARV (右室异常)，每类各 30 例。每一病例都包括一个心脏周期的 4D nifti格式图像，并且标注了舒张末期 (ED) 与收缩末期 (ES) 帧。官方将数据划分为 100 例的训练集和 50 例的测试集，每个子类在训练集中有 20 例，在测试集中有 10 例。所有 150 例数据和标注都已公开。

> https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb


#### 数据集说明：


```
acdc_challenge_20170617/ 
    training/ -训练集
        patient001/ -病例 001-100
            Info.cfg -病例说明
            patient001_4d.nil.gz -整体影像（4d），包含`NbFrame`帧
            patient001_frame01.nii.gz -舒张末期单帧
            patient001_frame01_gt.nii.gz -对应的label
            patient001_frame12.nii.gz -收缩末期单帧
            patient001_frame12_gt.nii.gz -对应的label
        ...
    testing/ -测试集
    ...

```


#### NIFTI 格式说明：
标准NIFTI图像（扩展名是.nii），其中包含了头文件h（hdr）及图像资料（img）。单独的.nii格式文件的优势就是可以用标准的压缩软件（如gzip），而且一些分析软件包可以直接读取和写入压缩的.nii文件（扩展名为.nii.gz）。

使用软件预览：itk-snap

使用python读取.nii（.nii.gz）文件：`loader.py`

## 3.数据预处理

#### 重采样

体素即体积元素（Volume Pixel），一张3d图像可以看成是由若干个体素构成的，体素是一张3d医疗图像在空间上的最小单元。这里将体素大小修改为 1.37mm * 1.37mm，并且保持深度不变。

#### 切片

这里只考虑了ED时期的数据，将水平面的全层抽取。对于输出的图像大小不一致的问题，可能的策略包括：填充0值、切分patch、resize、裁剪...这里选择将图片填充0值到 384* 384。

上述代码见 `convert.py`。

#### 数据增强

待补充...

## 4.模型

#### Model 1: UNet

UNet是典型的Encoder-Decoder结构，在Encoder中先对图片进行卷积和池化，然后对特征图做上采样或者反卷积，同时对之前的特征图进行通道上的拼接concat。UNet网络层越深得到的特征图，有着更大的视野域，同时通过特征的拼接，来实现边缘特征的找回。

## 5.评估指标

Dice：

## 6.训练

## 7.结果

## 8.参考

[1] https://zhuanlan.zhihu.com/p/658483739

[2] https://blog.csdn.net/lavinia_chen007/article/details/125389503

[3] https://blog.csdn.net/cvxiayixiao/article/details/133232554

[4] https://arxiv.org/abs/1709.04496
