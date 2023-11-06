# ACDC 心脏数据集 语义分割

## 1.背景

ACDC (Automatic Cardiac Diagnosis Challenge) 是 MICCAI 2017 中的的挑战赛之一，旨在对心脏动态磁共振成像 (cine-MRI) 中的舒张期 (ED) 和收缩期 (ES) 帧进行左心室 (LV) 、右心室 (RV) 和心肌 (Myo) 分割。精确分割心脏图像对于评估心脏功能，如射血分数（EF）、每次搏动的血量（SV）、左心室质量和心肌厚度，这些进而为诊断和治疗心脏疾病提供关键信息。


> MICCAI 是 **医学图像计算与计算机辅助干预国际会议** （International Conference on Medical Image Computing and Computer Assisted Intervention）的缩写。该会议自 1998 年开始，每年举办一次，是医学图像处理领域中最具有权威性和影响力的会议之一，被广泛认为是医学图像处理领域内的顶级会议之一。而 MICCAI CHALLENGES 是 MICCAI 的一个重要组成部分。它是一个国际性的竞赛平台，面向医学图像计算和计算机辅助干预领域的研究人员和开发者，旨在鼓励和推动该领域的技术发展和应用。

## 2.数据集

该数据集涵盖 150 个病例，分为 5 个子类：NOR (正常)、MINF (心肌梗死伴随收缩性心力衰竭)、DCM (扩张型心肌病)、HCM (肥厚型心肌病) 和 ARV (右室异常)，每类各 30 例。每一病例都包括一个心脏周期的 4D nifti格式图像，并且标注了舒张末期 (ED) 与收缩末期 (ES) 帧。官方将数据划分为 100 例的训练集和 50 例的测试集，每个子类在训练集中有 20 例，在测试集中有 10 例。所有 150 例数据和标注都已公开。

> https://humanheart-project.creatis.insa-lyon.fr/database/#collection/637218c173e9f0047faa00fb


#### 数据集说明：


```
dataset/ 
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

软件预览：itk-snap（基本的使用方法待补充...）

使用python读取：`preview.py`

## 3.数据预处理
