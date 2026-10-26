# DBNet 文本检测：算法思想与 PyTorch 代码实现

DBNet — Differentiable Binarization Network（可微分二值化网络）

论文：[Real-time Scene Text Detection with Differentiable Binarization](https://arxiv.org/abs/1911.08947)，AAAI 2020。

官方代码：[MhLiao/DB (PyTorch)](https://github.com/MhLiao/DB)。

DBNet 是 OCR 文本检测中非常经典的算法，也是理解现代分割式文本检测（Segmentation-based Text Detection）的重要基础。

本文重点是理解算法原理与工程实现，不追求繁琐的数学推导。

## 一、What：DBNet 到底是什么？

一句话：DBNet 将文本检测转换成图像分割问题，通过可微分二值化，让模型学会更准确地分离文字区域，最后通过轮廓提取获得文本框。

传统目标检测，例如 YOLO，通常直接预测物体的位置与边界框。

而 DBNet 不直接回归文本框坐标，而是预测图像中哪些像素属于文本，再从文本区域中提取检测框。

[From Pixels to Words: PaddleOCR Pipeline Explained | by Sai Dheeraj Gummadi | Data Science in Your Pocket | Feb, 2026 | Medium](https://images.openai.com/static-rsc-4/bBcvmehDPzHaLaLx-fCOf5WPivaVfGBLitrVkWuSWQERzG-oaNE-zL5JEoR_-z_ES1-zcoCpzA0RMDeql8PpLEO7yXCAP21srtdBgrUk9T1Z2ngCiqEQkVi5gwST5PFktEOgtl7tS1r0fl-3TLHn2iqG3d3hAGq677WLpVhd1A8?purpose=inline)

输入图像

[OCR-(DB+CRNN)-代码分析-实践环境配置及运行 - Lwd_curent](https://images.openai.com/static-rsc-4/sHI5LhpxlyBKz6BF61W6vRyjzyds94MCeNq9tNDScWavBhqn2qiH3SmtLYfogl0inQM2YiqYftUR7dTAypWH17m_jVUoHjHthBnYOjFSzD8ftoIjezRGh_e2erJF4jKCerDpbr8lcu_Pa21GZx61_XhXYQLvZeTyeqUp9nqDzz4?purpose=inline)

文字概率图

二值化、连通区域、轮廓提取、扩张

[【OCR理论】DBNet:任意形状文本检测模型1911.Real-time Scene Text Detection with Differentiable Binarization_approximate binary map-CSDN博客](https://images.openai.com/static-rsc-4/d-4uchG3LkFD2luqrwYFJ6dHSXc9P6uI-K-KBKEU8HXt3a6f1-K66BTDSu1PhvFcmQicKLNvikelCKYR9S5TDaMBhtJ1qrznUpa8U-Ow_6iNuvtS5JKawszZPi8a-0n0-neiVVYMFVYO43l9WSgcJ5qLyjpmUfzu_oqy8zftozM?purpose=inline)

最终检测框

### 为什么文字检测适合分割？

文字与普通物体有所不同：

- 长宽比变化大：一个单词可能是 1:2，一行文字可能是 1:30。
- 方向不固定：水平、倾斜、竖直都有可能。
- 形状不规则：例如弯曲的招牌文字。
- 排列紧密：相邻文字区域容易粘连。

基于像素的分割方法，可以更灵活地描述这些文字形状。

不过，分割式方法也带来了一个问题：如何把像素准确地转换成彼此独立的文本实例？

这就是 DBNet 重点解决的问题。

## 二、核心创新：Differentiable Binarization（可微分二值化）

这是整篇论文最重要的知识点。

### 1. 传统二值化的问题

假设网络输出一张文本概率图（Probability Map）：

每个像素对应一个 $0\sim1$ 之间的概率。

传统方法直接使用固定阈值：

\\[ B(x,y)= \begin{cases} 1 & P(x,y)\geq 0.5\\\ 0 & P(x,y)<0.5 \end{cases} \\]

例如：

| 像素概率 | 阈值 0.5 | 结果 |
| ---- | ------ | -- |
| 0.9  | 0.5    | 文字 |
| 0.6  | 0.5    | 文字 |
| 0.4  | 0.5    | 背景 |
| 0.1  | 0.5    | 背景 |

问题在于：

1. 不同位置使用相同的阈值，难以适应复杂的文字边界。
2. 硬二值化操作不可微，无法正常通过梯度反向传播来优化前面的网络。

### 2. DBNet 的解决方案

DBNet 让网络同时预测两张图：

- P — Probability Map（概率图）：每个像素属于文字的概率。
- T — Threshold Map（阈值图）：每个像素对应的自适应二值化阈值。

然后通过一个可微分函数产生第三张图：

\- B — Approximate Binary Map（近似二值图）。

核心公式：

\\[ \boxed{\hat B(x,y)=\sigma\left(k(P(x,y)-T(x,y))\right)} \\]

展开为：

\\[ \hat B(x,y)=\frac{1}{1+e^{-k(P(x,y)-T(x,y))}} \\]

其中：

- $\sigma$：Sigmoid 激活函数。
- $P$：预测的文本概率。
- $T$：网络学习的阈值。
- $k$：放大系数，原论文取 50。

### 3. 为什么这样可以实现二值化？

k = 1k = 10k = 50

00.250.50.751-0.2-0.15-0.1-0.075-0.05-0.02500.0250.050.0750.10.150.2

示意曲线：横轴为 P−T，纵轴为近似二值化输出。k 越大，变化越陡峭。

当 $k=50$ 时：

- $P$ 明显大于 $T$：结果接近 1。
- $P$ 明显小于 $T$：结果接近 0。
- $P$ 接近 $T$：结果快速变化。

这样既模拟了传统二值化的阶跃特征，又保留了可微分性质，可以参与反向传播。

更重要的是，DBNet 把二值化放进了训练过程。

二值化产生的损失会反向影响概率图分支与阈值图分支，让网络学习如何更好地分离文字和背景，尤其是相邻、粘连的文本实例。

这才是 DBNet 真正的价值，而不仅仅是把固定阈值换成了动态阈值。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 三、网络结构：Backbone + FPN + DB Head

从网络结构看，DBNet 并不复杂，主要由三部分组成：

Input Image

输入 RGB 图像：[B, 3, H, W]

Backbone（特征提取网络）

ResNet-18 / ResNet-50 等

C2, C3, C4, C5 — 多尺度特征

FPN（Feature Pyramid Network，特征金字塔）

多尺度特征上采样、融合、拼接

Probability Head

概率图 P

Threshold Head

阈值图 T（训练时）

Differentiable Binarization

\\(\hat B=\operatorname{sigmoid}(k(P-T))\\)

训练：P / T / B 联合监督

推理：只需 P

### 1. Backbone：提取图像特征

原论文采用 ResNet-18 或 ResNet-50，并在部分实验中加入 DCN（Deformable Convolution，可变形卷积）。

输出多尺度特征：

| 特征层 | 相对输入分辨率 | 主要作用      |
| --- | ------- | --------- |
| C2  | 1/4     | 文字边缘和细节   |
| C3  | 1/8     | 局部文字特征    |
| C4  | 1/16    | 较高层语义     |
| C5  | 1/32    | 大感受野、全局语义 |

### 2. FPN：融合多尺度信息

FPN 可以理解为把深层语义信息传递给浅层高分辨率特征。

它有利于同时识别小文字、大文字、短单词和长文本行。

在 DBNet 的实现中，四个尺度的特征经过自顶向下融合，再上采样到输入的 1/4 分辨率，最后沿 Channel 维度拼接。

### 3. DB Head：预测概率图和阈值图

两条预测分支结构相近：

`3×3 Conv → BN → ReLU → 2×Upsample → 2×Upsample → Sigmoid`

这样可以从 1/4 分辨率的融合特征恢复到输入图像的分辨率。

一个容易理解错误的地方：

DBNet 训练时通常需要概率图 P 和阈值图 T，但实际部署时可以移除阈值分支和可微分二值化运算，只保留概率图分支。

原因是可微分二值化已经在训练阶段帮助网络学到了更好的文本分割结果。推理时直接对 P 使用固定阈值，通常就能得到很好的结果。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 四、训练数据：为什么要把文本框缩小？

这也是 DBNet 非常重要的设计。

### 1. Probability Map 的监督标签

假设标注数据是一个文本四边形：

```
polygon = [    [100, 100],    [300, 100],    [300, 140],    [100, 140]]
```

DBNet 不直接把这个四边形内部全部标成正样本，而是先将它向内收缩（Shrink）。

为什么收缩？

因为相邻两行文字、两个单词可能靠得很近。如果直接使用原始完整区域，很容易在分割结果中发生粘连。

收缩后，每个文本实例更容易分离。

收缩距离：

\\[ \boxed{D=\frac{A(1-r^2)}{L}} \\]

其中：

- $A$：原始文本多边形面积。
- $L$：多边形周长。
- $r$：收缩比例，原论文取 0.4。

这里的 $r$ 不是直接把长宽分别乘以 0.4，而是用于计算多边形向内偏移的距离。

收缩后的区域是概率图的正样本，其余有效区域为负样本。

### 2. Threshold Map 的监督标签

阈值图的制作稍微特殊。

先将原始文本多边形向外扩张一个距离 D，同时保留向内收缩的区域。

然后根据像素到原始多边形边界的距离，构造阈值标签。

直观上：

- 越靠近文字轮廓边界，标签值越高。
- 离边界越远，标签值越低。
- 只有指定的扩张区域参与阈值图监督。

官方实现通常将目标值缩放到 0.3～0.7 范围。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



可以把它理解为：概率图主要学习文字核心区域，阈值图重点学习文字周围的边界信息。

### 3. 三种损失函数（Loss Function）

原论文定义：

\\[ \boxed{L=L_s+\alpha L_b+\beta L_t} \\]

| Loss  | 作用        | 论文设置                       |
| ----- | --------- | -------------------------- |
| $L_s$ | 概率图分割损失   | BCE + Hard Negative Mining |
| $L_b$ | 近似二值图分割损失 | BCE + Hard Negative Mining |
| $L_t$ | 阈值图回归损失   | Masked L1                  |

原论文采用：

\\[ \alpha=1,\quad \beta=10 \\]

BCE（Binary Cross-Entropy，二元交叉熵）用来学习前景与背景的分类。

Hard Negative Mining（难负样本挖掘）用于解决文本像素少、背景像素多的问题。常见的正负样本采样比例为 1:3。

L1 Loss 用来约束预测阈值图与阈值标签之间的距离。

注意：原始论文采用的损失定义，与后续不同开源实现中的损失配置可能不同。例如官方 PyTorch 仓库也提供 BCE、Dice、L1 的组合版本，不能把所有 DBNet 实现都理解成固定使用同一套 Loss。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



## 五、How：PyTorch 代码实现

我整理了一个可以运行的简化版 DBNet，包含 ResNet-18、FPN、双预测分支、标签生成、损失函数和 OpenCV 后处理。

下载 dbnet_minimal.py

已执行前向传播、Loss 计算、反向传播和推理流程测试。

这是用于理解原理的教学版，不是原论文的完整复现，也没有预训练权重。

### 1. 网络预测的核心代码

最关键的是下面几行：

```
# FPN 输出融合后的特征图 features# 分支1：预测文字概率图P = self.prob_head(features)# 分支2：预测自适应阈值图T = self.thresh_head(features)# 可微分二值化B = torch.sigmoid(50 * (P - T))return {    "prob": P,    "thresh": T,    "approx_binary": B}
```

注意：P、T 和 B 都是与输入图像空间对应的单通道图像，而不是文本框坐标。

### 2. Loss 的实现

```
def db_loss(pred, prob_gt, thresh_gt, thresh_mask, valid_mask):    # 概率图损失：含难负样本挖掘    loss_p = balanced_bce(        pred["prob"], prob_gt, valid_mask    )    # 近似二值图损失    loss_b = balanced_bce(        pred["approx_binary"], prob_gt, valid_mask    )    # 阈值图损失：仅计算监督区域    diff = abs(pred["thresh"] - thresh_gt)    loss_t = (diff * thresh_mask).sum() / (        thresh_mask.sum() + 1e-6    )    return loss_p + loss_b + 10 * loss_t
```

这里采用原论文定义的三个损失项及权重。完整文件已经实现了 `balanced_bce()`，会优先选择损失大的背景像素参与训练。

### 3. Inference：如何把概率图转换成文本框？

推理阶段是这样一个流程：

概率图 P

固定阈值二值化

findContours 提取轮廓

计算置信度并过滤

Unclip 轮廓扩张

minAreaRect 获得四边形文本框

其中有两个特别重要的步骤：

步骤 A：轮廓提取

```
binary = (prob_map > 0.3).astype("uint8")contours, _ = cv2.findContours(    binary,    cv2.RETR_LIST,    cv2.CHAIN_APPROX_SIMPLE)
```

注意这里的 0.3 是推理时的固定阈值，不是网络训练时输出的阈值图 T。

原论文实验中使用 0.2；在 PaddleOCR 等工程实现里，0.3 是常见的配置值。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv

+1



步骤 B：Unclip（轮廓扩张）

由于训练时概率图学习的是收缩后的文本区域，推理出的轮廓也往往偏小。

因此需要将轮廓向外扩张：

\\[ D'=\frac{A'\times r'}{L'} \\]

其中 $A'$ 和 $L'$ 分别是预测轮廓的面积和周长，$r'$ 是扩张系数，论文取 1.5。

扩张后利用：

```
rect = cv2.minAreaRect(contour)box = cv2.boxPoints(rect)
```

即可获得旋转四边形文本框。

需要注意，真实代码中应该先对轮廓做 Unclip，再对扩张后的轮廓计算 `minAreaRect`。上面两行只是展示 OpenCV 的文本框提取操作。

对于弯曲文字，也可以直接输出多边形，而不是强制转换成四边形。

## 六、工程中最值得关注的参数

| 参数                  | 常见值       | 作用              |
| ------------------- | --------- | --------------- |
| `shrink_ratio`      | 0.4       | 生成文本收缩标签        |
| `k`                 | 50（论文）    | 控制近似二值化的陡峭程度    |
| `thresh_min/max`    | 0.3 / 0.7 | 阈值监督标签范围        |
| `det_db_thresh`     | 0.3       | 推理时的像素二值化阈值     |
| `det_db_box_thresh` | 0.6       | 文本框置信度过滤        |
| `unclip_ratio`      | 1.5～2.0   | 将收缩轮廓恢复成较完整文本区域 |

这些不是通用最优参数，要根据数据集调整。尤其注意：原论文的 $k=50$，但原作者公开 PyTorch 代码中的 `SegDetector` 构造函数默认 `k=10`，实际取值应以具体模型训练配置为准。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



工程上通常这样排查问题：

- 漏检小文字：先检查输入分辨率、概率图和后处理的最小尺寸过滤，不要只靠降低阈值。
- 相邻文字粘连：重点检查标注粒度、shrink_ratio、概率图分割质量。
- 文本框偏小或偏大：优先检查 unclip_ratio。
- 大量背景误检：检查 box_thresh、负样本和训练数据。
- 推理速度不理想：检查 Backbone、输入尺寸、后处理耗时，而不是仅关注神经网络的 FLOPs。

## 七、DBNet 的优缺点

| 优势              | 局限             |
| --------------- | -------------- |
| 适合多方向、长文本及不规则文本 | 相邻文字严重粘连时仍可能合并 |
| 不依赖 Anchor 设计   | 需要较好的多边形标注     |
| 训练时联合优化分割与二值化   | 标签生成和后处理存在几何计算 |
| 推理结构简单，方便轻量化部署  | 极小文字仍受特征分辨率限制  |
| 可以选择四边形或多边形输出   | 不具备文字识别与语义理解能力 |

DBNet++ 则在 DB 的基础上进一步加入 ASF（Adaptive Scale Fusion，自适应尺度融合），增强了不同尺度文本特征的融合能力。

[image](https://www.google.com/s2/favicons?domain=https://arxiv.org\&sz=32)

arXiv



## 八、面试中如何介绍 DBNet？

可以用下面这一段英文概括：

> DBNet is a segmentation-based text detection model. Its key innovation is differentiable binarization, which enables the network to learn an adaptive threshold map during training. The network predicts a probability map and a threshold map, and combines them through a differentiable sigmoid function. During inference, only the probability map is needed. Text boxes are generated using contour extraction and polygon expansion.

对应中文核心思想：

DBNet 的创新不是设计了多复杂的 Backbone，而是通过改进训练方式，让分割模型直接学习更清晰、更容易分离的文本区域，从而简化推理后处理。

对已有 CNN、OCR 检测经验的工程师来说，我认为真正值得掌握的是四件事：为什么使用收缩标签、为什么引入阈值图、可微分二值化如何产生梯度，以及为什么推理时可以删除阈值分支。

能够解释这四个问题，并且自己实现一遍标签生成、DB Loss 和 Unclip 后处理，基本就掌握了 DBNet 最重要的算法思想和工程实现。



