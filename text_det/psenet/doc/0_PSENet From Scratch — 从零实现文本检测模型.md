# PSENet From Scratch — 从零实现文本检测模型

我建议项目取名：

## MiniPSENet

mini-psenet · PyTorch · ICDAR 2015

项目定位： 一个用于理解 PSENet 算法、从零实现、训练和可视化的轻量级教学项目。

我的建议是：不直接复制官方 PSENet 项目，而是自己实现一个最小可训练版本。 尽量减少代码量，但不省略决定算法本质的部分。

最终你将亲自完成：

- 使用 PyTorch 实现 PSENet 网络。
- 理解并生成不同尺度的文本 Kernel 标签。
- 实现 Dice Loss 和训练流程。
- 实现 Progressive Scale Expansion（PSE）后处理。
- 使用真实公开数据集训练，最终检测任意方向的场景文字。

## 一、先确定技术方案

| 项目        | 选择               | 原因             |
| --------- | ---------------- | -------------- |
| 语言        | Python 3.10/3.11 | 简单、兼容性较好       |
| 深度学习框架    | PyTorch          | 直接实现训练         |
| Backbone  | ResNet18         | 比 ResNet50 更轻量 |
| 特征融合      | 简化版 FPN          | 保留多尺度特征融合      |
| Kernel 数量 | 3                | 更容易理解和可视化      |
| Loss      | Dice Loss        | 保留论文核心思想       |
| 后处理       | Python BFS       | 不使用 C++ 扩展     |
| 公开数据集     | ICDAR 2015       | 规模适中，标注容易理解    |
| 输入尺寸      | 640×640          | 先控制训练资源        |
| 目标        | 训练成功、理解算法        | 不追求论文 SOTA     |

这里的 ResNet18、3 个 Kernel 和简化 FPN 是为了学习而做的工程简化，不属于严格的论文复现。

### 为什么选择 ICDAR 2015？

[tedeval-a-fair-evaluation-metric-for-scene-text-detectors](https://images.openai.com/static-rsc-4/fXCvsFKkmSpEYs2BLR_xCDPf_8s3AkM4xphbbqaMjtbdCD-4UsqWYqnzSoUiEz1Kmz2-2bVDIH4QZKKz7hp1UHccQ_kb2yz3muKXB4k2KwsfwJGZZ8QXXVyZ5HHv_FuwdJaDPgDwub9_WHclNaesmq0MlknnnjcOSfzCiL0Mpbo?purpose=inline)

ICDAR 2015 Incidental Scene Text

训练集约 1,000 张图像，测试集约 500 张，包含各种自然场景中的倾斜、透视和不规则排列文字。训练标注采用四边形，方便我们自己生成多尺度 Kernel。

[image](https://www.google.com/s2/favicons?domain=https://rrc-red.cvc.uab.es\&sz=32)

Robust Reading Competition

+1



数据规模不大，非常适合学习。不过，训练成功不代表能够达到论文精度，尤其在不使用额外训练数据和长时间训练的情况下。

## 二、把整个项目拆成 7 步

01

项目环境与数据准备

创建项目、安装依赖、下载 ICDAR 2015、可视化标注

当前阶段

02

生成 Kernel 标签

Polygon Shrink、3 个尺度的文本 Mask

03

实现 PSENet 网络

ResNet18、FPN、输出 3 通道预测图

04

实现 Loss

Dice Loss、训练区域掩码、损失组合

05

实现模型训练

Dataset、DataLoader、Optimizer、Checkpoint

06

实现 PSE 后处理

连通域、BFS 扩张、文本实例恢复

07

测试与总结

检测可视化、Precision/Recall/F1、理解与 DBNet 的差异

每一步都遵循同样的学习流程：

What（是什么）→ Why（为什么）→ How（代码实现）→ Verify（如何验证）。

尽量保证每一步结束后都有可以运行、可以查看的结果，而不是一次写完所有代码才发现问题。

## 三、先用一句话理解 PSENet

PSENet（Progressive Scale Expansion Network，渐进式尺度扩张网络）的核心思想是：

先找到每个文本区域内部容易区分的小核心，再逐步向外扩张，恢复完整文本。

[Shape Robust Text Detection with Progressive Scale Expansion Network - 리뷰 - 1 — All I Need Is Data.](https://images.openai.com/static-rsc-4/HeMBdIaWNPD1_pHpfz0bsgLGPMY3x4QOePrzsLzYgqjf7ocsv8dUglEScj7Bjc8qjNGWBB8csUc_6ABooLg0JMSI8H0vKDsBEjaTzg8W73bJZ0q1kjftKfNs_O6GlLULybiI3X6EZEhQTKnLLL5-GgokUSeOGu_1aUTN_0YQ-Xw?purpose=inline)

例如图像中有两行文字，彼此非常接近。普通语义分割可能把它们连成一块。

PSENet 同时预测 3 张 Mask：

K1

小核心

K2

中等区域

K3

完整文字

示意图：从 K1 → K2 → K3 逐步扩张，蓝色和橙色代表不同文字实例。

网络实际上只预测三个尺度的前景概率，不同文字实例的身份是在后处理时，通过小 Kernel 的连通域和 BFS 扩张确定的。 这是 PSENet 与普通文本二分类分割最重要的区别。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

CVF Open Access

+1



整体架构可以简化为：

输入图片 [B, 3, 640, 640]

ResNet18 + FPN

分割头 [B, 3, 640, 640]

Sigmoid → 3 张概率图

阈值化 + 连通域 + PSE

文本实例轮廓 / 四边形

以后实现 PSENet 时，始终记住这个结构即可。

## 四、Step 1：项目初始化与数据准备

这一阶段先不写网络，也不训练。

目标：能正确读取 ICDAR 2015 图像与标注，并把文字框画出来。

### 1. 创建项目

建议在 Ubuntu 或 macOS 的终端执行：

```
mkdir mini-psenet
cd mini-psenet

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy \
    opencv-python pyclipper matplotlib tqdm
```

如果使用 NVIDIA GPU，建议按照 [PyTorch 官方安装指南](https://pytorch.org/get-started/locally/)选择对应 CUDA 版本。你也可以在 MacBook Pro M3 Max 上使用 PyTorch 的 MPS 后端。

检查安装：

```
python -c "import torch; print('PyTorch:', torch.__version__); print('CUDA:', torch.cuda.is_available()); print('MPS:', torch.backends.mps.is_available())"
```

预期至少一个可用计算设备是 CPU；CUDA 和 MPS 的可用情况取决于机器和安装环境。

### 2. 建立目录结构

我们先用少量文件组织项目，不引入复杂框架：

```
mini-psenet/
├── data/
│   └── icdar2015/
│       ├── train_images/
│       ├── train_gt/
│       ├── test_images/
│       └── test_gt/
│
├── models/              # PSENet 网络
├── datasets/            # 数据加载与 Kernel 生成
├── losses/              # Dice Loss
├── postprocess/         # PSE 后处理
├── scripts/             # 可视化与调试
│
├── checkpoints/
├── outputs/
├── train.py
├── predict.py
└── README.md
```

现在只创建目录：

```
mkdir -p data/icdar2015/{train_images,train_gt,test_images,test_gt}
mkdir -p models datasets losses postprocess
mkdir -p scripts checkpoints outputs
```

### 3. 下载 ICDAR 2015

官方地址：[ICDAR 2015 Robust Reading Competition — Downloads](https://rrc.cvc.uab.es/?ch=4\&com=downloads)

可能需要注册账号。选择 Challenge 4 / Task 1（Text Localization），下载下面四个文件。

[image](https://www.google.com/s2/favicons?domain=https://gitee.com\&sz=32)

Gitee

+1



| 文件名                                              | 内容   |
| ------------------------------------------------ | ---- |
| `ch4_training_images.zip`                        | 训练图像 |
| `ch4_training_localization_transcription_gt.zip` | 训练标注 |
| `ch4_test_images.zip`                            | 测试图像 |
| `Challenge4_Test_Task1_GT.zip`                   | 测试标注 |

下载到项目的 `downloads/` 目录后，在项目根目录执行：

```
unzip -j -q downloads/ch4_training_images.zip \
    -d data/icdar2015/train_images

unzip -j -q downloads/ch4_training_localization_transcription_gt.zip \
    -d data/icdar2015/train_gt

unzip -j -q downloads/ch4_test_images.zip \
    -d data/icdar2015/test_images

unzip -j -q downloads/Challenge4_Test_Task1_GT.zip \
    -d data/icdar2015/test_gt
```

其中 `-j` 用于去掉 ZIP 中的内部目录层级。

检查数据：

```
ls data/icdar2015/train_images | wc -l
ls data/icdar2015/train_gt | wc -l
```

正常情况下，两者应各有 1,000 个文件。

### 4. 理解 ICDAR 2015 标注

每张图像对应一个标注文件，例如：

`img_1.jpg` → `gt_img_1.txt`

标注格式如下：

```
x1,y1,x2,y2,x3,y3,x4,y4,transcription
```

前 8 个值表示文本四边形的四个角点，最后一个字段是文字内容。

`###` 表示不应参与正常检测训练评估的忽略文本区域。标注按原始像素坐标记录，因此读取图像时要注意 EXIF 旋转导致的坐标错位。

[image](https://www.google.com/s2/favicons?domain=https://rrc-red.cvc.uab.es\&sz=32)

Robust Reading Competition

+1



### 5. 编写第一段 Python 代码：可视化 Ground Truth

创建文件 `scripts/visualize_gt.py`：

执行：

```bash
python scripts/visualize_gt.py
```

成功后，打开 `outputs/gt_example.jpg`，应该能看到绿色的文字四边形和可能存在的红色忽略区域。

### 6. Step 1 验收标准

完成情况

0/5

PyTorch 成功导入，并确认运行设备

训练集图像数量为 1,000

训练标注数量为 1,000

成功生成 outputs/gt_example.jpg

文字框与实际文字位置正确对应

## 五、下一步做什么？

完成 Step 1 后，我们进入真正体现 PSENet 思想的 Step 2：Kernel Label Generation（多尺度文本核标签生成）。

这一步会从一块真实的四边形文本区域开始，使用 `pyclipper` 将 Polygon 缩小到 50%、75% 和 100% 的相对尺度，得到三张二值 Mask。随后验证嵌套关系，为模型训练准备监督信号。

我建议先只处理一张图，确保数学原理和代码输出都理解了，再制作完整 Dataset 和 DataLoader。

参考资料：

- [PSENet 原论文（CVPR 2019）](https://openaccess.thecvf.com/content_CVPR_2019/html/Wang_Shape_Robust_Text_Detection_With_Progressive_Scale_Expansion_Network_CVPR_2019_paper.html)
- [PSENet 官方 PyTorch 实现](https://github.com/whai362/PSENet)

你现在只需要完成 Step 1。 把环境检测输出和 `gt_example.jpg` 的结果告诉我，我们再进入 Step 2。后续尽量保持每一步代码短小、独立、能够立即验证。



