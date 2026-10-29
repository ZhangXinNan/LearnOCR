# Step 2 — Kernel Label Generation（多尺度文本核标签生成）

&#x20;Step 1 已完成

现在进入 PSENet 最重要的算法思想之一：如何从一份普通文本框标注，生成多个不同尺度的文本区域，用于监督网络训练。

这一阶段我们先不使用 PyTorch 训练模型，而是专注于理解标签生成的数学原理、实现方法和可视化。

## 1. 这一阶段要完成什么？

输入：ICDAR 2015 中的一张图片及其四边形文本标注。

输出：

K1 — 0.50

最小核心

K2 — 0.75

中等核心

K3 — 1.00

完整文本区域

示意图：实际标签由多边形向内收缩得到，并不是简单的矩形缩放。

三个 Kernel 应满足：

\\[ K_1\subseteq K_2\subseteq K_3 \\]

为什么要这样做？

假设两行文本距离非常近，直接预测完整区域可能粘连。PSENet 先利用缩小后的核心区分不同文本实例，再由小到大扩张，最终恢复完整文字区域。

这正是论文 Shape Robust Text Detection With Progressive Scale Expansion Network 的核心设计。

[image](https://www.google.com/s2/favicons?domain=https://openaccess.thecvf.com\&sz=32)

Open Access CVF

+1



## 2. 理解核心数学公式

给定一个原始文本多边形 \\(P\\)，我们需要计算向内收缩的距离 \\(d\\)：

\\[ \boxed{d=\frac{A(1-r^2)}{L}} \\]

其中：

| 符号      | 含义                  | 对应代码                |
| ------- | ------------------- | ------------------- |
| \\(A\\) | 多边形面积（Area）         | `cv2.contourArea()` |
| \\(L\\) | 多边形周长（Perimeter）    | `cv2.arcLength()`   |
| \\(r\\) | 收缩尺度比例（Scale Ratio） | `0.5`、`0.75`、`1.0`  |
| \\(d\\) | 多边形向内偏移距离           | `distance`          |

注意：r 不是宽高缩放系数，而是用于计算内缩距离的面积比例参数。 对一般多边形，内缩后的实际面积比例不一定精确等于 \\(r^2\\)。

例如，一个 200×40 像素的矩形，面积是 8000，周长是 480。取 \\(r=0.5\\)：

\\[ d=\frac{8000(1-0.5^2)}{480}=12.5 \\]

也就是说，把矩形四条边分别向内部移动约 12.5 像素。

三种尺度分别为：

| Kernel | r    | d（该矩形）   |
| ------ | ---- | -------- |
| K1     | 0.50 | 12.50 px |
| K2     | 0.75 | 7.29 px  |
| K3     | 1.00 | 0 px     |

对于非规则四边形，不能简单地把顶点向中心移动，因为这样无法保证均匀收缩。我们使用 `pyclipper` 来实现多边形内偏移（Polygon Offset），与 PSENet 论文的几何处理思想一致。

[image](https://www.google.com/s2/favicons?domain=https://docsaid.org\&sz=32)

DOCSAID

+1



## 3. 实现代码

我已经准备好两份独立的 Python 文件，你可以直接放进 `LearnOCR`，也可以逐行阅读、自己重新实现。

LearnOCR — PSENet Step 2

2 个 Python 文件 + 1 份 README

下载 Step 2 完整代码 ZIP

kernel.py

读取标注、收缩 Polygon、生成 Kernel 和 Training Mask

visualize_kernels.py

将输入图片、三个 Kernel 和训练掩码可视化

放到项目后，结构如下：

```
LearnOCR/
├── data/
│   └── icdar2015/
│       ├── train_images/
│       └── train_gt/
│
├── text_detection/
│   └── psenet/
│       ├── kernel.py
│       └── visualize_kernels.py
│
└── outputs/
```

### 重点理解 `kernel.py`

我把关键代码提取出来讲解。

第一部分：计算内缩距离

```python
area = abs(cv2.contourArea(polygon))
perimeter = cv2.arcLength(polygon, True)
distance = area * (1.0 - ratio**2) / perimeter
```

这里完全对应上面的数学公式。

第二部分：缩小多边形

```python
offset = pyclipper.PyclipperOffset()
offset.AddPath(    polygon.tolist(),    pyclipper.JT_ROUND,    pyclipper.ET_CLOSEDPOLYGON)
paths = offset.Execute(-distance)
```

注意 `-distance`，负号表示向内部偏移。`Execute()` 返回的可能是多个 Polygon，因为不规则文本区域收缩后可能分裂成几块。

第三部分：生成多个二值标签

```python
ratios = (0.5, 0.75, 1.0)
for i, ratio in enumerate(ratios):
    paths = shrink_polygon(polygon, ratio)
    if paths:
        cv2.fillPoly(kernels[i], paths, 1)
```

其中，每张 Kernel Mask 都是二维的：

- `1`：文本区域（Foreground）
- `0`：背景（Background）

完整实现还会对 `###` 忽略区域生成 `training_mask`，以及跳过无法形成有效核心的极小文本实例。这里不直接删除数据，而是在计算损失时通过 `training_mask` 排除相应像素。

## 4. 运行与验证

在 `LearnOCR` 根目录执行。

首先进行不依赖真实数据集的几何测试：

```
python text_detection/psenet/kernel.py
```

正常输出类似：

```
Synthetic test PASS
kernel pixel counts: [3009, 5049, 8241]
```

这代表三个 Kernel 的前景像素数量依次增加。

然后用真实 ICDAR 2015 图像生成标签：

```
python text_detection/psenet/visualize_kernels.py
```

输出文件：

`outputs/psenet_step2.png`

下面是我使用合成矩形测试得到的五栏可视化示例。实际 ICDAR 2015 文字框会更加复杂。

PSENet Step 2 kernel mask visualization

合成测试示例。白色代表文本前景；最后一栏中黑色代表不参与 Loss 的忽略区域。示例使用几何内缩替代实现验证了流程；本机的 pyclipper 执行仍需你验证。

你应该重点观察：

1. K1 明显小于 K2，K2 小于 K3。
2. K3 基本恢复原始标注的完整文本区域。
3. 靠近的文字在 K1 中通常更容易保持分离。
4. `###` 区域不应该参与损失计算。

另外可以换一张图片：

```
python text_detection/psenet/visualize_kernels.py \
    --name img_2.jpg
```

## 5. 理解一个容易混淆的问题

假设 ICDAR 2015 一张图片中有 20 个有效文字框。

是不是需要生成 20×3=60 张 Mask？

不需要。

我们会对每个文字实例分别生成 3 个 Kernel，然后把同一尺度的所有文字区域合并到一张二值 Mask。

最终整张图片仍然只生成：

假设原始图片大小为 H × W

Kernel Ground Truth

[3, H, W]

Training Mask

[H, W]

到 Step 3 时，PSENet 网络会预测相同空间尺寸的三通道输出，分别对应 K1、K2、K3。网络并不知道每个像素属于哪个独立的文字实例；实例的区分需要后面的 PSE 算法完成。

## 6. Step 2 验收

验收清单

0/5

kernel.py 独立几何测试通过

成功生成真实 ICDAR 2015 图片的 3 张 Kernel Mask

确认 K1 ⊆ K2 ⊆ K3

理解 ### 对应的 Training Mask

能够解释为什么 PSENet 不直接只预测完整文本区域

这一阶段不要急着写神经网络。 我希望你首先完全理解：为什么是多尺度 Kernel、怎么收缩 Polygon、为什么需要 Ignore Mask，以及这三张标签之后如何监督网络。

完成后，下一阶段我们实现：

Step 3 — Minimal PSENet Architecture：ResNet18 + 简化 FPN + 3 通道分割头。

届时我们将用 PyTorch 搭建网络，输入 `[B,3,640,640]`，输出 `[B,3,640,640]` 的预测结果，并逐层打印特征图尺寸，弄清楚数据是如何流经整个模型的。



