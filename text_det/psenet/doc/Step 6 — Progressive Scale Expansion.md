# LearnOCR · Step 6：Progressive Scale Expansion（PSE）

这一步实现 PSENet 最有代表性的算法：渐进式尺度扩张（Progressive Scale Expansion）。

前面 Step 2–5 已经实现了多尺度 Kernel 标签生成、网络、Loss 和训练。现在我们要解决最后一个关键问题：

如何将神经网络预测的 3 张二值 Mask 转换成一个个独立的文本检测框？

我已经完成了 Python 实现、测试和可视化，并验证它能够与 Step 5 的模型及 Checkpoint 格式衔接。

LearnOCR — PSENet Step 6

PSE 实现、单元测试、推理脚本、可视化示例、README

下载 Step 6 完整代码 ZIP

## 一、PSE 到底解决什么问题？

假设图片中有两行挨得很近的文字，网络预测的最大 Kernel（K3）可能已经把它们连接成一个整体。

如果直接对 K3 使用 `cv2.connectedComponents()`，只能得到一个文本实例。

但 PSENet 不一样：

[PSE教学测试结果：上排K1、K2、K3二值区域，下排三个阶段的实例标签逐步扩张，其中不同颜色的实例在连接区域仍保持分离。](https://images.openai.com/static-rsc-4/HeMBdIaWNPD1_pHpfz0bsgLGPMY3x4QOePrzsLzYgqjf7ocsv8dUglEScj7Bjc8qjNGWBB8csUc_6ABooLg0JMSI8H0vKDsBEjaTzg8W73bJZ0q1kjftKfNs_O6GlLULybiI3X6EZEhQTKnLLL5-GgokUSeOGu_1aUTN_0YQ-Xw?purpose=inline)

核心过程：从最小 Kernel 的独立连通区域开始，逐步扩张至更大的 Kernel；不同文本实例始终保留各自的 ID。

我们先对 K1 计算连通域：

- 左边文本得到 `ID=1`
- 右边文本得到 `ID=2`

然后向 K2 扩张，再向 K3 扩张。即使后面两个文本区域相连，已有的实例 ID 也不会合并。

这正是 PSENet 相比普通文本分割算法的重要特点。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



## 二、理解算法的核心过程

PSE 只需要两种基础算法：

| 算法     | 英文                        | 作用               |
| ------ | ------------------------- | ---------------- |
| 连通域分析  | Connected Components      | 找到 K1 中各个独立文本核心  |
| 广度优先搜索 | Breadth-First Search（BFS） | 将各个核心逐渐扩张至 K2、K3 |

具体来说：

1\. Connected Components

对 K1 进行连通域分析，每个连通域分配独立 ID

2\. Expand to K2

使用 BFS，让已有 ID 向 K2 的有效区域扩张

3\. Expand to K3

继续扩张，直到恢复完整文本区域

4\. Extract Polygons

根据最终实例区域提取轮廓和旋转矩形

### 最核心的 BFS 逻辑

下面是简化后的伪代码：

```
# K1 中每个连通区域分配不同 IDlabels = connected_components(K1)# 逐层扩张到 K2 和 K3for kernel in [K2, K3]:    queue = deque(all_labeled_pixels(labels))    while queue:        pixel = queue.popleft()        for neighbor in four_neighbors(pixel):            if kernel[neighbor] and labels[neighbor] == 0:                labels[neighbor] = labels[pixel]                queue.append(neighbor)
```

注意最重要的判断：

```
labels[neighbor] == 0
```

它保证一个像素只能被分配一次。如果某个像素已经属于文字实例 1，文字实例 2 就不能再占用它。

这是一种先到先得的竞争扩张机制。

另外，网络的三个预测通道不一定天然满足 K1 ⊆ K2 ⊆ K3。因此在后处理开始前，我们还要先对二值 Mask 取交集，确保它们满足嵌套关系。

## 三、Step 6 新增文件

本次不需要修改前面 Step 2–5 的代码。

```
LearnOCR/
└── text_detection/
    └── psenet/
        ├── kernel.py
        ├── model.py
        ├── loss.py
        ├── dataset.py
        ├── train.py
        │
        ├── pse.py           # 新增：PSE核心算法
        ├── test_pse.py      # 新增：单元测试
        ├── demo_pse.py      # 新增：扩张过程可视化
        └── predict.py       # 新增：真实图像推理
```

`pse.py` 完全基于 NumPy 和 OpenCV，不依赖 C++、Cython 或 PyTorch。这样做是为了让你能够亲自阅读和理解 BFS 扩张过程。

## 四、先运行纯算法测试

这一步不需要训练好的模型，也不需要 ICDAR 2015。

从 `LearnOCR` 根目录运行：

```
python text_detection/psenet/test_pse.py
```

我在本地运行得到：

```
Two touching text regions stay separate: PASS
Non-nested predictions are safely intersected: PASS
Tiny false-positive seeds do not grow: PASS
Box extraction and confidence filtering: PASS
Empty predictions: PASS

ALL STEP 6 TESTS PASSED
```

然后生成可视化：

```
python text_detection/psenet/demo_pse.py
```

输出文件：

`outputs/psenet_step6/synthetic_demo.png`

实际生成的合成数据测试图。上排为二值 Kernel，下排为逐级扩张的实例标签。两种颜色代表两个不同文字实例。

可以看到 K2、K3 已经连成一个区域，但两个文字实例仍然保留不同颜色。

这正是我们希望验证的核心行为。

## 五、接入 Step 5 训练好的模型

假设你已经完成 Step 5 完整训练，产生了：

`outputs/psenet_step5/train/best.pt`

执行：

```
python text_detection/psenet/predict.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --image data/icdar2015/test_images/img_1.jpg
```

程序会自动完成：

1. 根据 Step 5 的配置缩放、归一化图像。
2. 加载 MiniPSENet 并预测 3 张 Kernel 概率图。
3. 对概率图阈值化，执行 PSE。
4. 提取每个文本实例的轮廓和四边形，并映射回原图坐标。

结果保存为：

| 文件                     | 内容                 |
| ---------------------- | ------------------ |
| `prediction.jpg`       | 原始图片 + 检测框         |
| `prediction_debug.png` | K1/K2/K3、逐级扩张、最终实例 |
| `prediction.json`      | 每个实例的四边形坐标、置信分数    |

如果你目前只完成了单图过拟合，也可以先使用：

```
python text_detection/psenet/predict.py \
    --checkpoint outputs/psenet_step5/overfit/last.pt \
    --image data/icdar2015/train_images/img_1.jpg
```

这样更容易检查网络是否学会了那张图片。

我已经用一个随机初始化的 MiniPSENet Checkpoint 验证了推理、坐标转换及输出文件生成流程；但尚未用你实际训练的权重检验检测效果。

## 六、建议亲自做三个实验

实验 A：让 K1 的两个核心连接

预期结果：Connected Components 可能直接得到一个实例，PSE 无法再把它拆成两个。说明最小 Kernel 的分离质量非常重要。

实验 B：修改 Threshold

实验 C：比较 K3 连通域与 PSE 结果

## 七、Step 6 验收标准

学习进度

0/5

能够解释 PSE 与普通连通域检测的区别

理解 BFS 队列如何扩张不同的实例 ID

test_pse.py 全部测试通过

能够可视化 K1 → K2 → K3 的扩张

使用 Step 5 Checkpoint 对图片进行推理

## 下一阶段：Step 7 — Evaluation & Visualization

到目前为止，我们已经实现了 PSENet 的完整基本流程：

Image → ResNet18 → FPN → Kernel Prediction → PSE → Text Polygons

下一步我建议完成项目的最后一个阶段：

Step 7：在 ICDAR 2015 上评估文本检测效果。

我们会实现文本实例级 Precision、Recall、F1，分析漏检、误检、文字粘连和分割失败的原因，并把评估结果整理进 LearnOCR 的 PSENet README。

最后再对照论文，分析我们的简化版 PSENet 与原始 PSENet 有哪些差异，以及这些简化可能如何影响精度。

完成 Step 7 后，这个教学型 PSENet 就可以作为 `LearnOCR` 中第一个完整的经典算法实现封版。



