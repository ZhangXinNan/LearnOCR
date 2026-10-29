# LearnOCR · Step 7：Evaluation & Visualization

这一阶段，我们完成 MiniPSENet 的最后一部分：文本检测效果评估、错误分析与结果可视化。

到此，整个 PSENet 教学项目就具备了从数据准备、模型训练到检测评估的完整流程。

LearnOCR — PSENet Step 7

评估代码、测试、可视化、README · ZIP

下载 Step 7 完整代码

这次新增 5 个 Python 文件，不需要修改 Step 2–6 的代码。

## 一、这一步要解决什么问题？

Step 5 中，我们用 Pixel Dice 衡量网络预测区域与 GT 的重合程度。

但对于文本检测来说，Pixel Dice 高，不代表检测效果一定好。

例如，一张图片中有两行文字。如果 PSENet 把它们合并成一个检测框，即使像素覆盖得很好，也只能匹配一个文字实例，另一个可能计为漏检。

所以我们需要三个指标：

| 指标         | 英文    | 说明                      |
| ---------- | ----- | ----------------------- |
| Precision  | 精确率   | 检测出来的文字中，有多少是正确的？       |
| Recall     | 召回率   | 真实文字中，有多少被检测出来？         |
| F1 / Hmean | 调和平均值 | 综合评价 Precision 和 Recall |

计算方式：

\\[ Precision=\frac{TP}{TP+FP} \\]

\\[ Recall=\frac{TP}{TP+FN} \\]

\\[ F1=\frac{2PR}{P+R} \\]

其中：

- TP（True Positive）：正确检测。
- FP（False Positive）：误检。
- FN（False Negative）：漏检。

ICDAR 2015 文本定位采用基于四边形 Intersection over Union（IoU，交并比）的匹配，核心阈值是 0.5。

[image](https://www.google.com/s2/favicons?domain=https://rrc.cvc.uab.es\&sz=32)

Robust Reading Competition

+1



## 二、先看错误可视化

这是我使用合成数据实际运行 Step 7 代码生成的测试图片。

不同颜色表示：

| 颜色       | 含义           |
| -------- | ------------ |
| &#x20;绿色 | TP：检测正确      |
| &#x20;红色 | FP：误检        |
| &#x20;橙色 | FN：漏检        |
| &#x20;灰色 | Ignore：不参与评估 |
| 青色       | 已匹配的真实文本框    |

这张示例图中：

# 50%

Precision

# 50%

Recall

# 50%

F1

因为只有一个 TP、一个 FP 和一个 FN。

这是用人工构造的框验证评估器，不是训练好的 PSENet 在 ICDAR 2015 上的真实精度。

## 三、代码结构

将 ZIP 解压到原来的 `LearnOCR` 仓库。

```
LearnOCR/
└── text_detection/
    └── psenet/
        ├── kernel.py
        ├── model.py
        ├── loss.py
        ├── dataset.py
        ├── train.py
        ├── pse.py
        ├── predict.py
        │
        ├── eval_metrics.py       # Step 7
        ├── evaluate.py           # Step 7
        ├── test_evaluation.py    # Step 7
        ├── demo_evaluation.py    # Step 7
        └── plot_evaluation.py    # Step 7
```

其中最值得你亲自阅读的是 `eval_metrics.py`，它包含三个核心部分：

1. `polygon_iou()`：计算两个文本四边形的 IoU。
2. `evaluate_image()`：预测框与 GT 匹配，确定 TP、FP、FN。
3. `counts_to_metrics()`：计算 Precision、Recall 和 F1。

对 `###` 区域，评估器会根据预测框落入忽略区域的面积比例决定是否忽略，而不是只要两个框相交就不计分。这与常见 ICDAR 2015 RRC 评估实现的思路一致。

[image](https://www.google.com/s2/favicons?domain=https://huggingface.co\&sz=32)

tomofi/MaskTextSpotterV3-OCR at main



## 四、一步一步运行

### Step 7.1 — 单元测试

先确认评估规则正确：

```
python text_detection/psenet/test_evaluation.py
```

我已经执行了 8 项测试，全部通过，包括 IoU、重复检测框、忽略区域、阈值边界和文件格式。

然后运行可视化示例：

```
python text_detection/psenet/demo_evaluation.py
```

生成：

`outputs/psenet_step7/synthetic_demo.png`

### Step 7.2 — 先评估 5 张验证图片

前提是已经完成 Step 5 的完整训练，拥有：

`outputs/psenet_step5/train/best.pt`

运行：

```
python text_detection/psenet/evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split val \
    --max-images 5 \
    --visualize 5
```

程序会自动执行：

读取 Step 5 验证集图片及 GT

加载 best.pt 并预测三张 Kernel

Step 6 PSE 生成文本实例四边形

计算 IoU 并进行一对一匹配

统计 TP / FP / FN → Precision / Recall / F1

保存预测框、错误可视化及评估文件

### Step 7.3 — 完整验证集评估

确认前面的流程成功后，评估 Step 5 划分出来的全部验证图片，通常是 100 张。

```
python text_detection/psenet/evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split val \
    --visualize 12
```

完成后生成评估图表：

```
python text_detection/psenet/plot_evaluation.py \
    --dir outputs/psenet_step7/validation
```

输出如下：

```
outputs/psenet_step7/validation/
├── summary.json
├── per_image.csv
├── details.json
├── metrics.png
├── f1_distribution.png
├── rrc_results.zip
└── visualizations/
    ├── img_x.jpg
    └── ...
```

其中 `visualizations/` 会优先保存 F1 最低的图片，帮助你观察模型在哪些场景下失效。

## 五、推荐亲自完成的三个实验

实验 1：调整 Threshold

分别使用 --threshold 0.4、0.5、0.6，对比 Precision 和 Recall。观察阈值降低是否引入更多误检，以及阈值提高是否导致漏检增加。

实验 2：观察相邻文字粘连

实验 3：比较 Pixel Dice 与实例 F1

这些实验比单纯得到一个 F1 分数更有助于理解 PSENet。

## 六、最后进行官方测试集评估

当验证集效果确定后，可以使用 ICDAR 2015 官方 500 张测试图片：

```
python text_detection/psenet/evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split test \
    --out outputs/psenet_step7/test
```

这要求你已经下载对应的 `test_gt` 标注文件。

需要说明：本项目提供的是教学版评估实现，采用最高 IoU 优先的一对一匹配。它与官方 RRC 评估规则接近，但并非逐行复现。如果要报告与论文可直接比较的 Benchmark 数字，应该使用官方评估脚本。

本次我已经验证了单元测试、合成可视化，以及使用随机模型权重的完整评估文件生成流程。因为没有你训练好的 `best.pt`，尚未得到真实模型的 ICDAR 2015 Precision、Recall 和 F1。

## 七、完成 Step 7 后，LearnOCR 的 PSENet 可以封版

MiniPSENet 学习路线

Step 1 · Dataset Preparation

Step 2 · Kernel Label Generation

Step 3 · ResNet18 + FPN

Step 4 · Dice Loss + OHEM

Step 5 · Training Pipeline

Step 6 · Progressive Scale Expansion

Step 7 · Evaluation & Visualization

代码实现阶段完成；最终效果仍需使用你实际训练的权重进行验证。

我建议这之后先不要增加新功能，而是完成一次完整训练、评估和错误分析，然后整理 `text_detection/psenet/README.md`，记录：

- PSENet 的核心思想与关键公式。
- 网络结构、训练配置与最终指标。
- 典型的正确检测、漏检、误检示例。
- 为什么多尺度 Kernel 和 PSE 能够分离相邻文字。
- 简化实现与原论文的区别。

这样 PSENet 就真正从一篇读起来比较抽象的论文，变成了一个你能够理解、运行、解释和分析的算法。

随后我最推荐进入 LearnOCR 的第二个算法：DBNet。 它可以复用很多训练和评估经验，让我们把注意力集中在 PSENet 的 Progressive Expansion 与 DBNet 的 Differentiable Binarization 这两种核心设计的区别上。



