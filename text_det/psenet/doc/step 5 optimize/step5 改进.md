# MiniPSENet：如何在保持代码简单的前提下提高 P/R/F1？

可以提高，而且我建议暂时不要换 ResNet50。

我检查了之前给你生成的 Step 2–7 源代码，也对照了 PSENet 官方实现。目前没有发现明显的网络结构或损失函数接口错误，但发现几处值得优先优化的地方。

你的目标是通过 PyTorch 理解经典 OCR 算法，所以我建议保留 ResNet18 + FPN + 3 Kernel + PSE 的结构，先优化训练和推理配置。

## 一、当前结果说明什么？

ICDAR 2015 · Test · 500 images

# 44.5%

Precision

# 30.6%

Recall

# 36.3%

F1

635

TP

791

FP

1,442

FN

比较关键的数字是：

- 测试集共有 2,077 个需要检测的真实文字实例。
- 模型只输出了 1,426 个参与评估的预测框。
- 其中正确匹配的只有 635 个。

这说明目前既存在明显的漏检，也存在误检或定位不准确的问题。Recall 低是最值得优先调查的现象，但单纯增加预测框的数量不一定提高 F1。

作为参考，官方 PSENet 的 ResNet50 在 ICDAR 2015 上报告过约 78.5%–80.2% 的 F1，但它使用了更大的测试输入、更完整的训练策略和不同的模型配置。不能直接把这个差距归结为 ResNet18 和 ResNet50。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 二、我发现最值得优先解决的是输入分辨率

你现在使用的是：

```
"input_size": 512
```

而我们的 `dataset.py` 使用 Letterbox：保持图片宽高比，将整张图片缩放到 512×512 的正方形画布中。

ICDAR 2015 的图片通常是 1280×720。经过 Letterbox 后，实际有图像内容的区域只有：

原图 1280×720

原始图像

512×512 Letterbox

512×288

示意图：上下黑色区域是 Padding，不包含真实文字信息。

对于一段原来高度 12 像素的小文字：

- 缩放后高度仅约 4.8 像素。
- ResNet18 的 C2 特征层为 1/4 分辨率。
- 对应的文字在 C2 上可能只有 1–2 个像素高。
- K1 再进行收缩，可能只剩极少像素，甚至完全消失。

这很可能是 Recall 低的重要原因之一。

因此，在不换 Backbone 的情况下，增大有效图像分辨率，是我最优先建议尝试的改动。

官方实现也非常重视输入分辨率，曾分别采用短边 736、1024 的测试设置；但这和我们使用的正方形 Letterbox 尺寸不是同一种定义。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



## 三、建议按这个优先级优化

| 优先级 | 优化项                   | 修改代码量     | 主要目的           |
| --- | --------------------- | --------- | -------------- |
| 1   | 调整推理阈值、最小面积           | 0 行       | 减少漏检，改善 P/R 平衡 |
| 2   | 增大输入分辨率               | 0 行       | 提高小文字检测能力      |
| 3   | 从 20 Epoch 增加到 60–100 | 0 行       | 让网络充分学习        |
| 4   | 增加简单几何数据增强            | 约 20–30 行 | 提高泛化能力         |
| 5   | ResNet18 → ResNet50   | 小幅修改      | 增强特征提取能力       |

我的建议是先完成前三项，观察 F1 能否明显提升。

## 四、第一轮：不重新训练，直接优化推理

目前 `evaluate.py` 已支持这些参数：

```
--size
--threshold
--min-score
--min-area
--min-kernel-area
```

另外，我在 Step 6 的 `pse.py` 中确认了一个细节：

当 `threshold=0.5` 且 `min_score=0.5` 时，`min_score` 基本不起额外的过滤作用。

因为实例已经由 K3 中概率不低于 0.5 的像素扩张得到，其平均 K3 概率自然不低于 0.5。

所以我们应优先调整 `threshold`、`min_kernel_area` 和输入尺寸。

建议在 Validation 集 上运行以下实验，不要继续用 Test 集调参数。

推理实验配置

A · Baseline

B · 降低阈值

C · 保留小核心

D · 提高分辨率

当前实验命令

&#x20;复制

python text_detection/psenet/evaluate.py \   --checkpoint outputs/psenet_step5/train/best.pt \   --split val \   --size 512 \   --threshold 0.5 \   --min-kernel-area 5 \   --min-area 30 \   --out outputs/psenet_optimization/A

A 是基线；B 验证降低预测阈值是否能增加 TP；C 验证是否有很多小文本核心被面积过滤掉；D 验证大尺寸推理是否改善检测。

D 可以直接使用原先 512 输入训练得到的权重，因为模型是全卷积网络。但推理分辨率不同于训练分辨率，不保证一定改善结果，且 Python BFS 会更慢。

建议先记录四组验证集结果，选择 F1 最好的配置，再考虑重新训练。

## 五、第二轮：延长训练时间

原来我推荐的训练命令是：

```
python text_detection/psenet/train.py \
    --mode train \
    --pretrained \
    --size 512 \
    --batch-size 2 \
    --epochs 20 \
    --lr 0.0001
```

现在回头看，20 Epoch 更适合验证代码能否运行，并不适合判断 PSENet 的最终性能。

作为参考，MMOCR 公布的 ICDAR 2015 PSENet 训练配置使用了约 600 Epoch。当然，它采用的是更完整的网络与训练策略，不意味着我们的教学项目必须训练这么久。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub



我建议先将训练延长到 60 Epoch，观察 Validation Loss 和 Pixel Dice 的变化。

如果继续现有训练，可以使用：

```
python text_detection/psenet/train.py \
    --mode train \
    --size 512 \
    --batch-size 2 \
    --epochs 60 \
    --lr 0.0001 \
    --resume outputs/psenet_step5/train/last.pt
```

这里的 60 表示训练到第 60 个 Epoch，而不是再增加 60 个。

我还检查出之前的 `train.py` 有一个需要留意的小问题：恢复训练时加载 `optimizer_state` 会覆盖命令行指定的新学习率。所以这条命令可以沿用原学习率继续训练，但如果以后想降低学习率，需要额外处理这一点。

先看原来的 `history.csv`：如果 Validation Loss 还在持续下降，增加训练轮数通常值得尝试；如果 Validation Loss 已经明显反弹，那么单纯延长训练可能只会加剧过拟合。

## 六、第三轮：尝试 768×768 训练

如果前面优化仍然不能得到满意的结果，我更建议先提高训练分辨率，而不是换 ResNet50。

新建一组独立实验，保留原来的 512 模型以便比较：

```
python text_detection/psenet/train.py \
    --mode train \
    --pretrained \
    --size 768 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_768
```

这会从 ImageNet 预训练的 ResNet18 开始一个新实验，而不是继续现有模型。

在典型 1280×720 的 ICDAR 图片上：

| 输入画布      | 实际图像内容   | 相对像素计算量 |
| --------- | -------- | ------- |
| 512×512   | 512×288  | 1×      |
| 640×640   | 640×360  | 1.56×   |
| 768×768   | 768×432  | 2.25×   |
| 1024×1024 | 1024×576 | 4×      |

768 是我建议优先尝试的折中选择。更高的分辨率意味着更多小文字细节被保留，但训练成本、显存占用和后处理耗时也会增加。

## 七、ResNet50 值得换吗？

值得作为后续对照实验，但目前不是首选。

ResNet50 与 ResNet18 的主要区别：

| 项目        | ResNet18   | ResNet50           |
| --------- | ---------- | ------------------ |
| 残差结构      | BasicBlock | Bottleneck         |
| 参数量       | 约 11.7M    | 约 25.6M            |
| 最深特征通道    | 512        | 2048               |
| 训练成本      | 较低         | 更高                 |
| 小数据集训练    | 较轻量        | 更依赖良好预训练和正则化       |
| 对你当前项目的价值 | 学习原理、快速实验  | 研究 Backbone 对精度的影响 |

从算法教学的角度看，ResNet18 和 ResNet50 不会改变 PSENet 的核心原理。两者仍然使用相同的 Kernel、Dice Loss 和 PSE。

切换到 ResNet50 需要修改 `model.py` 中的 Backbone 和 FPN 输入通道数，重新训练模型；不能简单地将 `resnet18()` 替换成 `resnet50()` 后继续加载现有 Checkpoint。

相比之下，调整输入分辨率、训练轮数和阈值，不会增加 PSENet 算法本身的复杂度。

## 八、我的推荐路线

保持 ResNet18，逐步改进

V1 · 已完成

F1 = 36.3%

512 输入 / 原始训练配置 / Test 集结果

V2 · 推理参数优化

保持当前权重，调阈值、Kernel 面积、输入尺寸，用验证集选择参数

V3 · 延长训练

ResNet18 + 512，训练至 60 Epoch，观察训练和验证曲线

V4 · 分辨率优化

ResNet18 + 768，独立训练并评估是否改善小文本召回率

V5 · 可选：ResNet50 对照

其他条件尽量保持一致，专门研究 Backbone 对检测效果的影响

还有一点非常重要：后续全部使用同一份 Validation 集进行参数选择，最终确定配置后再在 Test 集上评估一次。现在这个 36.3% 可以作为已有的测试参考，但不应该反复依据测试集结果调参。

我建议你先把 Step 5 的 `history.csv`，特别是最后几轮的 Train Loss、Val Loss、Val Pixel Dice，以及 2–3 张 Step 7 检测可视化图片发给我。 我们可以判断主要问题究竟是训练不足、小文字丢失、PSE 合并错误，还是检测框定位不准，再决定 V2 和 V3 应该如何修改。

这样既能提高精度，也能真正理解每个优化为什么有效，而不是不断换更复杂的网络。



