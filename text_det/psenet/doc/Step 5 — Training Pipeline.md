# LearnOCR · Step 5：PSENet Training Pipeline

现在进入 PSENet 的真正训练阶段。

前四步我们已经完成了：

- Step 1：ICDAR 2015 数据读取与标注可视化
- Step 2：多尺度 Kernel 标签生成
- Step 3：ResNet18 + FPN + 3 通道分割头
- Step 4：Dice Loss + OHEM + Training Mask

Step 5 的目标：将这些模块连接起来，让 PSENet 真正学会预测文本区域。

我已经编写好完整的训练代码，兼容前面的实现，不需要修改 Step 2–4 的文件。

LearnOCR — PSENet Step 5

训练代码、测试、可视化、README · ZIP

下载 Step 5 完整代码

## 一、训练流程

ICDAR 2015 Dataset

图片 + Polygon 标注

Data Preprocessing

保持宽高比缩放 + Padding + 坐标变换

Kernel Label Generation

生成 K1 / K2 / K3 + Training Mask

MiniPSENet

ResNet18 + FPN + Head

PSENet Loss

Dice Loss + OHEM + Ignore Mask

Backward + Optimizer

loss.backward() + AdamW

Checkpoint + Visualization

保存权重、Loss 曲线和预测图

这里最值得掌握的是训练中的五行代码：

```
optimizer.zero_grad()logits = model(images)losses = criterion(logits, kernels, training_mask)losses["loss"].backward()optimizer.step()
```

它们分别完成：清空旧梯度、网络前向传播、计算损失、反向传播、更新参数。

这就是 PyTorch 的基本训练过程。后续学习 DBNet、CRNN 等模型时，这部分逻辑也基本相同。

## 二、新增文件

本次仅增加以下文件：

```
LearnOCR/
└── text_detection/
    └── psenet/
        ├── kernel.py           # Step 2
        ├── model.py            # Step 3
        ├── loss.py             # Step 4
        │
        ├── dataset.py          # Step 5 新增
        ├── train.py            # Step 5 新增
        ├── test_training.py    # Step 5 新增
        └── plot_history.py     # Step 5 新增
```

| 文件                 | 功能                      |
| ------------------ | ----------------------- |
| `dataset.py`       | 读取图像、同步变换标注、生成训练 Tensor |
| `train.py`         | 单图过拟合、完整训练、验证、权重保存      |
| `test_training.py` | 测试数据、掩码与一次参数更新          |
| `plot_history.py`  | 生成 Loss 和 Pixel Dice 曲线 |

## 三、第一阶段：先过拟合一张图片

不要一开始就训练全部 1,000 张图片。

先选择 `img_1.jpg`，只用这一张图重复训练。这样可以迅速发现网络、标签和损失之间的问题。

在 `LearnOCR` 根目录运行：

```
python text_detection/psenet/test_training.py
```

通过后执行：

```
python text_detection/psenet/train.py \
    --mode overfit \
    --name img_1.jpg \
    --size 320 \
    --overfit-steps 250 \
    --lr 0.0003 \
    --save-every 50
```

这次实验使用：

| 参数            | 值              | 目的        |
| ------------- | -------------- | --------- |
| 图片数量          | 1              | 排除数据复杂度   |
| 输入大小          | 320×320        | 减少计算量     |
| Batch Size    | 1              | 每步都看同一张图  |
| 优化器           | AdamW          | 更新网络参数    |
| Learning Rate | 3e-4           | 初始学习率     |
| 训练步数          | 250            | 先观察能否收敛   |
| Backbone      | 随机初始化 ResNet18 | 验证从零学习的能力 |

250 步只是初始实验设置，不保证一定达到理想结果。

### 预期观察到什么？

示意：理想的单图过拟合过程

00.250.50.751050100150200250

示意数据，不是实际训练结果

实际训练曲线可能波动，不一定单调下降。这里仅说明我们希望观察到的趋势。

我们希望看到：

1. 总 Loss 整体下降。
2. K3 的预测逐渐接近完整文本区域。
3. K1、K2 的预测逐渐接近较小的 Kernel 标签。
4. 文本区域的像素级 Dice 逐渐提高。

能够过拟合一张图，是训练流程正确的重要信号，但还不能说明模型具备泛化能力。

## 四、第二阶段：完整 ICDAR 2015 训练

单图过拟合成功后，再使用完整数据集。

训练集约有 1,000 张图片。我们从官方训练集中划分 900 张训练、100 张验证，暂时不使用官方测试集。

训练 / 验证集划分

Training · 900

Validation · 100

训练命令：

```
python text_detection/psenet/train.py \
    --mode train \
    --pretrained \
    --size 512 \
    --batch-size 2 \
    --epochs 20 \
    --lr 0.0001
```

这里推荐启用 `--pretrained`，使用 ImageNet 预训练的 ResNet18。因为 ICDAR 2015 数据相对较少，预训练 Backbone 通常更适合进行完整数据集实验。预训练权重可能需要首次联网下载。

[image](https://www.google.com/s2/favicons?domain=https://github.com\&sz=32)

GitHub

+1



### 根据设备调整

| 设备                      | 建议输入尺寸  | Batch Size |
| ----------------------- | ------- | ---------- |
| MacBook Pro M3 Max（MPS） | 512     | 2          |
| GTX 1070（CUDA）          | 512     | 1–2        |
| CPU 调试                  | 192–320 | 1          |

这些是建议起点，具体还要根据显存或内存使用情况调整。

## 五、训练结果保存在哪里？

训练时程序会自动保存：

```
outputs/psenet_step5/
├── overfit/
│   ├── history.csv
│   ├── preview_step_0000.jpg
│   ├── preview_step_0050.jpg
│   ├── preview_step_0100.jpg
│   ├── ...
│   ├── best.pt
│   └── last.pt
│
└── train/
    ├── split.json
    ├── history.csv
    ├── val_preview_epoch_001.jpg
    ├── ...
    ├── best.pt
    └── last.pt
```

其中 `best.pt` 和 `last.pt` 都包含网络参数及优化器状态。

训练后执行：

```
python text_detection/psenet/plot_history.py \
    --history outputs/psenet_step5/overfit/history.csv
```

得到 `loss_curve.png` 和 `pixel_dice_curve.png`。

注意目前的 Pixel Dice 只是 K3 完整文本区域的像素级重合指标，还不是文本实例检测的 Precision、Recall 或 F1。

## 六、已完成的代码验证

我进行了以下检查：

- Python 文件语法检查通过。
- 使用合成数据验证了 Letterbox 缩放和 Polygon 坐标同步。
- 验证了 Padding、忽略区域和 Kernel 标签。
- 验证了网络前向传播、Loss、反向传播与优化器更新。
- 使用合成数据执行了完整的单图训练和训练/验证循环。
- 验证了 Checkpoint、预测图与训练曲线文件可以生成。

当前执行环境没有安装 `pyclipper`，因此本地集成测试使用了替代的多边形偏移实现；还没有在你的真实 ICDAR 2015 数据上验证本次训练代码。你已经安装的 Step 2 环境应继续使用原来的 `pyclipper` 实现。

## 七、这一步最重要的学习内容

建议重点研究三件事：

第一，Dataset 到底做了什么？

它不是简单地读取图片。还需要保证图像、Polygon、Kernel 和 Training Mask 之间的空间位置完全一致。

第二，模型到底如何学会预测？

观察每次 `optimizer.step()` 之后的 Loss 和预测概率图，理解梯度下降如何逐渐改变输出。

第三，为什么要分开训练集和验证集？

如果训练 Loss 下降、验证 Loss 却没有改善，模型可能只记住了训练图像，没有学会对新图片进行泛化。

## 下一步：Step 6 — Progressive Scale Expansion

完成训练后，我们就进入 PSENet 最有特色的部分：

PSE 后处理：Connected Components + BFS Progressive Expansion。

也就是从 K1 的小核心开始，逐渐扩张到 K2、K3，最终得到不同文字实例的轮廓。

这是 PSENet 相比普通分割网络最值得理解的算法部分，也是我建议你亲自逐行实现的部分。