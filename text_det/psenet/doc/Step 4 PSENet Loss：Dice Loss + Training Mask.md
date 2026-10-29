# LearnOCR · Step 4：PSENet Loss

这一步是整个 PSENet 学习项目的重要节点：我们终于可以把前面实现的 Kernel 标签生成（Step 2） 和 神经网络（Step 3） 连接起来。

本阶段只做损失函数和一次反向传播验证，暂时不写完整训练循环。

## 一、先理解 PSENet 的损失函数

PSENet 的原论文采用两个损失的加权和：

\\[ \boxed{L=\lambda L_c+(1-\lambda)L_s} \\]

其中：

| 损失        | 英文                 | 监督目标    | 作用          |
| --------- | ------------------ | ------- | ----------- |
| \\(L_c\\) | Complete Text Loss | K3 完整文本 | 区分文本与背景     |
| \\(L_s\\) | Shrunk Kernel Loss | K1、K2   | 学习不同尺度的文本核心 |

原论文及相关实现通常采用 \\(\lambda=0.7\\)，即完整文本占 70%，小 Kernel 平均损失占 30%。

[image](https://www.google.com/s2/favicons?domain=https://www.researchgate.net\&sz=32)

ResearchGate

+1



对于我们的三个 Kernel：

\\[ L=0.7L\_{K3}+0.3\frac{L\_{K1}+L\_{K2}}{2} \\]

不过这里还要考虑两个掩码：文本损失使用 OHEM 选择训练像素，小 Kernel 损失只关注预测为完整文本的有效区域。

### 1. 为什么使用 Dice Loss，而不是普通 BCE？

场景文字检测存在严重的前景与背景不平衡。

例如，假设一张图片有 100,000 个像素，其中只有 5,000 个像素属于文本。如果模型把所有像素都预测为背景，像素级准确率仍然可以达到 95%。

因此，我们更关注预测文本区域与真实区域的重合程度。

Dice 的直观含义

部分重合

高度重合

完全重合

Ground Truth

Prediction

普通 Dice 系数可以写为：

\\[ D=\frac{2|P\cap G|}{|P|+|G|} \\]

但 PSENet 使用的是适合连续概率图的平方分母版本：

\\[ D(P,G)= \frac{2\sum\_{x,y}P\_{xy}G\_{xy}+\epsilon} {\sum\_{x,y}P\_{xy}^2+\sum\_{x,y}G\_{xy}^2+\epsilon} \\]

\\[ \boxed{L\_{\text{Dice}}=1-D(P,G)} \\]

这里 \\(P\\) 是 Sigmoid 后的预测概率，\\(G\\) 是 0/1 标签，\\(\epsilon\\) 用于防止除零。

当预测和标签完全一致时，Dice Loss 接近 0；当两者几乎没有交集时，Loss 趋向 1。

[image](https://www.google.com/s2/favicons?domain=https://docsaid.org\&sz=32)

DOCSAID

+1



### 2. Training Mask：如何处理 `###`？

这和 Step 2 直接相关。

假设一张图片有三处文字：

- `HELLO`：正常文字。
- `WORLD`：正常文字。
- `###`：模糊或不需要评估的文字区域。

我们通过 `training_mask` 指示哪些像素参与损失计算：

\\[ M\_{\text{valid}}(x,y)= \begin{cases} 1 & \text{有效训练区域}\\\ 0 & \text{忽略区域} \end{cases} \\]

计算 Dice 时将预测和标签同时乘以 Mask：

\\[ D(P\cdot M,G\cdot M) \\]

这样 `###` 区域既不贡献正样本损失，也不贡献负样本损失。

注意：`training_mask=0` 并不意味着背景。它意味着不监督、不计算损失。

### 3. 两个更容易忽略的细节

OHEM（Online Hard Example Mining，在线难样本挖掘）

完整文本损失不是对所有背景像素一视同仁，而是优先选择那些模型错误地认为是文字的背景像素。

例如某张图有 1,000 个有效正样本像素，按 3:1 的负正比例，最多挑选 3,000 个最难的负样本像素。

Kernel Loss 的 Mask

原论文对小 Kernel 的损失使用由预测完整文本概率阈值化得到的 Mask：

\\[ W=\mathbf{1}[\operatorname{Sigmoid}(S\_{K3})\geq0.5] \\]

实际实现还要与 `training_mask` 相交，排除忽略区域。

也就是说，小 Kernel 的监督主要发生在模型当前预测为文本的区域中。这种设计避免了大量非文本区域的重复监督，但在训练初期如果模型没有预测出任何文本，Kernel Loss 也可能没有有效像素。

[image](https://www.google.com/s2/favicons?domain=https://www.researchgate.net\&sz=32)

ResearchGate

+1



## 二、Step 4 完整代码

LearnOCR — PSENet Step 4

Python 源代码、测试、使用说明 · 7.4 KB

下载 Step 4 完整代码 ZIP

| 文件                | 作用                        |
| ----------------- | ------------------------- |
| `loss.py`         | Dice Loss、OHEM、PSENetLoss |
| `test_loss.py`    | 损失计算及梯度测试                 |
| `demo_loss.py`    | 使用 ICDAR 2015 图片验证前向和反向传播 |
| `README_STEP4.md` | 原理、运行方法及注意事项              |

直接解压到现有 `LearnOCR` 仓库即可，不会覆盖 Step 2 和 Step 3 的文件。

核心调用只有几行：

```
from loss import PSENetLosscriterion = PSENetLoss(    text_weight=0.7,    use_ohem=True,    negative_ratio=3,    kernel_mask_mode="predicted",)logits = model(images)result = criterion(logits, kernels, training_mask)loss = result["loss"]loss.backward()
```

其中三个输入分别是：

| 输入              | Tensor Shape   |
| --------------- | -------------- |
| `logits`        | `[B, 3, H, W]` |
| `kernels`       | `[B, 3, H, W]` |
| `training_mask` | `[B, H, W]`    |

## 三、运行单元测试

从 `LearnOCR` 项目根目录执行：

```
python text_detection/psenet/test_loss.py
```

我已在 PyTorch CPU 环境中完成测试，结果如下：

```
Dice + ignored pixels: PASS
OHEM top-3 negatives + ignore mask: PASS
All ignored gives loss=0 and gradient=0: PASS
Ignored logits don't change loss or gradient: PASS
Predicted-vs-GT kernel mask behavior: PASS
MiniPSENet + real PSENetLoss backward: PASS

loss=0.51205
text=0.47910
kernel=0.58893

ALL STEP 4 TESTS PASSED
```

这里的 Loss 来自随机初始化模型和合成标签，仅用于检查实现，不代表实际训练效果。

## 四、使用真实 ICDAR 2015 图片测试

你已经完成数据下载和标注可视化，可以直接执行：

```
python text_detection/psenet/demo_loss.py \
    --name img_1.jpg \
    --size 256
```

这段脚本会串起之前完成的模块：

ICDAR 2015 图片 + 四边形标注

Step 2：生成 K1 / K2 / K3 和 Training Mask

Step 3：MiniPSENet 前向传播

Step 4：Dice Loss + OHEM + Mask

loss.backward() → 计算网络梯度

这里的 `--size 256` 是为了降低测试计算量，脚本暂时使用正方形缩放。正式训练时我们会改成更加合理的图像与标注联合变换。

目前我验证了合成标签上的真实网络反向传播；由于当前环境没有你的 ICDAR 2015 图像和 `pyclipper` 依赖，真实数据演示需要你在本机运行。

## 五、Step 4 验收标准

学习完成情况

0/6

理解平方分母 Dice Loss 的公式

理解 ### 为什么需要 Training Mask

理解 OHEM 为什么选择难负样本

理解 Lc、Ls 及 0.7/0.3 的权重

成功运行 test_loss.py

成功使用 ICDAR 2015 图片进行反向传播

## 六、下一阶段：Step 5 — Training Pipeline

下一步我们就能在 ICDAR 2015 上真正训练模型了。

我们将实现 `Dataset`、`DataLoader`、图像增强、训练循环、AdamW 优化器和 Checkpoint 保存，同时增加一个很重要的学习实验：

先让模型只训练并过拟合一张图片。

如果模型能够逐步把这一张图片的三个 Kernel 学好，说明网络、标签和损失函数基本连接正确，再扩展到整个训练集。

这比直接训练 100 个 Epoch 更有利于定位问题和理解算法。



