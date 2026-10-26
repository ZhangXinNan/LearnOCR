# PSENet(Progressive Scale Expansion Network)文本检测总结

> 论文:*Shape Robust Text Detection with Progressive Scale Expansion Network*(Wang et al., CVPR 2019)

## 一、直观理解

**问题:** 基于分割的方法有个老大难:**相邻的文本实例容易粘在一起**。两行文字靠得很近时,分割图上会连成一片,后处理无法分开。回归框的方法(如 EAST)又很难处理弯曲文本。

**类比:** 想象在一块地里种了几簇相距很近的草,要确定每簇的范围。

1. 先在每簇草的**正中心**插一面彩旗(最小的核,彼此离得很远,绝不会混在一起)。
2. 然后每面旗向外一圈圈"圈地",每一轮范围稍大一点。
3. 两面旗圈到同一块地时,**谁先到就归谁**,不再争抢。
4. 最终圈到完整文字范围为止。

**核心一句话:** 为每个文本实例预测一组从小到大的"核"(kernel),用最小核把实例分开,再通过渐进式扩张还原成完整形状。

## 二、算法思想

### 1. 整体流程

```
输入图像 → ResNet + FPN → 融合特征
        → 输出 n 张分割图 S_1, S_2, ..., S_n
          (S_1 最小的核,S_n 是完整文本区域)
                 ↓
      渐进尺度扩张算法(PSE):从 S_1 出发,逐级扩张到 S_n
```

论文默认 n = 6,最小核的比例 m = 0.5。

### 2. 标签生成

沿用 Vatti clipping 收缩多边形,第 i 个核的收缩比例为:

$$r_i = 1 - \frac{1-m}{n-1}(n-i)$$

收缩距离 `d_i = A(1 - r_i²) / L`(A 为面积,L 为周长)。i = n 时 r = 1,即原始标注;i = 1 时 r = m,即最小核。

### 3. 损失函数

$$L = \lambda L_c + (1-\lambda) L_s,\quad \lambda = 0.7$$

| 项 | 作用对象 | 说明 |
|---|---|---|
| $L_c$ | 完整文本区域 $S_n$ | Dice Loss + OHEM(正负样本 1:3) |
| $L_s$ | 收缩的核 $S_1 \dots S_{n-1}$ | Dice Loss,只计算在 $S_n$ 预测为文字的像素内,排除大量背景干扰 |

用 Dice Loss 的原因是文字像素占比很小,类别极不平衡。

### 4. 渐进尺度扩张(PSE)算法

输入是 n 张二值核图,步骤如下:

1. 对 $S_1$ 做连通域分析,**每个连通域就是一个文本实例**的初始种子。
2. 对 $S_2$:从当前各实例的像素出发做 BFS,把邻接且在 $S_2$ 中为 1 的像素并入该实例。
3. 若一个像素同时被多个实例争抢,**先到先得**,后来者不能抢。
4. 对 $S_3, \dots, S_n$ 重复第 2、3 步,最后得到完整实例。

因为冲突只发生在边界的少量像素上,"先到先得"带来的误差很小,却保证了相邻文本不会粘连。

## 三、代码实现(PyTorch 核心部分)

### 1. Neck + Head

```python
import torch, torch.nn as nn, torch.nn.functional as F

class PSEHead(nn.Module):
    """输入 FPN 的 P2~P5(各 256 通道),输出 n 张分割图"""
    def __init__(self, in_ch=256, n=6):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(in_ch * 4, 256, 3, padding=1, bias=False),
            nn.BatchNorm2d(256), nn.ReLU(True))
        self.out = nn.Conv2d(256, n, 1)

    def forward(self, p2, p3, p4, p5, out_size):
        size = p2.shape[2:]
        feats = [p2] + [F.interpolate(p, size=size, mode='bilinear',
                                      align_corners=False) for p in (p3, p4, p5)]
        x = self.fuse(torch.cat(feats, dim=1))
        x = self.out(x)
        # 上采样到目标尺寸,逐通道 sigmoid
        x = F.interpolate(x, size=out_size, mode='bilinear', align_corners=False)
        return torch.sigmoid(x)       # (B, n, H, W),通道 0 是最小核,n-1 是完整区域
```

### 2. 损失函数

```python
class PSELoss(nn.Module):
    def __init__(self, lam=0.7, ohem_ratio=3, eps=1e-6):
        super().__init__()
        self.lam, self.ratio, self.eps = lam, ohem_ratio, eps

    def dice(self, pred, gt, mask):
        pred, gt = pred * mask, gt * mask
        inter = (pred * gt).sum(dim=(1, 2))
        union = (pred ** 2).sum(dim=(1, 2)) + (gt ** 2).sum(dim=(1, 2)) + self.eps
        return 1 - 2 * inter / union                      # (B,)

    def ohem_mask(self, pred, gt, train_mask):
        """每张图保留全部正样本 + 得分最高的 3 倍数量负样本"""
        masks = []
        for p, g, m in zip(pred, gt, train_mask):
            pos = (g > 0.5) & (m > 0.5)
            neg = (g <= 0.5) & (m > 0.5)
            n_pos = int(pos.sum())
            if n_pos == 0:
                masks.append(m.clone()); continue
            n_neg = min(int(neg.sum()), n_pos * self.ratio)
            neg_scores = p[neg]
            if n_neg > 0:
                thr = neg_scores.topk(n_neg)[0][-1]
                sel = pos | (neg & (p >= thr))
            else:
                sel = pos
            masks.append(sel.float())
        return torch.stack(masks)

    def forward(self, pred, gt_kernels, train_mask):
        # pred / gt_kernels: (B, n, H, W),索引 n-1 为完整文本区域
        n = pred.shape[1]
        s_n, g_n = pred[:, -1], gt_kernels[:, -1]

        # L_c:完整文本区域,带 OHEM
        M = self.ohem_mask(s_n.detach(), g_n, train_mask)
        l_c = self.dice(s_n, g_n, M).mean()

        # L_s:各收缩核,只在 S_n 预测为文字(>0.5)的像素内计算
        W = ((s_n > 0.5).float() * train_mask).detach()
        l_s = sum(self.dice(pred[:, i], gt_kernels[:, i], W)
                  for i in range(n - 1)).mean() / (n - 1)

        return self.lam * l_c + (1 - self.lam) * l_s
```

### 3. 标签生成

```python
import numpy as np, cv2, pyclipper
from shapely.geometry import Polygon

def make_kernel_labels(poly, h, w, n=6, m=0.5):
    """返回 (n, h, w);索引 0 为最小核,n-1 为完整文本区域"""
    labels = np.zeros((n, h, w), dtype=np.uint8)
    pg = Polygon(poly)
    for i in range(1, n + 1):
        r = 1 - (1 - m) / (n - 1) * (n - i)
        if i == n:
            shrunk = np.array(poly)
        else:
            d = pg.area * (1 - r ** 2) / pg.length
            pco = pyclipper.PyclipperOffset()
            pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
            res = pco.Execute(-d)
            if not res:           # 收缩后消失(过小文本),跳过
                continue
            shrunk = np.array(res[0])
        cv2.fillPoly(labels[i - 1], [shrunk.astype(np.int32)], 1)
    return labels
```

多个实例的标签需要叠加(取各实例结果的最大值),被标为"难以辨认"的区域记入 `train_mask` 屏蔽。

### 4. PSE 后处理

```python
from collections import deque

def pse(kernels, min_area=10):
    """kernels: (n, H, W) 二值数组,索引 0 为最小核。返回实例标签图。"""
    n, H, W = kernels.shape
    _, label = cv2.connectedComponents(kernels[0].astype(np.uint8), connectivity=4)
    label = label.astype(np.int32)

    for s in range(1, n):                       # 逐级扩张
        q = deque(zip(*np.nonzero(label)))      # 当前所有已标记像素入队
        while q:
            y, x = q.popleft()
            l = label[y, x]
            for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                ny, nx = y + dy, x + dx
                if 0 <= ny < H and 0 <= nx < W \
                   and kernels[s, ny, nx] and label[ny, nx] == 0:
                    label[ny, nx] = l           # 先到先得
                    q.append((ny, nx))
    return label

def postprocess(pred, thr=0.5, box_thresh=0.8, min_area=10):
    """pred: (n, H, W) 的概率图"""
    text = pred[-1] > thr
    kernels = ((pred > thr) & text[None]).astype(np.uint8)   # 核必须落在文字区域内
    label = pse(kernels)

    boxes = []
    for idx in range(1, label.max() + 1):
        mask = label == idx
        if mask.sum() < min_area:
            continue
        if pred[-1][mask].mean() < box_thresh:      # 实例平均得分过滤
            continue
        pts = np.argwhere(mask)[:, ::-1].astype(np.float32)   # (y,x) -> (x,y)
        boxes.append(cv2.boxPoints(cv2.minAreaRect(pts)))
    return boxes
```

实际工程中,PSE 的 BFS 循环用 Python 会很慢,官方实现是用 C++ / Cython 加速的(`pse` 扩展)。上面的写法主要用来展示算法逻辑。

## 四、要点回顾

| 维度 | 要点 |
|---|---|
| 核心创新 | 多尺度核 + 渐进式扩张,用最小核分开相邻实例 |
| 擅长 | 任意形状(弯曲)文本、密集相邻文本 |
| 不足 | PSE 后处理串行、耗时,速度偏慢(远低于 DBNet) |
| 常见坑 | 最小核比例 m 过大则近距离文本仍会粘连,过小则小文本的核会消失;核的数量 n 增加会提高精度但拖慢后处理 |

**和 DBNet 的对比:** 两者都是分割类方法,都用 Vatti clipping 收缩标签。PSENet 靠**多个核 + 扩张算法**解决粘连,DBNet 靠**单个收缩图 + 自适应阈值 + unclip**解决。因此 DBNet 后处理更简单、速度更快,PSENet 的思路则更直观。

如果你想要一个能直接训练的完整工程结构(数据集、训练循环、评估),我可以接着展开。



