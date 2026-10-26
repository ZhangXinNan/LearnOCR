# DBNet(Differentiable Binarization)文本检测总结

> 论文:*Real-time Scene Text Detection with Differentiable Binarization*(Liao et al., AAAI 2020)

## 一、直观理解

**问题:** 传统的分割类文本检测(如 PSENet)先预测"这个像素是不是文字"的概率图,再用一个固定阈值(比如 0.5)二值化,最后后处理成文本框。这个二值化步骤不可导,不能参与训练,而且固定阈值对粗细、密度不同的文字很不友好。

**类比:** 好比批改"这块区域是不是文字"的试卷,传统方法是所有人都按 60 分及格线划线,而 DBNet 让网络自己学一张"及格线地图"(阈值图):背景干净的地方线可以低一点,文字挤在一起的地方线要高一点。同时,"打分 → 判定及格"这一步被做成了平滑可导的函数,判卷标准本身也能被训练优化。

**核心一句话:** 把二值化放进网络里,让阈值图由网络自适应学习,并用一个可微的近似阶跃函数把它接入训练。

## 二、算法思想

### 1. 整体流程

```
输入图像 → Backbone(ResNet+FPN)→ 融合成 1/4 尺寸特征
         ├→ 概率图 P(每个像素属于文字的概率)
         └→ 阈值图 T(每个像素自适应阈值)
                ↓
      近似二值图 B̂ = 1 / (1 + exp(-k(P - T)))   (k=50)
```

- 训练时同时监督 P、T、B̂。
- **推理时只用 P**,阈值图可以丢掉,所以不增加推理开销,速度很快。

### 2. 可微二值化(DB)

标准二值化是阶跃函数:`B = 1 if P ≥ T else 0`,不可导。

DBNet 用带放大系数 k 的 sigmoid 近似它:

$$\hat B_{i,j}=\frac{1}{1+e^{-k(P_{i,j}-T_{i,j})}},\quad k=50$$

它的好处不只是"可导"。k 放大了梯度,对预测错误的区域(比如 P 略小于 T 却是文字)给出更大的梯度,有助于模型更好地区分文字边缘。

### 3. 标签生成(借鉴 PSENet)

用 Vatti clipping(`pyclipper`)处理多边形标注:

- **收缩文本区域(shrink map):** 向内收缩 D 得到概率图的标签,其中
  `D = A(1 - r²) / L`,A 为面积,L 为周长,r = 0.4。收缩的目的是把相邻文本实例分开。
- **阈值图标签:** 将原多边形向外扩张 D,得到的环带内每个像素取到最近边界的距离并归一化,在边界处值最大(最终缩放到 [0.3, 0.7]),越远离边界越小。

### 4. 损失函数

$$L = L_s + \alpha L_b + \beta L_t,\quad \alpha=1,\ \beta=10$$

| 项 | 对应 | 损失 |
|---|---|---|
| $L_s$ | 概率图 | BCE + 难例挖掘(正:负 = 1:3) |
| $L_b$ | 近似二值图 | 论文用 BCE,官方实现常用 Dice Loss |
| $L_t$ | 阈值图 | 扩张区域内的 L1 loss |

### 5. 推理后处理

1. 用固定阈值(如 0.3)二值化概率图 P。
2. 提取轮廓。
3. 因为训练时标签被收缩过,需要用 **unclip** 往外膨胀还原:`D' = A'·r'/L'`,r' = 1.5。
4. 按框内平均概率打分,过滤低分框(如 < 0.6)。

## 三、代码实现(PyTorch 核心部分)

### 1. Neck:FPN 融合

```python
import torch, torch.nn as nn, torch.nn.functional as F

class DBFPN(nn.Module):
    def __init__(self, in_channels=(64, 128, 256, 512), out_channels=256):
        super().__init__()
        self.in_convs = nn.ModuleList(
            [nn.Conv2d(c, out_channels, 1, bias=False) for c in in_channels])
        self.out_convs = nn.ModuleList(
            [nn.Conv2d(out_channels, out_channels // 4, 3, padding=1, bias=False)
             for _ in in_channels])

    def forward(self, feats):                      # feats: c2, c3, c4, c5
        inner = [conv(f) for conv, f in zip(self.in_convs, feats)]
        for i in range(len(inner) - 1, 0, -1):     # 自顶向下
            inner[i - 1] = inner[i - 1] + F.interpolate(
                inner[i], size=inner[i - 1].shape[2:], mode='nearest')
        outs = [conv(x) for conv, x in zip(self.out_convs, inner)]
        size = outs[0].shape[2:]                   # 统一到 1/4 尺寸后拼接
        outs = [outs[0]] + [F.interpolate(o, size=size, mode='nearest') for o in outs[1:]]
        return torch.cat(outs, dim=1)              # (B, 256, H/4, W/4)
```

### 2. Head:概率图 + 阈值图 + 近似二值图

```python
class DBHead(nn.Module):
    def __init__(self, in_ch=256, k=50):
        super().__init__()
        self.k = k
        def branch():
            return nn.Sequential(
                nn.Conv2d(in_ch, in_ch // 4, 3, padding=1, bias=False),
                nn.BatchNorm2d(in_ch // 4), nn.ReLU(True),
                nn.ConvTranspose2d(in_ch // 4, in_ch // 4, 2, 2),   # 上采样到原图尺寸
                nn.BatchNorm2d(in_ch // 4), nn.ReLU(True),
                nn.ConvTranspose2d(in_ch // 4, 1, 2, 2), nn.Sigmoid())
        self.binarize = branch()
        self.thresh = branch()

    def forward(self, x):
        p = self.binarize(x)
        if not self.training:
            return p                               # 推理只要概率图
        t = self.thresh(x)
        b = torch.reciprocal(1 + torch.exp(-self.k * (p - t)))  # 可微二值化
        return torch.cat([p, t, b], dim=1)
```

### 3. 损失函数

```python
class DBLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=10.0, ohem_ratio=3, eps=1e-6):
        super().__init__()
        self.alpha, self.beta, self.ratio, self.eps = alpha, beta, ohem_ratio, eps

    def bce_ohem(self, pred, gt, mask):
        pos, neg = (gt * mask), ((1 - gt) * mask)
        n_pos = int(pos.sum())
        n_neg = min(int(neg.sum()), n_pos * self.ratio)
        loss = F.binary_cross_entropy(pred, gt, reduction='none')
        pos_loss = (loss * pos).sum()
        neg_loss = (loss * neg).view(-1).topk(n_neg)[0].sum() if n_neg > 0 else 0
        return (pos_loss + neg_loss) / (n_pos + n_neg + self.eps)

    def dice(self, pred, gt, mask):
        inter = (pred * gt * mask).sum()
        union = (pred * mask).sum() + (gt * mask).sum() + self.eps
        return 1 - 2 * inter / union

    def forward(self, preds, batch):
        p, t, b = preds[:, 0], preds[:, 1], preds[:, 2]
        l_s = self.bce_ohem(p, batch['shrink_map'], batch['shrink_mask'])
        l_b = self.dice(b, batch['shrink_map'], batch['shrink_mask'])
        l_t = (torch.abs(t - batch['thresh_map']) * batch['thresh_mask']).sum() \
              / (batch['thresh_mask'].sum() + self.eps)
        return l_s + self.alpha * l_b + self.beta * l_t
```

### 4. 标签生成(核心片段)

```python
import numpy as np, cv2, pyclipper
from shapely.geometry import Polygon

def make_shrink_map(poly, shrink_map, r=0.4):
    pg = Polygon(poly)
    d = pg.area * (1 - r ** 2) / pg.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    shrunk = np.array(pco.Execute(-d)[0])
    cv2.fillPoly(shrink_map, [shrunk.astype(np.int32)], 1)

def make_thresh_map(poly, thresh_map, thresh_mask, r=0.4):
    pg = Polygon(poly)
    d = pg.area * (1 - r ** 2) / pg.length
    pco = pyclipper.PyclipperOffset()
    pco.AddPath(poly, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
    dilated = np.array(pco.Execute(d)[0])
    cv2.fillPoly(thresh_mask, [dilated.astype(np.int32)], 1)
    # 对扩张区域内每个像素计算到多边形各边的最小距离,归一化后写入 thresh_map
    # 即 1 - dist / d,最后整体映射到 [0.3, 0.7],这里省略逐像素距离计算的细节
```

### 5. 后处理

```python
def postprocess(prob_map, thresh=0.3, box_thresh=0.6, unclip_ratio=1.5):
    bitmap = (prob_map > thresh).astype(np.uint8)
    contours, _ = cv2.findContours(bitmap, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for c in contours:
        rect = cv2.minAreaRect(c)
        pts = cv2.boxPoints(rect)
        # 框内平均概率打分
        mask = np.zeros_like(bitmap); cv2.fillPoly(mask, [pts.astype(np.int32)], 1)
        score = (prob_map * mask).sum() / (mask.sum() + 1e-6)
        if score < box_thresh:
            continue
        # unclip:向外膨胀还原收缩的区域
        pg = Polygon(pts)
        d = pg.area * unclip_ratio / pg.length
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(pts.astype(np.int32).tolist(), pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        expanded = np.array(pco.Execute(d)[0])
        boxes.append(cv2.boxPoints(cv2.minAreaRect(expanded)))
    return boxes
```

## 四、要点回顾

| 维度 | 要点 |
|---|---|
| 核心创新 | 可微二值化 + 自适应阈值图 |
| 为什么快 | 推理时丢掉阈值分支,后处理简单 |
| 为什么准 | 阈值图让边界更精细;k 放大了错误区域的梯度 |
| 擅长 | 任意形状文本(弯曲文本)、实时检测 |
| 常见坑 | 标签生成里的 shrink/unclip 比例要配套;极小文本框容易被过滤;密集文本时 `unclip_ratio` 过大会粘连 |

后续改进可以看 **DBNet++**(加入自适应尺度融合模块 ASF),或者 PaddleOCR 里的 **PP-OCR 检测模型**(DBNet 的轻量化工程实现)。

如果你想要完整可训练的代码结构(数据集、训练循环、评估),我可以接着展开。



