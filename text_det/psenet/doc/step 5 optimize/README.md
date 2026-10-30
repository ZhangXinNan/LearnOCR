

# 第一轮：不重新训练，直接优化推理

## A · Baseline
```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val \
  --size 512 \
  --threshold 0.5 \
  --min-kernel-area 5 \
  --min-area 30 \
  --out outputs/psenet_optimization/A
```
TP=148, FP=212, FN=378
Precision=0.4111  Recall=0.2814  F1=0.3341

## B · 降低阈值
```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val \
  --size 512 \
  --threshold 0.4 \
  --min-kernel-area 5 \
  --min-area 30 \
  --out outputs/psenet_optimization/B
```
TP=142, FP=209, FN=384
Precision=0.4046  Recall=0.2700  F1=0.3238


## C · 保留小核心
```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val \
  --size 512 \
  --threshold 0.5 \
  --min-kernel-area 2 \
  --min-area 30 \
  --out outputs/psenet_optimization/C
```
TP=148, FP=213, FN=378
Precision=0.4100  Recall=0.2814  F1=0.3337


## D · 提高分辨率
```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val \
  --size 768 \
  --threshold 0.5 \
  --min-kernel-area 5 \
  --min-area 30 \
  --out outputs/psenet_optimization/D
```
TP=192, FP=247, FN=334
Precision=0.4374  Recall=0.3650  F1=0.3979


```bash
python evaluate.py \
  --checkpoint outputs/psenet_step5/train/best.pt \
  --split val \
  --size 1024 \
  --threshold 0.5 \
  --min-kernel-area 5 \
  --min-area 30 \
  --out outputs/psenet_optimization/D1024
```
TP=231, FP=239, FN=295
Precision=0.4915  Recall=0.4392  F1=0.4639


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split test \
    --size 512 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer512
```
TP=635, FP=791, FN=1442
Precision=0.4453  Recall=0.3057  F1=0.3625

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split test \
    --size 768 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer768
```
TP=897, FP=930, FN=1180
Precision=0.4910  Recall=0.4319  F1=0.4595

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train/best.pt \
    --split test \
    --size 1024 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer1024
```
TP=1041, FP=950, FN=1036
Precision=0.5229  Recall=0.5012  F1=0.5118

# 第二轮：延长训练时间

# 第三轮：尝试 768×768 训练
```bash
python train.py \
    --mode train \
    --pretrained \
    --size 768 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_768
```



```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split val \
    --split-json outputs/psenet_step5/train_768/split.json \
    --size 768 \
    --visualize 20 \
    --out outputs/psenet_step7/validation_768
```

## 768 -->> 768

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split val \
    --split-json outputs/psenet_step5/train_768/split.json \
    --size 768 \
    --out outputs/psenet_optimization/768model_768infer
```
TP=250, FP=214, FN=276
Precision=0.5388  Recall=0.4753  F1=0.5051

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --out outputs/psenet_step7/test_768
```

TP=1100, FP=726, FN=977
Precision=0.6024  Recall=0.5296  F1=0.5637

## 768 -->> 1024
```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split val \
    --split-json outputs/psenet_step5/train_768/split.json \
    --size 1024 \
    --out outputs/psenet_optimization/768model_1024infer
```
TP=303, FP=203, FN=223
Precision=0.5988  Recall=0.5760  F1=0.5872


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --size 1024 \
    --visualize 20 \
    --out outputs/psenet_step7/test_768model_1024infer
```
TP=1225, FP=671, FN=852
Precision=0.6461  Recall=0.5898  F1=0.6167


# 第三轮：尝试 1024×1024 训练
```bash
python train.py \
    --mode train \
    --pretrained \
    --size 1024 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_1024
```

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split test \
    --out outputs/psenet_step7/test_1024
```
TP=1078, FP=751, FN=999
Precision=0.5894  Recall=0.5190  F1=0.5520


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split val \
    --split-json outputs/psenet_step5/train_1024/split.json \
    --size 1024 \
    --out outputs/psenet_optimization/1024model_1024infer
```
TP=216, FP=229, FN=310
Precision=0.4854  Recall=0.4106  F1=0.4449

## 1024 -->> 768
```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split val \
    --split-json outputs/psenet_step5/train_1024/split.json \
    --size 768 \
    --out outputs/psenet_optimization/1024model_768infer
```
TP=190, FP=208, FN=336
Precision=0.4774  Recall=0.3612  F1=0.4113


