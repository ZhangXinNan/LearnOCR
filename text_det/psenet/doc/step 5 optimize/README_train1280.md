
# train

```bash
python train.py \
    --mode train \
    --pretrained \
    --size 1280 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_1280
```

# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1280/best.pt \
    --split test \
    --size 640 \
    --visualize 20
```
TP=619, FP=543, FN=1458
Precision=0.5327  Recall=0.2980  F1=0.3822

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1280/best.pt \
    --split test \
    --size 960 \
    --visualize 20
```
TP=968, FP=670, FN=1109
Precision=0.5910  Recall=0.4661  F1=0.5211


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1280/best.pt \
    --split test \
    --size 1280 \
    --visualize 20
```
TP=1181, FP=638, FN=896
Precision=0.6493  Recall=0.5686  F1=0.6063


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1280/best.pt \
    --split test \
    --size 1600 \
    --visualize 20
```
TP=1238, FP=654, FN=839
Precision=0.6543  Recall=0.5961  F1=0.6238



