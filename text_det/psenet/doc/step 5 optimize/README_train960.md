
# train

```bash
python train.py \
    --mode train \
    --pretrained \
    --size 960 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_960
```

# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_960/best.pt \
    --split test \
    --size 640 \
    --visualize 20
```
TP=841, FP=633, FN=1236
Precision=0.5706  Recall=0.4049  F1=0.4737


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_960/best.pt \
    --split test \
    --size 960 \
    --visualize 20
```
TP=1190, FP=662, FN=887
Precision=0.6425  Recall=0.5729  F1=0.6058

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_960/best.pt \
    --split test \
    --size 1280 \
    --visualize 20
```
TP=1314, FP=687, FN=763
Precision=0.6567  Recall=0.6326  F1=0.6444

