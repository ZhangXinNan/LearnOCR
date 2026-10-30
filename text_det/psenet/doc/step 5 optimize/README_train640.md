
# train

```bash
python train.py \
    --mode train \
    --pretrained \
    --size 640 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_640
```

# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_640/best.pt \
    --split test \
    --size 640 \
    --visualize 20
```
TP=790, FP=794, FN=1287
Precision=0.4987  Recall=0.3804  F1=0.4316

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_640/best.pt \
    --split test \
    --size 960 \
    --visualize 20
```
TP=1038, FP=913, FN=1039
Precision=0.5320  Recall=0.4998  F1=0.5154

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_640/best.pt \
    --split test \
    --size 1280 \
    --visualize 20
```
TP=1115, FP=1133, FN=962
Precision=0.4960  Recall=0.5368  F1=0.5156

