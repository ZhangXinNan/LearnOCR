
# train

```bash
python train.py \
    --mode train \
    --pretrained \
    --size 320 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_320
```
54000,60,val,0.376604,0.436196,0.237559,0.540229

# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_320/best.pt \
    --split test \
    --size 320 \
    --visualize 20
```
TP=368, FP=733, FN=1709
Precision=0.3342  Recall=0.1772  F1=0.2316

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_320/best.pt \
    --split test \
    --size 640 \
    --visualize 20
```
TP=712, FP=1440, FN=1365
Precision=0.3309  Recall=0.3428  F1=0.3367


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_320/best.pt \
    --split test \
    --size 960 \
    --visualize 20
```
TP=755, FP=2181, FN=1322
Precision=0.2572  Recall=0.3635  F1=0.3012

