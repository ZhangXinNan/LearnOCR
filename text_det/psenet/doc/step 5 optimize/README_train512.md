
# train

```bash
python train.py \
    --mode train \
    --pretrained \
    --size 512 \
    --batch-size 1 \
    --epochs 60 \
    --lr 0.0001 \
    --output outputs/psenet_step5/train_512
```

# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_512/best.pt \
    --split test \
    --size 512 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer512
```
TP=653, FP=969, FN=1424
Precision=0.4026  Recall=0.3144  F1=0.3531


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_512/best.pt \
    --split test \
    --size 768 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer768
```
TP=925, FP=1028, FN=1152
Precision=0.4736  Recall=0.4454  F1=0.4591


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_512/best.pt \
    --split test \
    --size 1024 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model512_infer1024
```
TP=1012, FP=1067, FN=1065
Precision=0.4868  Recall=0.4872  F1=0.4870


