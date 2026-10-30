
# train

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


# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split test \
    --size 512 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model1024_infer512
```
TP=500, FP=611, FN=1577
Precision=0.4500  Recall=0.2407  F1=0.3137


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split test \
    --size 768 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model1024_infer768
```
TP=846, FP=736, FN=1231
Precision=0.5348  Recall=0.4073  F1=0.4624


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split test \
    --size 1024 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model1024_infer1024
```
TP=1078, FP=751, FN=999
Precision=0.5894  Recall=0.5190  F1=0.5520


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_1024/best.pt \
    --split test \
    --size 1536 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model1024_infer1536
```
TP=1221, FP=795, FN=856
Precision=0.6057  Recall=0.5879  F1=0.5966



