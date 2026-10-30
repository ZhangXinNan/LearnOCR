
# train
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




# test

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --size 512 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model768_infer512
```
TP=744, FP=742, FN=1333
Precision=0.5007  Recall=0.3582  F1=0.4176


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --size 768 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model768_infer768
```
TP=1100, FP=726, FN=977
Precision=0.6024  Recall=0.5296  F1=0.5637


```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --size 1024 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model768_infer1024
```
TP=1225, FP=671, FN=852
Precision=0.6461  Recall=0.5898  F1=0.6167

```bash
python evaluate.py \
    --checkpoint outputs/psenet_step5/train_768/best.pt \
    --split test \
    --size 1536 \
    --visualize 20 \
    --out outputs/psenet_step7/test_model768_infer1536
```

TP=1203, FP=624, FN=874
Precision=0.6585  Recall=0.5792  F1=0.6163



