

```bash
mkdir psenet
cd psenet

python3 -m venv .venv
source .venv/bin/activate

python -m pip install --upgrade pip
python -m pip install torch torchvision numpy \
    opencv-python pyclipper matplotlib tqdm
```


```bash
conda create -n psenet_py311 python=3.11
conda activate psenet_py311


```