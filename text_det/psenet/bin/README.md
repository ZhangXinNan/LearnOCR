

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
conda create -n xinocr_py311 python=3.11

conda activate xinocr_py311

pip install torch torchvision
#pip3 install torch torchvision --user
#python3 -m pip install torch torchvision
```