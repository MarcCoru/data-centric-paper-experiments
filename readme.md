# Experiments for the Data-Centric review paper

[Paper Overleaf link](https://www.overleaf.com/project/64249095801a56f6b87ec122)

## Method

ResNet-18

train baseline model:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/common
python common/train.py
```

## DFC2020 Data

download DFC data in the `datasets` directory via 
```
python datasets/download_dfc.py
```

