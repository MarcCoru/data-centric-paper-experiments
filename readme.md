# Experiments for the Data-Centric review paper

[Paper Overleaf link](https://www.overleaf.com/project/64249095801a56f6b87ec122)

## Schedule

- [ ] implementation of 'common' code (data, method, training script) (deadline to be discussed, probably end of June)
- [ ] Experiment 1
- [ ] Experiment 2
- [ ] Experiment 3
- [ ] Experiment 4

Overall deadline July 31st 2023 (paper draft deadline)

## Method

ResNet-18

train baseline model:
```bash
export PYTHONPATH="${PYTHONPATH}:${PWD}/common
python common/train.py
```

## Data

* Sen12MS Dataset (training/validation)
* DFC2020 (evaluation)

## Experiments

### 1. Core-Set Selection (Marc)

Idea: select relevant samples (for one target DFC2020 region) from the Sen12MS dataset 

Output: A map of the relevant samples (they should be somewhat "close" to the test area)

### 2. To be fixed (e.g., label noise correction)

### 3. To be fixed (e.g., curriculum learning)

### 4. To be fixed (e.g., adversarial examples)
