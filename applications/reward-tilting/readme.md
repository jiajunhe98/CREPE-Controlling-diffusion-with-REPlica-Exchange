# Reward-tilting on ImageNet-64/512

## 🛠️Env and code preparation
Our code is based on the EDM2 repo directly.  Therefore, please clone the repo from [https://github.com/NVlabs/edm2/tree/main](https://github.com/NVlabs/edm2/tree/main) and install the environment from their repo.

### 👉Tips for installing ImageReward 

Skip this section if you have ImageRward correctly installed; otherwise, here are some tip (unfinished).



## ✨Sampling
```
python crepe-guidance-cfg-reward.py --img_net 64/512 --class_idxs 417/723/496/468/791
```

## 🌅Visualise and Evaluate

We provide a notebook to visualise and evaluate (CLIP and IR) for the sample obtained in ```visualise_and_evaluate.ipynb```.
