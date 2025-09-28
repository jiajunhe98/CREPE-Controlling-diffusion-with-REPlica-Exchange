# Inference-time tempering with CREPE

This folder contains code for CREPE on inference-time tempering. 

### Env

Our code runs with ```python==3.11``` and ```pytorch==2.1.0```. However, it should be compatible with other version as well.
Additionally, the code requires to install ```openmm``` and ```openmmtools``` as follows:

```
conda install -c conda-forge openmm openmmtools
```

If one needs to evaluate TICA of the generated sample, please also install ```pyEMMA``` from [http://www.emma-project.org/latest/index.html](http://www.emma-project.org/latest/index.html). This is not necessary for the sampling part.



### Sampling 

```
python main.py --target a4/a6/aldp
```



### Evaluation TVD, W2 and TICA MMD

```
coming soon
```


