# CREPE: Controlling diffusion with REPlica Exchange

CREPE is an inference-time control algorithm for diffusion models. 
This repo contains various applications of CREPE. 
The structure is as follows:

```
applications
    ├── tempering
    │        - code for inference-time tempering for Boltzmann sampling
    ├── stitching
    │        - code for stitching trajectories in the maze
    ├── cfg
    │        - debiasing cfg on ImageNet
    ├── reward-tilting
    │        - prompted reward-tilting on ImageNet
    └── ctmc
            - debiasing cfg for CTMC models
```
The required environment may differ for each application. Please refer to their own folder.
