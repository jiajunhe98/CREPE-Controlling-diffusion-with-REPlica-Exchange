# CREPE🥞: Controlling diffusion with REPlica Exchange

[![arXiv](https://img.shields.io/badge/arXiv-2509.23265-b31b1b.svg)](https://arxiv.org/abs/2509.23265)

CREPE🥞 is an inference-time control algorithm for diffusion models. 
This repo contains various applications of CREPE:
- Tempering with CREPE for Boltzmann Sampling
- Trajectory stitching with CREPE for maze
- Debiasing CFG with CREPE for image generation
- Prompted reward-tilting with CREPE  for image generation
- Debiasing CFG with CREPE on CTMC

  
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
            - debiasing cfg for CTMC models (coming soon!)
```
The required environment may differ for each application. Please refer to their own folder.

🚧The code for ctmc is still under construction. 
If you need anything on these tasks please leave an issue or email us.
