# CREPE🥞: Controlling diffusion with REPlica Exchange

[![ICLR 2026](https://img.shields.io/badge/ICLR-2026-6f42c1.svg)](#)
[![arXiv](https://img.shields.io/badge/arXiv-2509.23265-b31b1b.svg)](https://arxiv.org/abs/2509.23265)
[![Website](https://img.shields.io/website?label=Live%20Demo&url=https%3A%2F%2Fcrepe-diffusion.github.io%2F)](https://crepe-diffusion.github.io/)


CREPE🥞 is an inference-time control algorithm for diffusion models (both Gaussian diffusion and CTMC!). 


Illustration of CREPE (right) and comparison with SMC control (left):
![](./assets/crepe2.png)



Example of CREPE for prompted reward-tilting on ImageNet-512:
![](./assets/crepe1.png)




This repo contains various applications of CREPE:
- [Tempering on Boltzmann distribution of Alanine Di/Tetra/Hexa-peptide](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/main/applications/tempering)
- [Trajectory stitching for maze](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/main/applications/stitching)
- [Debiasing CFG for image generation](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/main/applications/cfg)
- [Prompted reward-tilting for image generation](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/tree/main/applications/reward-tilting)
- Debiasing CFG on CTMC (Coming soon!)

  
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
            - debiasing cfg for CTMC models (🚧under construction, coming soon!)
```
The required environment may differ for each application. Please refer to their own folder.


Do not hesitate to reach out if you have any questions!


### Reference
```
@article{he2025crepe,
  title={CREPE: Controlling Diffusion with Replica Exchange},
  author={He, Jiajun and Jeha, Paul and Potaptchik, Peter and Zhang, Leo and Hern{\'a}ndez-Lobato, Jos{\'e} Miguel and Du, Yuanqi and Syed, Saifuddin and Vargas, Francisco},
  journal={arXiv preprint arXiv:2509.23265},
  year={2025}
}
```
