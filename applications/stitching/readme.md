# Stitching Trajectories with CREPE


## Environment

We save the required data trajectories in numpy, and hence the environment only requires basic packages (e.g., ```matplotlib```, ```numpy```) and ```pytorch```.

## Data preparation

Please download the short trajectory data and our pretrained network from [this link](https://drive.google.com/file/d/1lOyuDadwpcaL2_bRvfG6JYTvxhfUOctN/view?usp=sharing) and [this link](https://drive.google.com/file/d/1uak2ofiT-wp1S-syOPxSq2VXIsIE2R2L/view?usp=sharing).

Please put these files in this folder.


## Sampling

We provide two notebooks for this application.

1. [```maze_diffusion_official_task.ipynb```](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/main/applications/stitching/maze_diffusion_official_task.ipynb) stitches trajectories for 5 tasks consider in [CompDiffuser](https://github.com/devinluo27/comp_diffuser_release).

<!-- You can also choose to [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)]([https://drive.google.com/file/d/1_NoW59Oq2DIPxl4ZiFahqKXsCLXmMsl8/view?usp=sharing](https://colab.research.google.com/github/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/main/applications/stitching/colab/maze_diffusion_official_task_colab.ipynb)) -->


2. [```maze_diffusion_online.ipynb```](https://github.com/jiajunhe98/CREPE-Controlling-diffusion-with-REPlica-Exchange/blob/main/applications/stitching/maze_diffusion_online.ipynb) stitches trajectories to bridge between two points first, and then a new reward is added to enforce the trajectories to go through an intermediate points.





## Visualisation

Coming soon


