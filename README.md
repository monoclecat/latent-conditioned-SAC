## Versatile Skill Learning through Deep RL

This repository is an (as of now incomplete) implementation of *Discovering Diverse Solutions in Deep Reinforcement 
Learning (Osa et al., 2021, [link](https://arxiv.org/abs/2103.07084))*. 


## Installation
Please follow the installation instructions of [spinningup](https://spinningup.openai.com/en/latest/user/installation.html) including the installation of [mujoco](http://www.mujoco.org/).
For this use case, the version 1.5 of mujoco is used.
Please use the line to enable mujoco:
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{PATH_TO_MUJOCO_LIBARY}/.mujoco/mjpro150/bin
```


## Running

Run LSAC with: 
```bash
python -m spinup.run lsac --env HalfCheetah-v2
```

