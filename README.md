## Versatile Skill Learning through Deep RL

This repository is an (as of now incomplete) implementation of *Discovering Diverse Solutions in Deep Reinforcement 
Learning (Osa et al., 2021, [link](https://arxiv.org/abs/2103.07084))*. 


## Installation
Please follow the installation instructions of 
[spinningup](https://spinningup.openai.com/en/latest/user/installation.html) 
including the installation of [mujoco](http://www.mujoco.org/).
For this use case, the version 1.5 of mujoco is used.
Make sure the following environment variable is set. 

```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{PATH_TO_MUJOCO_LIBARY}/.mujoco/mjpro150/bin
```


## Training a policy

To run, first make sure the root of this repository is on your PYTHONPATH. Add it if necessary. 
```bash
export PYTHONPATH="${PYTHONPATH}:~/.../pr_versatile_skill_learning"
cd ~/.../pr_versatile_skill_learning
python -m spinup.run lsac --env HalfCheetah-v2  --exp_name LSAC-Cheetah
# or
python -m spinup.algos.pytorch.lsac.lsac --env HalfCheetah-v2  --exp_name LSAC-Cheetah
```

The LSAC algorithm also logs to Tensorboard files. These are placed in the `runs/` directory of your current working 
directory (cwd). It is important that your cwd is always the base of this repository. Run the following command in 
a separate terminal window (you might need to install Tensorboard first). 

```bash
tensorboard --logdir="~/.../pr_versatile_skill_learning/runs"
```

## Running a trained policy
Trained models are saved in the `data/` directory at the base of this repository. 
The following is an example of how to run one. 
When training a latent-conditioned policy, make sure to add the `--disc_skill` and/or `--cont_skill` flag. 
Not sure how many skills the agent has been trained on? 
Just run the agent without specifying the skill and the error message will tell you. 
```bash
# Example for an agent with discrete skills and a two-dimensional continuous skill
python -m spinup.run test_policy data/lsac/lsac_s0 --disc-skill 1 --cont-skill -0.3 0.7
# or
python -m spinup.utils.test_policy data/lsac/lsac_s0 --disc-skill 1 --cont-skill -0.3 0.7
```

