## Latent-conditioned Soft Actor-Critic

> This repository is an implementation of *Discovering Diverse Solutions in Deep Reinforcement Learning (Osa et al., 2021, [arXiv](https://arxiv.org/abs/2103.07084))*. 

Recent work in the field of deep reinforcement learning has been concerned with learning _skills_, diverse behaviors of the agent that can be chosen by setting the values of latent variables which the policy is conditioned on. 
In contrast to previous work which learn skills by modifying the reward, Osa et al. learn diverse skill via a separate objective. 
It is optimized alongside the unaltered soft actor-critic objectives. 

This repository is an implementation of their proposed algorithm, termed _latent-conditioned soft actor-critic (LSAC)_. 
The code is based on the [OpenAi SpinningUp](https://spinningup.openai.com/) implementation of soft actor-critic and uses **PyTorch**.

The LSAC implementation can be found under `spinup/algos/pytorch/lsac/`. 
The implementation uses SpinningUp's logging tools and additionally incorporates Tensorboard. 

## Installation
Please follow the installation instructions of [spinningup](https://spinningup.openai.com/en/latest/user/installation.html). 
This installation requires [MuJoCo](http://www.mujoco.org/), for which a license must be obtained. 
Make sure your Python version is no greater than 3.6, as this repository requires Tensorflow with a version <2.0.

For MuJoCo, make sure the following environment variable is set (we are using version 1.5).
```bash
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:{PATH_TO_MUJOCO_LIBARY}/.mujoco/mjpro150/bin
```

Make sure the root of this repository is on your PYTHONPATH. Add it if necessary. 
```bash
export PYTHONPATH="${PYTHONPATH}:~/.../pr_versatile_skill_learning"
```

## Training a policy

To train a policy, navigate into the base directory of this repo and run the command to start the LSAC training procedure.
```bash
cd ~/.../pr_versatile_skill_learning
python -m spinup.run lsac --env HalfCheetah-v2  --exp_name LSAC-Cheetah
```

The LSAC algorithm also logs to Tensorboard files. 
These are placed in the `runs/` directory of your current working directory. 
Run the following command in a separate terminal window (you might need to install Tensorboard first). 

```bash
tensorboard --logdir="~/.../pr_versatile_skill_learning/runs"
```

## Running a trained policy
Trained models are saved in the `data/` directory in the base of this repository. 
The following is an example of how to evaluate a trained agent. 

When training a latent-conditioned policy, make sure to add the `--disc_skill` and/or `--cont_skill` flag. 
> Not sure how many skills the agent has been trained on? Just run the agent without specifying the skill and the error message will tell you. 

```bash
# Example for an agent with discrete skills and a two-dimensional continuous skill
python -m spinup.run test_policy data/lsac/lsac_s0 --disc-skill 1 --cont-skill -0.3 0.7
```

### Modulation of continuous skills
When running a trained policy, one of the continuous skills can be modulated. 
This means the skill is set to plus/minus a value (default: `1`), switching every few seconds (default: `2.0`)

Example: Let the second continuous skill be modulated with an amplitude of `1.2`, switching from `-1.2` and `+1.2` and 
vice-versa every `3.5` seconds:
```bash
python -m spinup.run test_policy data/lsac/lsac_s0 --disc-skill 1 --cont-skill -0.3 0.7 -modu_skill 2 -modu_amp 1.2 -modu_t 3.5
```
