from copy import deepcopy
import itertools
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from torch.utils.tensorboard import SummaryWriter
import gym
import time
import spinup.algos.pytorch.lsac.core as core
from spinup.utils.logx import EpochLogger
from spinup.utils.run_utils import setup_logger_kwargs


class ReplayBuffer:
    """
    Store tuples of observation, action, next observation, reward, latent (skill) variable and done signal
    according to the "Algorithm 1" box in the paper.
    In the paper, the done signal is not required to be in the replay buffer.
    """

    def __init__(self, obs_dim, act_dim, disc_skill_dim, cont_skill_dim, size):
        self.obs_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.obs2_buf = np.zeros(core.combined_shape(size, obs_dim), dtype=np.float32)
        self.act_buf = np.zeros(core.combined_shape(size, act_dim), dtype=np.float32)
        self.rew_buf = np.zeros(size, dtype=np.float32)
        self.disc_skill_buf = np.zeros(core.combined_shape(size, disc_skill_dim), dtype=np.bool_)
        self.cont_skill_buf = np.zeros(core.combined_shape(size, cont_skill_dim), dtype=np.float32)
        self.done_buf = np.zeros(size, dtype=np.float32)
        self.ptr, self.size, self.max_size = 0, 0, size

    def store(self, obs, act, rew, disc_skill, cont_skill, next_obs, done):
        self.obs_buf[self.ptr] = obs
        self.obs2_buf[self.ptr] = next_obs
        self.act_buf[self.ptr] = act
        self.rew_buf[self.ptr] = rew
        self.disc_skill_buf[self.ptr] = disc_skill
        self.cont_skill_buf[self.ptr] = cont_skill
        self.done_buf[self.ptr] = done
        self.ptr = (self.ptr+1) % self.max_size
        self.size = min(self.size+1, self.max_size)

    def sample_batch(self, batch_size=32):
        idxs = np.random.randint(0, self.size, size=batch_size)
        batch = dict(obs=self.obs_buf[idxs],
                     obs2=self.obs2_buf[idxs],
                     act=self.act_buf[idxs],
                     rew=self.rew_buf[idxs],
                     disc_skill=self.disc_skill_buf[idxs],
                     cont_skill=self.cont_skill_buf[idxs],
                     done=self.done_buf[idxs])
        return {k: torch.as_tensor(v, dtype=torch.float32) for k,v in batch.items()}


def lsac(env_fn, actor_critic=core.OsaSkillActorCritic, ac_kwargs=dict(), seed=0,
         steps_per_epoch=5000, epochs=100, replay_size=int(1e6), gamma=0.99,
         polyak=0.995, lr=3e-4, alpha=0.1, batch_size=256, start_steps=10000,
         update_after=4096, num_test_episodes=10,
         logger_kwargs=dict(), save_freq=1, num_disc_skills=0, num_cont_skills=2,
         interval_max_JQ = 2, interval_max_JINFO = 3, clip=0.2):
    """
    Latent-Conditioned Soft Actor-Critic (LSAC)


    Args:
        env_fn : A function which creates a copy of the environment.
            The environment must satisfy the OpenAI Gym API.

        actor_critic: The constructor method for a PyTorch Module with an ``act`` 
            method, a ``pi`` module, a ``q1`` module, and a ``q2`` module.
            The ``act`` method and ``pi`` module should accept batches of 
            observations as inputs, and ``q1`` and ``q2`` should accept a batch 
            of observations and a batch of actions as inputs. When called, 
            ``act``, ``q1``, and ``q2`` should return:

            ===========  ================  ======================================
            Call         Output Shape      Description
            ===========  ================  ======================================
            ``act``      (batch, act_dim)  | Numpy array of actions for each 
                                           | observation.
            ``q1``       (batch,)          | Tensor containing one current estimate
                                           | of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ``q2``       (batch,)          | Tensor containing the other current 
                                           | estimate of Q* for the provided observations
                                           | and actions. (Critical: make sure to
                                           | flatten this!)
            ===========  ================  ======================================

            Calling ``pi`` should return:

            ===========  ================  ======================================
            Symbol       Shape             Description
            ===========  ================  ======================================
            ``a``        (batch, act_dim)  | Tensor containing actions from policy
                                           | given observations.
            ``logp_pi``  (batch,)          | Tensor containing log probabilities of
                                           | actions in ``a``. Importantly: gradients
                                           | should be able to flow back into ``a``.
            ===========  ================  ======================================

        ac_kwargs (dict): Any kwargs appropriate for the ActorCritic object 
            you provided to SAC.

        seed (int): Seed for random number generators.

        steps_per_epoch (int): Number of steps of interaction (state-action pairs) 
            for the agent and the environment in each epoch.

        epochs (int): Number of epochs to run and train agent.

        replay_size (int): Maximum length of replay buffer.

        gamma (float): Discount factor. (Always between 0 and 1.)

        polyak (float): Interpolation factor in polyak averaging for target 
            networks. Target networks are updated towards main networks 
            according to:

            .. math:: \\theta_{\\text{targ}} \\leftarrow 
                \\rho \\theta_{\\text{targ}} + (1-\\rho) \\theta

            where :math:`\\rho` is polyak. (Always between 0 and 1, usually 
            close to 1.)

        lr (float): Learning rate (used for both policy and value learning).

        alpha (float): Entropy regularization coefficient. (Equivalent to 
            inverse of reward scale in the original SAC paper.)

        batch_size (int): Minibatch size for SGD.

        start_steps (int): Number of steps for uniform-random action selection,
            before running real policy. Helps exploration.

        update_after (int): Number of env interactions to collect before
            starting to do gradient descent updates. Ensures replay buffer
            is full enough for useful updates.

        update_every (int): Number of env interactions that should elapse
            between gradient descent updates. Note: Regardless of how long 
            you wait between updates, the ratio of env steps to gradient steps 
            is locked to 1.

        num_test_episodes (int): Number of episodes to test the deterministic
            policy at the end of each epoch.

        logger_kwargs (dict): Keyword args for EpochLogger.

        save_freq (int): How often (in terms of gap between epochs) to save
            the current policy and value function.

        num_disc_skills (int): The dimension of the discrete latent variable vector

        num_cont_skills (int): The dimension of the continuous latent variable vector

        interval_max_JQ: The interval for maximizing JQ

        interval_max_JINFO: The interval for maximizing JInfo

        clip (float): The importance weight clipping hyperparameter
    """
    total_num_skills = num_disc_skills + num_cont_skills
    torch.manual_seed(seed)
    np.random.seed(seed)

    env, test_env = env_fn(), env_fn()
    env_name = env.spec.id

    if len(logger_kwargs) == 0:
        logger_kwargs = setup_logger_kwargs(f"{env_name}_{num_disc_skills}disc_{num_cont_skills}cont_{args.exp_name}", args.seed)
        logger_kwargs['exp_name'] = args.exp_name
    logger = EpochLogger(**logger_kwargs)
    logger.save_config(locals())

    writer = SummaryWriter(comment=f"_{env_name}_{num_disc_skills}disc_{num_cont_skills}cont_{logger_kwargs['exp_name']}")
    # Make sure that current working dir is pr_versatile_skill_learning
    # Open tensorboard in a separate terminal with: tensorboard --logdir="~/.../pr_versatile_skill_learning/runs"

    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, num_disc_skills, num_cont_skills, **ac_kwargs)
    ac_targ = deepcopy(ac)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim,
                                 disc_skill_dim=num_disc_skills, cont_skill_dim=num_cont_skills, size=replay_size)

    # Count variables (protip: try to get a feel for how different size networks behave!)
    var_counts = tuple(core.count_vars(module) for module in [ac.pi, ac.q1, ac.q2])
    logger.log('\nNumber of parameters: \t pi: %d, \t q1: %d, \t q2: %d\n'%var_counts)

    # Set up function for computing SAC Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        z = torch.cat((data['disc_skill'], data['cont_skill']), dim=-1)

        q1 = ac.q1(o, a, z)
        q2 = ac.q2(o, a, z)

        # Bellman backup for Q functions
        with torch.no_grad():
            # Target actions come from *current* policy
            a2, logp_a2 = ac.pi(o2, z)

            # Target Q-values
            q1_pi_targ = ac_targ.q1(o2, a2, z)
            q2_pi_targ = ac_targ.q2(o2, a2, z)
            q_pi_targ = torch.min(q1_pi_targ, q2_pi_targ)
            backup = r + gamma * (1 - d) * (q_pi_targ - alpha * logp_a2)

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        writer.add_scalar("Q_Values/Q1_mean", q1.detach().mean(), t)
        writer.add_scalar("Q_Values/Q2_mean", q2.detach().mean(), t)
        q_info = dict(Q1Vals=q1.detach().numpy(),
                      Q2Vals=q2.detach().numpy())

        return loss_q, q_info

    # Set up function for computing SAC pi loss
    def compute_loss_pi(data):
        o = data['obs']
        z = torch.cat((data['disc_skill'], data['cont_skill']), dim=-1)

        pi, logp_pi = ac.pi(o, z)
        q1_pi = ac.q1(o, pi, z)
        q2_pi = ac.q2(o, pi, z)
        q_pi = torch.min(q1_pi, q2_pi)

        # Entropy-regularized policy loss
        loss_pi = (alpha * logp_pi - q_pi).mean()

        # Useful info for logging
        pi_info = dict(LogPi=logp_pi.detach().numpy())

        return loss_pi, pi_info

    def compute_loss_info(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']
        z_disc, z_cont = data['disc_skill'], data['cont_skill']
        z = torch.cat((data['disc_skill'], data['cont_skill']), dim=-1)

        q1_batch = ac.q1(o, a, z)
        q2_batch = ac.q2(o, a, z)
        q_batch = torch.minimum(q1_batch, q2_batch)
        q_batch_max = torch.max(q_batch).detach()  # detach this so we don't run into a problem with autograd
        q_batch -= q_batch_max

        pi, logp_pi = ac.pi(o, z, deterministic=True)  # deterministic because we don't want exploration noise

        q1_pi = ac.q1(o, pi, z)
        q2_pi = ac.q2(o, pi, z)
        q_pi = torch.minimum(q1_pi, q2_pi) - q_batch_max
        q_pi = torch.minimum(q_pi, torch.as_tensor(85.0))

        q_batch.exp_()
        q_pi.exp_()
        imp_weight = torch.div(q_pi, q_batch.sum())
        w_clip = torch.clamp(imp_weight, 1 - clip, 1 + clip)

        disc_logits, cont_mu, cont_var = ac.d(obs=o, act=pi)

        if disc_logits is not None:
            _, active_disc_skill = np.where(z_disc == 1)
            disc_loss_info = F.cross_entropy(disc_logits, torch.tensor(active_disc_skill), reduction='none')
            # writer.add_scalar("Loss/J_info_pre_W_scale", loss_info.mean(), t)
            disc_loss_info = disc_loss_info.mul(w_clip).mean()
            # loss_info = computeCrossEntropyLoss(logits=logits, target=z, weights=w_clip)
        else:
            disc_loss_info = None

        if cont_mu is not None:
            cont_loss_info = F.gaussian_nll_loss(input=cont_mu, target=z_cont, var=cont_var, reduction='none')

            # https://stackoverflow.com/q/53987906/5568461
            cont_loss_info = cont_loss_info.mul(w_clip.unsqueeze(dim=1))
            cont_loss_info = cont_loss_info.mean()
        else:
            cont_loss_info = None

        return disc_loss_info, cont_loss_info

    # Set up optimizers for policy, q-function and discriminator
    pi_optimizer = Adam(ac.pi.parameters(), lr=lr)
    q_optimizer = Adam(q_params, lr=lr)
    d_optimizer = Adam(ac.d.parameters(), lr=lr)

    # Set up model saving
    logger.setup_pytorch_saver(ac)

    def update_critics(data):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, q_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        writer.add_scalar("Loss/Q", loss_q.item(), t)
        logger.store(LossQ=loss_q.item(), **q_info)

    def update_actor(data):
        # Run one gradient descent step for pi.
        pi_optimizer.zero_grad()
        loss_pi, pi_info = compute_loss_pi(data)
        loss_pi.backward()
        pi_optimizer.step()

        writer.add_scalar("Loss/Pi", loss_pi.item(), t)
        writer.add_scalar("LogProb/Avg/LogPi", np.mean(pi_info['LogPi']), t)
        writer.add_scalar("LogProb/Std/LogPi", np.std(pi_info['LogPi']), t)
        logger.store(LossPi=loss_pi.item(), **pi_info)

        # Finally, update target networks by polyak averaging.
        with torch.no_grad():
            for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                # NB: We use an in-place operations "mul_", "add_" to update target
                # params, as opposed to "mul" and "add", which would make new tensors.
                p_targ.data.mul_(polyak)
                p_targ.data.add_((1 - polyak) * p.data)

    def update_J_info(data):
        d_optimizer.zero_grad()
        pi_optimizer.zero_grad()
        disc_loss_J_info, cont_loss_J_info = compute_loss_info(data)
        if disc_loss_J_info is not None:
            disc_loss_J_info.backward(retain_graph=cont_loss_J_info is not None)
            writer.add_scalar("Loss/disc_J_info", disc_loss_J_info.item(), t)
            logger.store(LossD=disc_loss_J_info.item())
        if cont_loss_J_info is not None:
            cont_loss_J_info.backward()
            writer.add_scalar("Loss/cont_J_info", cont_loss_J_info.item(), t)
        d_optimizer.step()
        pi_optimizer.step()

    def get_action(o, disc_skills, cont_skills, deterministic=False):
        skills = torch.cat((torch.as_tensor(disc_skills, dtype=torch.float32),
                            torch.as_tensor(cont_skills, dtype=torch.float32)), dim=-1)
        return ac.act(torch.as_tensor(o, dtype=torch.float32), skills, deterministic)

    def run_test_episode(disc_skill_vec, cont_skill_vec):
        ep_ret_s, ep_len_s = ([] for _ in range(2))
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not (d or (ep_len == env.spec.max_episode_steps)):
                # Take deterministic actions at test time
                o, r, d, _ = test_env.step(get_action(o, disc_skill_vec, cont_skill_vec, deterministic=True))
                ep_ret += r
                ep_len += 1
            ep_ret_s.append(ep_ret)
            ep_len_s.append(ep_len)
        return ep_ret_s, ep_len_s

    def test_skills():
        ep_ret_a, ep_len_a = ([] for _ in range(2))
        for i in range(num_disc_skills):
            skill_one_hot = np.zeros(num_disc_skills)
            skill_one_hot[i] = 1
            ep_ret_s, ep_len_s = run_test_episode(skill_one_hot, np.zeros(num_cont_skills))
            ep_ret_a.append(ep_ret_s)
            ep_len_a.append(ep_len_s)
            writer.add_scalar(f"TestDiscSkills_AvgEpReturn/{i+1}", np.mean(ep_ret_s), epoch)
            writer.add_scalar(f"TestDiscSkills_AvgEpLength/{i+1}", np.mean(ep_len_s), epoch)
            # logger.store(**{f"TestEpRet-Skill{i+1}": ep_ret_s}, **{f"TestEpLen-Skill{i+1}": ep_len_s})
        for i in range(num_cont_skills):
            for val in [-0.9, -0.45, 0.45, 0.9]:
                skill_one_hot = np.zeros(num_cont_skills)
                skill_one_hot[i] = val
                ep_ret_s, ep_len_s = run_test_episode(np.zeros(num_disc_skills), skill_one_hot)
                ep_ret_a.append(ep_ret_s)
                ep_len_a.append(ep_len_s)
                writer.add_scalar(f"TestContSkills_AvgEpReturn/{i + 1}={val}", np.mean(ep_ret_s), epoch)
                writer.add_scalar(f"TestContSkills_AvgEpLength/{i + 1}={val}", np.mean(ep_len_s), epoch)
        return ep_ret_a, ep_len_a

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    start_time = time.time()
    o, ep_ret, ep_len = env.reset(), 0, 0

    if num_disc_skills > 0:
        disc_skill = np.zeros(num_disc_skills)
        disc_skill[np.random.randint(num_disc_skills)] = True
    else:
        disc_skill = np.empty(0)
    cont_skill = np.random.random(size=num_cont_skills) * 2 - 1  # Init cont var between -1 and +1

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        # Until start_steps have elapsed, randomly sample actions
        # from a uniform distribution for better exploration. Afterwards, 
        # use the learned policy. 
        if t > start_steps:
            a = get_action(o, disc_skill, cont_skill)
        else:
            a = env.action_space.sample()

        # Step the env.
        if env_name == "Hopper-v2" or env_name == "Walker2d-v2":
            # Modifications to reward according to Osa et al.
            posbefore = env.sim.data.qpos[0]
            o2, _, d, _ = env.step(a)
            posafter, height, ang = env.sim.data.qpos[0:3]

            if env_name == "Hopper-v2":
                r = min((posafter - posbefore) / env.dt, 1)
            else:
                # Walker2d-v2
                r = min((posafter - posbefore) / env.dt, 2)

            r += 1.0
            r -= 1e-3 * np.square(a).sum()
        else:
            o2, r, d, _ = env.step(a)

        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time
        # horizon (that is, when it's an artificial terminal signal
        # that isn't based on the agent's state)
        d = False if ep_len==env.spec.max_episode_steps else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, disc_skill, cont_skill, o2, d)

        # Super critical, easy to overlook step: make sure to update 
        # most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == env.spec.max_episode_steps):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            o, ep_ret, ep_len = env.reset(), 0, 0

            if num_disc_skills > 0:
                disc_skill = np.zeros(num_disc_skills)
                disc_skill[np.random.randint(num_disc_skills)] = True
            else:
                disc_skill = np.empty(0)
            cont_skill = np.random.random(size=num_cont_skills) * 2 - 1  # Init cont var between -1 and +1

        if t >= update_after:
            # Not using update_every from orig SAC, as Osa don't say anything about this
            batch = replay_buffer.sample_batch(batch_size)
            update_critics(data=batch)

            update_JQ = t % interval_max_JQ == 0
            update_JINFO = t % interval_max_JINFO == 0

            if update_JQ or update_JINFO:
                # Freeze Q-networks so you don't waste computational effort
                # computing gradients for them during the policy learning step.
                for p in q_params:
                    p.requires_grad = False

            # different update intervals for the critic, the actor and the info-objective
            if update_JQ:
                update_actor(data=batch)
            if update_JINFO:
                update_J_info(data=batch)

            if update_JQ or update_JINFO:
                # Unfreeze Q-networks so you can optimize it at next DDPG step.
                for p in q_params:
                    p.requires_grad = True

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Save model
            if (epoch % save_freq == 0) or (epoch == epochs):
                logger.save_state({'env': env}, None)

            writer.add_scalar("Epoch/AvgEpRet", np.mean(logger.epoch_dict.get('EpRet')), epoch)
            writer.add_scalar("Epoch/AvgEpLen", np.mean(logger.epoch_dict.get('EpLen')), epoch)

            # Test the performance of the deterministic version of the agent.
            test_skills()

            # Log info about epoch
            logger.log_tabular('Epoch', epoch)
            logger.log_tabular('EpRet', with_min_and_max=True)
            logger.log_tabular('EpLen', average_only=True)
            # for i in range(num_skills):
            #     logger.log_tabular(f"TestEpRet-Skill{i+1}", with_min_and_max=True)
            #     logger.log_tabular(f"TestEpLen-Skill{i+1}", average_only=True)
            logger.log_tabular('TotalEnvInteracts', t)
            if t > update_after:
                logger.log_tabular('Q1Vals', with_min_and_max=True)
                logger.log_tabular('Q2Vals', with_min_and_max=True)
                logger.log_tabular('LogPi', with_min_and_max=True)
                logger.log_tabular('LossPi', average_only=True)
                logger.log_tabular('LossQ', average_only=True)
            logger.log_tabular('Time', time.time()-start_time)
            logger.dump_tabular()
    writer.flush()
    writer.close()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v2')
    parser.add_argument('--seed', '-s', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--exp_name', type=str, default='lsac')
    args = parser.parse_args()


    torch.set_num_threads(torch.get_num_threads())

    # ac = torch.load("data/NaN_investigation_Hopper_cont_from_checkpoint/NaN_investigation_Hopper_cont_from_checkpoint_s0/pyt_save/model.pt")
    ac = core.OsaSkillActorCritic

    lsac(lambda: gym.make(args.env), actor_critic=ac,
         seed=args.seed, epochs=args.epochs)
