import time
import joblib
import os
import os.path as osp
import tensorflow as tf
import torch
from torch.utils.tensorboard import SummaryWriter
from spinup import EpochLogger
from spinup.utils.logx import restore_tf_graph


def load_policy_and_env(fpath, itr='last', deterministic=False, disc_skill=None, cont_skill=None):
    """
    Load a policy from save, whether it's TF or PyTorch, along with RL env.

    Not exceptionally future-proof, but it will suffice for basic uses of the
    Spinning Up implementations.

    Checks to see if there's a tf1_save folder. If yes, assumes the model
    is tensorflow and loads it that way. Otherwise, loads as if there's a
    PyTorch save.

    :arg disc_skill If policy is conditioned on a one-hot-encoded discrete skill, enter the number of the skill.
    The number of skills is printed when calling the function, so it is safe to first run test_policy with no skill.

    :arg cont_skill If policy is condition on one or more continuous-valued skills, these are passed to this function
    as a list of float values. Their value is best chosen between -1 and +1.
    """

    # determine if tf save or pytorch save
    if any(['tf1_save' in x for x in os.listdir(fpath)]):
        backend = 'tf1'
    else:
        backend = 'pytorch'

    # handle which epoch to load from
    if itr == 'last':
        # check filenames for epoch (AKA iteration) numbers, find maximum value

        if backend == 'tf1':
            saves = [int(x[8:]) for x in os.listdir(fpath) if 'tf1_save' in x and len(x) > 8]

        elif backend == 'pytorch':
            pytsave_path = osp.join(fpath, 'pyt_save')
            # Each file in this folder has naming convention 'modelXX.pt', where
            # 'XX' is either an integer or empty string. Empty string case
            # corresponds to len(x)==8, hence that case is excluded.
            saves = [int(x.split('.')[0][5:]) for x in os.listdir(pytsave_path) if len(x) > 8 and 'model' in x]

        itr = '%d' % max(saves) if len(saves) > 0 else ''

    else:
        assert isinstance(itr, int), \
            "Bad value provided for itr (needs to be int or 'last')."
        itr = '%d' % itr

    # load the get_action function
    if backend == 'tf1':
        get_action = load_tf_policy(fpath, itr, deterministic)
    else:
        get_action = load_pytorch_policy(fpath, itr, deterministic, disc_skill, cont_skill)

    # try to load environment from save
    # (sometimes this will fail because the environment could not be pickled)
    try:
        state = joblib.load(osp.join(fpath, 'vars' + itr + '.pkl'))
        env = state['env']
    except:
        env = None

    return env, get_action


def load_tf_policy(fpath, itr, deterministic=False):
    """ Load a tensorflow policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'tf1_save' + itr)
    print('\n\nLoading from %s.\n\n' % fname)

    # load the things!
    sess = tf.Session()
    model = restore_tf_graph(sess, fname)

    # get the correct op for executing actions
    if deterministic and 'mu' in model.keys():
        # 'deterministic' is only a valid option for SAC policies
        print('Using deterministic action op.')
        action_op = model['mu']
    else:
        print('Using default action op.')
        action_op = model['pi']

    # make function for producing an action given a single state
    get_action = lambda x: sess.run(action_op, feed_dict={model['x']: x[None, :]})[0]

    return get_action


def load_pytorch_policy(fpath, itr, deterministic=False, disc_skill=None, cont_skill=None):
    """ Load a pytorch policy saved with Spinning Up Logger."""

    fname = osp.join(fpath, 'pyt_save', 'model' + itr + '.pt')
    print('\n\nLoading from %s.\n\n' % fname)

    model = torch.load(fname)
    if hasattr(model, '_num_disc_skills'):
        num_disc_skills = model.num_disc_skills()
    else:
        num_disc_skills = 0

    if hasattr(model, '_num_cont_skills'):
        num_cont_skills = model.num_cont_skills()
    else:
        num_cont_skills = 0

    if num_disc_skills > 0:
        assert disc_skill is not None and (0 < disc_skill <= num_disc_skills), \
            f"The loaded model knows {num_disc_skills} different DISCRETE skills. " \
            f"Please provide the command line argument -ds with a value " \
            f"between 1 and {num_disc_skills}. You entered: {disc_skill}."
        print(f"Active discrete skill is {disc_skill}")
        disc_vec = torch.zeros(num_disc_skills)
        disc_vec[disc_skill - 1] = True
    else:
        disc_vec = torch.zeros(0)

    if num_cont_skills > 0:
        assert cont_skill is not None and len(cont_skill) == num_cont_skills, \
            f"The loaded model knows {num_cont_skills} different CONTINUOUS skills. " \
            f"Please provide the command line argument -cs with {num_cont_skills} space separated float values, " \
            f"preferably between -1 and +1 (e.g. -cs -0.5 0.6). " \
            f"You entered: {', '.join(str(x) for x in cont_skill) if cont_skill is not None else 'None'}."
        print(f"Active continuous skill vector is {cont_skill}")
        cont_vec = torch.as_tensor(cont_skill)
    else:
        cont_vec = torch.zeros(0)

    if num_disc_skills > 0 or num_cont_skills > 0:
        def get_action(x, writer: SummaryWriter, t):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                action = model.act(x, torch.cat((disc_vec, cont_vec)), deterministic)
                pred_disc_skill, pred_cont_skill, cont_skill_var = model.d(x, torch.as_tensor(action))
                if pred_disc_skill is not None:
                    writer.add_scalars(f"PredDiscSkill/(disc_skill={disc_skill},cont_skill={cont_skill})",
                                       {str(x + 1): y for x, y in enumerate(pred_disc_skill)}, t)
                if pred_cont_skill is not None:
                    writer.add_scalars(f"PredContSkill/(disc_skill={disc_skill},cont_skill={cont_skill})",
                                       {f"mu{x + 1}": y for x, y in enumerate(pred_cont_skill)})
            return action
    else:
        def get_action(x, writer, t):
            with torch.no_grad():
                x = torch.as_tensor(x, dtype=torch.float32)
                action = model.act(x, deterministic)
            return action

    return get_action


def run_policy(env, get_action, max_ep_len=None, num_episodes=100, render=True):
    assert env is not None, \
        "Environment not found!\n\n It looks like the environment wasn't saved, " + \
        "and we can't run the agent in it. :( \n\n Check out the readthedocs " + \
        "page on Experiment Outputs for how to handle this situation."

    logger = EpochLogger()
    writer = SummaryWriter(comment="test_policy")
    o, r, d, ep_ret, ep_len, n = env.reset(), 0, False, 0, 0, 0
    t = 0

    start_paused = render
    while n < num_episodes:
        if render:
            env.render()
            time.sleep(1e-3)
        if start_paused:
            input("Press ENTER to start")
            start_paused = False

        a = get_action(o, writer, t)
        o, r, d, _ = env.step(a)
        ep_ret += r
        ep_len += 1
        t += 1

        if d or (ep_len == max_ep_len):
            logger.store(EpRet=ep_ret, EpLen=ep_len)
            print('Episode %d \t EpRet %.3f \t EpLen %d' % (n, ep_ret, ep_len))
            o, r, d, ep_ret, ep_len = env.reset(), 0, False, 0, 0
            n += 1

    logger.log_tabular('EpRet', with_min_and_max=True)
    logger.log_tabular('EpLen', average_only=True)
    logger.dump_tabular()
    writer.flush()
    writer.close()


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('fpath', type=str)
    parser.add_argument('--len', '-l', type=int, default=0)
    parser.add_argument('--episodes', '-n', type=int, default=100)
    parser.add_argument('--norender', '-nr', action='store_true')
    parser.add_argument('--itr', '-i', type=int, default=-1)
    parser.add_argument('--deterministic', '-d', action='store_true')
    parser.add_argument('--disc-skill', '-ds', type=int, default=None,
                        help='Set the discrete skill of the agent (one-hot-encoded latent variable vector.')
    parser.add_argument('--cont-skill', '-cs', type=float, default=None, nargs='+',
                        help='Set the continuous skill vector with space-separated float values between -1 and +1 '
                             '(e.g. -cs 0.5 0.2).')
    args = parser.parse_args()
    env, get_action = load_policy_and_env(args.fpath,
                                          args.itr if args.itr >= 0 else 'last',
                                          args.deterministic,
                                          args.disc_skill,
                                          args.cont_skill)
    run_policy(env, get_action, args.len, args.episodes, not (args.norender))