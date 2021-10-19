from spinup.utils.run_utils import ExperimentGrid
import spinup
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--env', required=True)
    parser.add_argument('--num_disc', type=int, required=True)
    parser.add_argument('--num_cont', type=int, required=True)
    parser.add_argument('--d_info', type=int)
    parser.add_argument('--directed', action='store_true')
    args = parser.parse_args()

    eg = ExperimentGrid(name='exp_grid')
    eg.add('env_name', args.env, '', True)
    eg.add('num_disc_skills', args.num_disc, 'disc', True)
    eg.add('num_cont_skills', args.num_cont, 'cont', True)
    if args.d_info is not None:
        eg.add('interval_max_JINFO', args.d_info, 'dinfo', True)
    if args.directed:
        eg.add('directed', args.directed, 'directed', True)
    eg.add('seed', [10*i for i in range(args.num_seeds)])
    eg.run(eval('spinup.lsac_pytorch'))
