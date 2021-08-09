from spinup.utils.run_utils import ExperimentGrid
import spinup
import torch

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_seeds', type=int, default=5)
    parser.add_argument('--env', required=True)
    args = parser.parse_args()

    eg = ExperimentGrid(name='exp_grid_sac_')
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [10*i for i in range(args.num_seeds)])
    eg.run(eval('spinup.sac_pytorch'))
