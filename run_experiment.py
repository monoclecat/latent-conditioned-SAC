from spinup.utils.run_utils import ExperimentGrid
import spinup
import torch

def createExpName(args, run_id):
    command = ''
    if args.exp_name == '':
        command = 'lsac_pytorch_' + str(run_id)
    else:
        command = args.exp_name + '_' + str(run_id)

    return command

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--cpu', type=int, default=1)
    parser.add_argument('--num_runs', type=int, default=3)
    parser.add_argument('--env', default='Hopper-v2')
    parser.add_argument('--exp_name', default='')
    args = parser.parse_args()

    eg = ExperimentGrid(name='lsac-pytorch-bench')
    eg.add('env_name', args.env, '', True)
    eg.add('seed', [10*i for i in range(args.num_runs)])
    # eg.add('exp_name', [createExpName(args, i) for i in range(args.num_runs)])
    eg.run(eval('spinup.lsac_pytorch'), num_cpu=args.cpu)