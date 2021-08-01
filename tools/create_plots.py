import os
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    args = parser.parse_args()

    VALID_ENVIRONMENTS = ["hopper-v2", "walker2d-v2", "humanoid-v2"]
    VALID_EXP_NAME = "exp_grid"

    data_dict = {}

    for exp_by_date in os.scandir(args.data_dir):
        if exp_by_date.is_dir():
            if re.search(VALID_EXP_NAME, exp_by_date.name) is None:
                print(f"Skipping {exp_by_date.path} because its experiment name is not {VALID_EXP_NAME}")
                continue
            for exp_by_time in os.scandir(exp_by_date.path):
                m = re.search(f"(?P<env>{'(' + ')|('.join(VALID_ENVIRONMENTS) + ')'})"
                                     f"_(?P<disc>disc\d+)_(?P<cont>cont\d+)_(?P<seed>s\d+)", exp_by_time.name)
                try:
                    env_name = m.group('env')
                except AttributeError:
                    print(f"Skipping experiment {exp_by_time.name} because its env is not in the VALID_ENVIRONEMNTS"
                          f" {', '.join(VALID_ENVIRONMENTS)}")
                    continue
                try:
                    num_disc = m.group('disc')
                except AttributeError:
                    print(f"Skipping experiment {exp_by_time.name} because it is missing information about the number"
                          f"of DISCRETE skills of the experiment")
                    continue
                try:
                    num_cont = m.group('cont')
                except AttributeError:
                    print(f"Skipping experiment {exp_by_time.name} because it is missing information about the number"
                          f"of CONTINUOUS skills of the experiment")
                    continue
                skill_id = f"{num_disc}_{num_cont}"
                try:
                    seed = m.group('seed')
                except AttributeError:
                    print(f"Skipping experiment {exp_by_time.name} because it is missing information about the seed"
                          f"of the experiment")
                    continue

                if env_name not in data_dict.keys():
                    data_dict[env_name] = {}
                if skill_id not in data_dict[env_name].keys():
                    data_dict[env_name][skill_id] = {}
                if seed in data_dict[env_name][skill_id].keys():
                    print("Something is wrong here. The same experiment configuration is already in the data_dict")

                data_dict[env_name][skill_id][seed] = pd.read_csv(os.path.join(exp_by_time.path, 'progress.txt'),
                                                                  sep='\t')
    plot_dir = os.path.join(os.getcwd(), 'plots')
    os.makedirs(plot_dir, exist_ok=True)
    for env, skill_dict in data_dict.items():
        env_dir = os.path.join(plot_dir, env)
        os.makedirs(env_dir, exist_ok=True)
        for skill_id, seed_dict in skill_dict.items():
            skill_dir = os.path.join(plot_dir, skill_id)
            os.makedirs(skill_dir, exist_ok=True)

            ep_ret = None
            for seed, progress in seed_dict.items():
                if ep_ret is None:
                    ep_ret = np.matrix(progress['Epoch/EpRetAverage'])
                else:
                    ep_ret = np.concatenate((ep_ret, np.matrix(progress['Epoch/EpRetAverage'])))
            x = [i for i in range(ep_ret.shape[1])]

            plt.figure()
            plt.title(f"{env} {skill_id}")
            plt.plot(x, ep_ret.mean(axis=0).tolist()[0], color=(1,0,0,1))
            plt.fill_between(x, ep_ret.min(axis=0).tolist()[0], ep_ret.max(axis=0).tolist()[0], color=(1,0,0,0.3))
            plt.show()

            plt.figure()
            plt.title(f"{env} {skill_id}")
            for row in range(ep_ret.shape[0]):
                plt.plot(x, ep_ret[row,:].tolist()[0])
            plt.show()




