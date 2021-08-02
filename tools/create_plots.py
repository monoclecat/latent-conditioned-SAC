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

    if os.path.basename(args.data_dir) != 'data':
        print(f"This function is meant to be run in the base /data directory! You are running this in the "
              f"{os.path.basename(args.data_dir)} directory. Please press ENTER to continue anyway.")
        input()

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
            skill_dir = os.path.join(env_dir, skill_id)
            os.makedirs(skill_dir, exist_ok=True)

            for col in [('Epoch/EpRetAverage', 'Avg Episode Reward'),
                        ('LogProb/LogPiAverage', 'Avg Policy LogProb'),
                        ('Entropy/DiscreteAverage', 'Avg Entropy Discrete Skills'),
                        ('Entropy/Continuous1Average', 'Avg Entropy Continuous Skill #1'),
                        ('Entropy/Continuous2Average', 'Avg Entropy Continuous Skill #2')]:
                data = None  # Matrix of values over all seeds of this env and skill combination
                for seed, progress in seed_dict.items():
                    try:
                        if any(np.isnan(progress[col[0]])):
                            break
                        if data is None:
                            data = np.matrix(progress[col[0]])
                        else:
                            data = np.concatenate((data, np.matrix(progress[col[0]])))
                    except KeyError:
                        break

                if data is None:
                    continue

                x = [i for i in range(data.shape[1])]

                plt.figure()
                title = f"{col[1]} (from min to max) {env} {skill_id}"
                plt.title(title)
                plt.fill_between(x, data.min(axis=0).tolist()[0], data.max(axis=0).tolist()[0], color=(1,0.7,0.7))
                plt.plot(x, data.mean(axis=0).tolist()[0], color=(1,0,0))
                plt.savefig(os.path.join(skill_dir, title.replace(' ', '_')+'.eps'))
                plt.show()

                # The Kumar et al. way
                plt.figure()
                title = f"{col[1]} (plus-minus 0.5 std) {env} {skill_id}"
                plt.title(title)
                std = data.std(axis=0)*0.5
                plt.fill_between(x, (data.mean(axis=0)-std).tolist()[0], (data.mean(axis=0)+std).tolist()[0], color=(1,0.7,0.7))
                plt.plot(x, data.mean(axis=0).tolist()[0], color=(1,0,0))
                plt.savefig(os.path.join(skill_dir, title.replace(' ', '_')+'.eps'))
                plt.show()

                plt.figure()
                title = f"{col[1]} (all) {env} {skill_id}"
                plt.title(title)
                for row in range(data.shape[0]):
                    plt.plot(x, data[row,:].tolist()[0])
                plt.savefig(os.path.join(skill_dir, title.replace(' ', '_')+'.eps'))
                plt.show()
