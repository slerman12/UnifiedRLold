import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = "./exp_local"

exps = []
# for exps_by_date in sorted(glob.glob(root + "/*/"))[-2:]:
for exps_by_date in sorted(glob.glob(root + "/*/")):
    exps = exps + glob.glob(exps_by_date + "/*/")
exps = sorted(exps)

results = {}

for exp in exps:
    if "experiment=" not in exp:
        continue
    name = exp.split("experiment=", 1)[1].split(",", 1)[0]
    # name = "bla"
    seed = int(exp.split("seed=", 1)[1].split(",", 1)[0])

    for env in glob.glob(exp + "/*"):
        env = env.split("/")[-1]

        try:
            train = pd.read_csv(exp + f'{env}/train.csv')
            eval = pd.read_csv(exp + f'{env}/eval.csv')
        except:
            continue

        if name not in results:
            results[name] = {}

        if env not in results[name]:
            results[name][env] = {}

        # dict of envs, each has dict of seeds, each has train csv and eval csv
        results[name][env].update({seed: {"train": train, "eval": eval}})

        # train.plot(x="frame", y="episode_reward", legend=None)
        # plt.xlabel('Frame')
        # plt.ylabel('Reward')
        # plt.title(env.capitalize())
        # plt.show()
        # break


        # with open(file) as f:
        #     lines = f.readlines()


RANDOM_SCORES = {
    'Alien': 227.8,
    'Amidar': 5.8,
    'Assault': 222.4,
    'Asterix': 210.0,
    'BankHeist': 14.2,
    'BattleZone': 2360.0,
    'Boxing': 0.1,
    'Breakout': 1.7,
    'ChopperCommand': 811.0,
    'CrazyClimber': 10780.5,
    'DemonAttack': 152.1,
    'Freeway': 0.0,
    'Frostbite': 65.2,
    'Gopher': 257.6,
    'Hero': 1027.0,
    'Jamesbond': 29.0,
    'Kangaroo': 52.0,
    'Krull': 1598.0,
    'KungFuMaster': 258.5,
    'MsPacman': 307.3,
    'Pong': -20.7,
    'PrivateEye': 24.9,
    'Qbert': 163.9,
    'RoadRunner': 11.5,
    'Seaquest': 68.4,
    'UpNDown': 533.4
}
RANDOM_SCORES = {key.lower(): RANDOM_SCORES[key] for key in RANDOM_SCORES}

HUMAN_SCORES = {
    'Alien': 7127.7,
    'Amidar': 1719.5,
    'Assault': 742.0,
    'Asterix': 8503.3,
    'BankHeist': 753.1,
    'BattleZone': 37187.5,
    'Boxing': 12.1,
    'Breakout': 30.5,
    'ChopperCommand': 7387.8,
    'CrazyClimber': 35829.4,
    'DemonAttack': 1971.0,
    'Freeway': 29.6,
    'Frostbite': 4334.7,
    'Gopher': 2412.5,
    'Hero': 30826.4,
    'Jamesbond': 302.8,
    'Kangaroo': 3035.0,
    'Krull': 2665.5,
    'KungFuMaster': 22736.3,
    'MsPacman': 6951.6,
    'Pong': 14.6,
    'PrivateEye': 69571.3,
    'Qbert': 13455.0,
    'RoadRunner': 7845.0,
    'Seaquest': 42054.7,
    'UpNDown': 11693.2
}
HUMAN_SCORES = {key.lower(): HUMAN_SCORES[key] for key in HUMAN_SCORES}

for name in results:
    print(f'experiment: {name}')
    num_seeds_per_env = None
    envs = sorted(results[name].keys())
    mean_per_env = {}
    human_norm = {}
    atari = False
    for env in envs:
        # consistency assertions and missing data checks
        if num_seeds_per_env is None:
            num_seeds_per_env = len(results[name][env])
            num_evals = results[name][env][1]["eval"].shape[0]
            num_frames = results[name][env][1]["eval"]["frame"].iloc[-1]
            print(f'{num_seeds_per_env} seeds, {num_frames} frames:')
        asrt = len(results[name][env]) == num_seeds_per_env
        msg = f'missing seeds for {env}. counted {len(results[name][env])} instead of {num_seeds_per_env}'
        # assert asrt, msg
        if not asrt:
            print(msg)
        for seed in results[name][env]:
            # asrt = results[env][seed]["eval"].shape[0] == num_evals
            # msg = f'missing evals for {env}. counted {results[env][seed]["eval"].shape[0]} instead of {num_evals} at seed {seed}'
            # # assert asrt, msg
            # if not asrt:
            #     print(msg)
            asrt = results[name][env][seed]["eval"]["frame"].iloc[-1] == num_frames
            msg = f'missing frames for {env}. counted {results[name][env][seed]["eval"]["frame"].iloc[-1]} instead of {num_frames} at seed {seed}'
            # assert asrt, msg
            if not asrt:
                print(msg)

        # tallying up results
        reward = [results[name][env][seed]["eval"]["episode_reward"].iloc[-1] for seed in results[name][env]]
        mean_reward = round(np.mean(reward), 1)
        std_reward = round(np.std(reward), 1)
        print(f'{env}: {mean_reward} ± {std_reward}')
        mean_per_env[env] = mean_reward
        if env in RANDOM_SCORES:
            atari = True
            human_norm[env] = (mean_per_env[env] - RANDOM_SCORES[env])/(HUMAN_SCORES[env] - RANDOM_SCORES[env])
    if atari:
        mean_human_norm = np.mean([human_norm[key] for key in human_norm])
        med_human_norm = np.median([human_norm[key] for key in human_norm])
    mean = np.mean([mean_per_env[key] for key in mean_per_env])
    median = np.median([mean_per_env[key] for key in mean_per_env])
    if atari:
        print(f'Mean Human Normalized: {mean_human_norm}')
        print(f'Median Human Normalized: {med_human_norm}')
    print(f'Mean: {mean}')
    print(f'Median: {median}')
    print()

# import pandas as pd
# import numpy as np
# import scipy as sp
# import glob
# import os
# from omegaconf import OmegaConf
#
# import matplotlib.pyplot as plt
# from matplotlib import cm
# plt.style.use('bmh')
# import seaborn as sns
# plt.rcParams['figure.dpi'] = 400
# plt.rcParams['font.size'] = 8
# plt.rcParams['legend.fontsize'] = 7
# plt.rcParams['legend.loc'] = 'lower right'
#
#
# def plot(df, key='episode_reward'):
#     envs = np.sort(df.env.unique())
#     ncol = 3
#     assert envs.shape[0] % ncol == 0
#     nrow = envs.shape[0] // ncol
#     fig, axs = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
#
#     for idx, env in enumerate(envs):
#         data = df[df['env'] == env]
#         row = idx // ncol
#         col = idx % ncol
#         ax = axs[row, col]
#         hue_order = np.sort(data.Agent.unique())
#         sns.lineplot(x='step', y=key, data=data, ci='sd', hue='Agent', hue_order=hue_order, ax=ax)
#         ax.set_title(f'{env}')
#     plt.tight_layout()
#     plt.show()
#
# # df = pd.read_csv('sac.csv')
# # plot(df)