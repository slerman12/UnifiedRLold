import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = "./exp_local"
latest_exps = sorted(glob.glob(root + "/*/"))[-1]

exps = sorted(glob.glob(latest_exps + "/*/"))

results = {}

for exp in exps:
    if "seed=" not in exp:
        continue
    seed = int(exp.split("seed=", 1)[1].split(",", 1)[0])
    env = glob.glob(exp + "/*")[0].split("/")[-1]

    try:
        train = pd.read_csv(exp + f'{env}/train.csv')
        eval = pd.read_csv(exp + f'{env}/eval.csv')
    except:
        continue

    if env not in results:
        results[env] = {}

    # dict of envs, each has dict of seeds, each has train csv and eval csv
    results[env].update({seed: {"train": train, "eval": eval}})

    # train.plot(x="frame", y="episode_reward", legend=None)
    # plt.xlabel('Frame')
    # plt.ylabel('Reward')
    # plt.title(env.capitalize())
    # plt.show()
    # break


    # with open(file) as f:
    #     lines = f.readlines()

num_seeds_per_env = None
for env in results:
    # consistency assertions and missing data checks
    if num_seeds_per_env is None:
        num_seeds_per_env = len(results[env])
        num_evals_per_env = results[env][1]["eval"].shape[0]
        frames = results[env][1]["eval"]["frame"].iloc[-1]
        print(f'{num_seeds_per_env} seeds, {frames} frames:\n')
    asrt = len(results[env]) == num_seeds_per_env
    msg = f'missing seeds for {env}. counted {len(results[env])} instead of {num_seeds_per_env}'
    # assert asrt, msg
    if not asrt:
        print(msg)
    for seed in results[env]:
        asrt = results[env][seed]["eval"].shape[0] == num_evals_per_env
        msg = f'missing evals for {env}. counted {results[env][seed]["eval"].shape[0]} instead of {num_evals_per_env} at seed {seed}'
        # assert asrt, msg
        if not asrt:
            print(msg)

    # tallying up results
    reward = [results[env][seed]["eval"]["episode_reward"].iloc[-1] for seed in results[env]]
    mean_reward = round(np.mean(reward), 1)
    std_reward = round(np.std(reward), 1)
    print(f'{env}: {mean_reward} ± {std_reward}')

