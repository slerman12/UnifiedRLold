import glob
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

root = "./exp_local"

exps = []
for exps_by_date in sorted(glob.glob(root + "/*/"))[-2:]:
# for exps_by_date in sorted(glob.glob(root + "/*/")):
    exps = exps + glob.glob(exps_by_date + "/*/")
exps = sorted(exps)

results = {}

for exp in exps:
    if "seed=" not in exp or "experiment=" not in exp:
        continue
    name = exp.split("experiment=", 1)[1].split(",", 1)[0]
    name = "bla"
    seed = int(exp.split("seed=", 1)[1].split(",", 1)[0])
    env = glob.glob(exp + "/*")[0].split("/")[-1]

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

for name in results:
    print(f'experiment: {name}')
    num_seeds_per_env = None
    for env in results[name]:
        # consistency assertions and missing data checks
        if num_seeds_per_env is None:
            num_seeds_per_env = len(results[name][env])
            num_evals = results[name][env][1]["eval"].shape[0]
            num_frames = results[name][env][1]["eval"]["frame"].iloc[-1]
            print(f'{num_seeds_per_env} seeds, {num_frames} frames:\n')
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
