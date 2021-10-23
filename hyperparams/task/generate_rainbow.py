RAINBOW_ENVS = [
    'Alien', 'Amidar', 'Assault', 'Asterix', 'BankHeist', 'BattleZone',
    'Boxing', 'Breakout', 'ChopperCommand', 'CrazyClimber', 'DemonAttack',
    'Freeway', 'Frostbite', 'Gopher', 'Hero', 'Jamesbond', 'Kangaroo', 'Krull',
    'KungFuMaster', 'MsPacman', 'Pong', 'PrivateEye', 'Qbert', 'RoadRunner',
    'Seaquest', 'UpNDown'
]

out = ""
for env in RAINBOW_ENVS:
    f = open(f"./atari/{env.lower()}.yaml", "w")
    f.write(r"""num_train_frames:  100001
stddev_schedule: 'linear(1.0,0.1,500000)'
task_name: {}""".format(env))
    f.close()
    out += ' "' + env.lower() + '"'
print(out)
