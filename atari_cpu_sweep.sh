#!/bin/sh
module load python3/3.8.3
seed=$1
for game in "alien" "amidar" "assault" "asterix" "bank_heist" "battle_zone" "boxing" "breakout" "chopper_command" "crazy_climber" "demon_attack" "freeway" "frostbite" "gopher" "hero" "jamesbond" "kangaroo" "krull" "kung_fu_master" "ms_pacman" "pong" "private_eye" "qbert" "road_runner" "seaquest" "up_n_down"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name atari task=atari/$game seed=$seed experiment=$game$seed"
	sleep 1
done
