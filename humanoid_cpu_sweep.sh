#!/bin/sh
#module load python3/3.8.3
seed=$1
experiment=$2
echo "seed $seed"
for game in "humanoid_walk" "humanoid_run" "humanoid_stand"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
  echo "queueing seed $seed task $game..."
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name dmc task=dmc/$game seed=$seed experiment=$experiment agent._target_=agents.rQdiaAgent" --num-cpus 10 --mem 120
#	sleep 1
done
