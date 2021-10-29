#!/bin/sh
#module load python3/3.8.3
numseeds=$1
experiment=$2
echo "experiment $experiment"
for i in $(seq 2 $numseeds)
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
	bash humanoid_cpu_sweep.sh $i $experiment
done
