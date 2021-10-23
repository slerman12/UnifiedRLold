#!/bin/sh
#module load python3/3.8.3
numseeds=$1
let sleeptime=$numseeds * 7 + 7
for i in $(seq 1 $numseeds)
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
	bash atari_cpu_sweep.sh $i
	sleep $sleeptime
done
