#!/bin/sh
#module load python3/3.8.3
seed=$1
experiment=$2
echo "seed $seed"
for game in "alien" "amidar" "assault" "asterix" "bankheist" "battlezone" "boxing" "breakout" "choppercommand" "crazyclimber" "demonattack" "freeway" "frostbite" "gopher" "hero" "jamesbond" "kangaroo" "krull" "kungfumaster" "mspacman" "pong" "privateeye" "qbert" "roadrunner" "seaquest" "upndown"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
  echo "queueing seed $seed game $game..."
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name atari task=atari/$game seed=$seed experiment=$experiment"
	sleep 3
done
