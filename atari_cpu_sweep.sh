#!/bin/sh
#module load python3/3.8.3
seed=$1
for game in "alien" "amidar" "assault" "asterix" "bankheist" "battlezone" "boxing" "breakout" "choppercommand" "crazyclimber" "demonattack" "freeway" "frostbite" "gopher" "hero" "jamesbond" "kangaroo" "krull" "kungfumaster" "mspacman" "pong" "privateeye" "qbert" "roadrunner" "seaquest" "upndown"
#for game in "alien" "amidar" "assault" "asterix" "bank_heist" "boxing" "breakout" "chopper_command"
#for game in "breakout"
do
  {
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name atari task=atari/$game seed=$seed experiment=$game$seed"
	} || {
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name atari task=atari/$game seed=$seed experiment=$game$seed"
	echo "saved"
	} || {
	python3 sbatch.py --bigger-gpu --name $game$seed --params "--config-name atari task=atari/$game seed=$seed experiment=$game$seed"
	echo "saved backup"
	}
	sleep 7
done
