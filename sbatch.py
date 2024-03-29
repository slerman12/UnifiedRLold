import argparse
import subprocess
from subprocess import check_output


# Arguments
parser = argparse.ArgumentParser()
parser.add_argument('--name', type=str, default="name")
parser.add_argument('--cpu', action='store_true', default=False,
                    help='uses CPUs')
parser.add_argument('--lab', action='store_true', default=False,
                    help='uses csxu')
parser.add_argument('--bigger-gpu', action='store_true', default=False,
                    help='uses K80 GPU')
parser.add_argument('--biggest-gpu', action='store_true', default=False,
                    help='uses V100 GPU')
parser.add_argument('--very-biggest-gpu', action='store_true', default=False,
                    help='uses A100 GPU')
parser.add_argument('--mem', type=int, default=25)
parser.add_argument('--num-cpus', type=int, default=4)
parser.add_argument('--file', type=str, default="run.py")
parser.add_argument('--params', type=str, default="task=atari/ms_pacman")
args = parser.parse_args()


def slurm_script_generalized():
    return r"""#!/bin/bash
#SBATCH {}
{}
{}
#SBATCH -t {} -o ./{}.log -J {}
#SBATCH --mem={}gb 
{}

source /scratch/slerman/miniconda/bin/activate agi
python3 {} {}
""".format("-c {}".format(args.num_cpus) if args.cpu else "-p gpu -c {}".format(args.num_cpus),
           "" if args.cpu else "#SBATCH --gres=gpu",
           "#SBATCH -p csxu -A cxu22_lab" if args.cpu and args.lab else "#SBATCH -p csxu -A cxu22_lab --gres=gpu" if args.lab else "",
           "15-00:00:00" if args.lab else "5-00:00:00",
           args.name, args.name,
           args.mem,
           "#SBATCH -C K80" if args.bigger_gpu else "#SBATCH -C V100" if args.biggest_gpu else "#SBATCH -C A100" if args.very_biggest_gpu else "",
           args.file, args.params)


with open("sbatch_script", "w") as file:
    file.write(slurm_script_generalized())
success = "error"
while "error" in success:
    try:
        success = str(subprocess.check_output(['sbatch {}'.format("sbatch_script")], shell=True))
        print(success[2:][:-3])
        if "error" in success:
            print("Errored... trying again")
    except:
        success = "error"
        if "error" in success:
            print("Errored... trying again")
print("Success!")
# out = check_output(["ntpq", "-p"])
