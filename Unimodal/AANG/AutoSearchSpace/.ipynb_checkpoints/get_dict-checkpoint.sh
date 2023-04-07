#!/bin/bash
#SBATCH -p gpu_highmem
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem 56GB
#SBATCH --time 2-00:00:00
#SBATCH --mail-type=ALL
#SBATCH --mail-user=your_email@gmail.com # TODO
#SBATCH --chdir=/work/sakter/AANG/AutoSearchSpace
#SBATCH --output=/work/sakter/AANG/logs/dict_chem_expert.out # TODO
#SBATCH --error=/work/sakter/AANG/logs/dict_chem_expert.err # TODO

eval "$(conda shell.bash hook)"
conda activate aang

python get_dict.py --dtype chem --method expert