#!/bin/bash
#SBATCH --nodes 10
#SBATCH --ntasks-per-node=48
#SBATCH -p sequana_cpu_bigmem
#SBATCH --job-name=Kmeans
#SBATCH --time=00:45:00
#SBATCH -e slurm-%j.err
#SBATCH -o slurm-%j.out

module load anaconda3/2024.02_sequana
eval "$(conda shell.bash hook)"

CONDA_ENV="/scratch/cenapadrjsd/reiglan.lourenco/RNA-seq/conda_envs/parsl_env/"
conda activate ${CONDA_ENV}

CDIR="/scratch/cenapadrjsd/reiglan.lourenco/K-means_Parsl/"
cd ${CDIR}

python main2.py --onslurm
