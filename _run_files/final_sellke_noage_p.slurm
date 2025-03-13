#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=48
#SBATCH --mem-per-cpu=3700

#SBATCH --time=48:00:00


module purge
module restore p1
source ~/p1/bin/activate
maturin develop --release

export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

python _final_sellke_noage_p.py