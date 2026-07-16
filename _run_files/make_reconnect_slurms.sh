#!/bin/bash
# Generates the reconnect .slurm run files (rdur_noage, rdur, sc_rdur, sc_rnodur,
# sc_rsbmdur, sc_rsbm, rnodur_noage, rnodur, rsbmdur, rsbm), one per r_*.py script,
# mirroring the existing p*.slurm files for the r_*.py scripts (reconnect equivalents
# of p_*.py), then submits each one with sbatch. Run from within _run_files/.

set -euo pipefail

# slurm filename -> python script name
declare -A SLURM_TO_PY=(
    [rdur_noage.slurm]="r_durdist_sims_noage.py"
    [rdur.slurm]="r_durdist_sims.py"
    [sc_rdur.slurm]="r_durdist_sims_sc.py"
    [sc_rnodur.slurm]="r_nodur_sc.py"
    [sc_rsbmdur.slurm]="r_sbmdur_sc.py"
    [sc_rsbm.slurm]="r_sbm_sc.py"
    [rnodur_noage.slurm]="r_sims_nodur_noage.py"
    [rnodur.slurm]="r_sims_nodur.py"
    [rsbmdur.slurm]="r_sims_sbm_dur.py"
    [rsbm.slurm]="r_sims_sbm.py"
)

for slurm_file in "${!SLURM_TO_PY[@]}"; do
    py_file="${SLURM_TO_PY[$slurm_file]}"
    cat > "$slurm_file" <<EOF
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

export OMP_NUM_THREADS=\$SLURM_CPUS_PER_TASK

python $py_file
EOF
    echo "wrote $slurm_file -> python $py_file"
    sbatch "$slurm_file"
done
