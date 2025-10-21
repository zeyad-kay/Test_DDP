#!/bin/bash
#SBATCH --account=rrg-ebrahimi_gpu
#SBATCH --nodes 1
#SBATCH --gpus=h100:2
#SBATCH --tasks-per-node=2
#SBATCH --cpus-per-task=2
#SBATCH --mem=2G
#SBATCH --time=00:30:00
#SBATCH --output=two_gpus_single_node.out

srun -N $SLURM_NNODES -n $SLURM_NNODES bash << EOF
virtualenv --no-download $SLURM_TMPDIR/env
source $SLURM_TMPDIR/env/bin/activate
pip install torch --no-index
EOF

export TORCH_NCCL_ASYNC_HANDLING=1
export MASTER_ADDR=$(hostname)
export NCCL_DEBUG=INFO

echo "r$SLURM_NODEID master: $MASTER_ADDR"
echo "r$SLURM_NODEID Launching python script"

source $SLURM_TMPDIR/env/bin/activate

srun python test_ddp.py
