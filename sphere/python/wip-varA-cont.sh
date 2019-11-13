#!/bin/sh
#PBS -N wip-varA
#PBS -l nodes=1:ppn=3
#PBS -l walltime=48:00:00
#PBS -q qfermi
#PBS -M adc@geo.au.dk
#PBS -m abe

# Grendel CUDA
source /com/gcc/4.6.4/load.sh
CUDAPATH=/com/cuda/5.5.22
export PATH=$HOME/bin:$PATH
export PATH=$CUDAPATH/bin:$PATH
export LD_LIBRARY_PATH=$CUDAPATH/lib64:$CUDAPATH/lib:$LD_LIBRARY_PATH

# Manually installed Python modules
export PYTHONPATH=$HOME/.local/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/lib64/python:$PYTHONPATH

# Manually installed Python
#export PATH=/home/adc/.local/bin:$PATH

# Shared Python2.7
PYTHON=/com/python/2.7.6
export PYTHONPATH=$PYTHON/lib:$PYTHONPATH
export PATH=$PYTHON/bin:$PATH

echo "`whoami`@`hostname`"
echo "Start at `date`"

ORIGDIR=/home/adc/code/sphere
#WORKDIR=/scratch/$PBS_JOBID
WORKDIR=$ORIGDIR

#cp -r $ORIGDIR/* $WORKDIR

cd $WORKDIR
nvidia-smi
rm CMakeCache.txt
cmake . && make
cd python
python continue_sim.py halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4125.0-f=0.2 1 0 4 &
python continue_sim.py halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4250.0-f=0.2 1 1 4 &
python continue_sim.py halfshear-darcy-sigma0=10000.0-k_c=2e-16-mu=2.08e-07-ss=2000.0-A=4375.0-f=0.2 1 2 4 &
wait

#cp $WORKDIR/output/* $ORIGDIR/output/

echo "End at `date`"
