#!/bin/sh
#PBS -N permeability-c_grad_p=0.5
#PBS -l nodes=1:ppn=3
#PBS -l walltime=1920:00:00
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
python permeability-starter.py 0 1.0 0.5 1.0e3 2.0e3 &
python permeability-starter.py 1 1.0 0.5 4.0e3 1.0e4 &
python permeability-starter.py 2 1.0 0.5 2.0e4 4.0e4 &
wait

#cp $WORKDIR/output/* $ORIGDIR/output/

echo "End at `date`"
