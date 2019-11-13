#!/usr/bin/env python

# Account and cluster information
# https://portal.xsede.org/sdsc-comet
# https://www.sdsc.edu/support/user_guides/comet.html
account = 'csd492'  # from `show_accounts`
jobname_prefix = 'cons-1e4-'
walltime = '2-0'   # hours:minutes:seconds or days-hours
partition = 'gpu-shared'
no_gpus = 1
no_nodes = 1
ntasks_per_node = 1
folder = '~/code/sphere/python'


# Simulation parameter values
effective_stresses = [10e3, 20e3, 100e3, 200e3, 1000e3, 2000e3]


# Script generating functions

def generate_slurm_script(jobname):

    script = '''#!/bin/bash
#SBATCH -A {account}
#SBATCH --job-name="{jobname}"
#SBATCH --output="{jobname}.%j.%N.out"
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{no_gpus}
#SBATCH --nodes={no_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --export=ALL

echo Job start `whoami`@`hostname`, `date`
module load cmake
module load cuda/7.0
module load python
module load scipy

cd {folder}
python ./{jobname}.py

echo Job end `whoami`@`hostname`, `date`
'''.format(account=account,
           jobname=jobname,
           walltime=walltime,
           partition=partition,
           no_gpus=no_gpus,
           no_nodes=no_nodes,
           ntasks_per_node=ntasks_per_node,
           folder=folder)
    with open(jobname + '.sh', 'w') as file:
        file.write(script)


def generate_slurm_continue_script(jobname):

    script = '''#!/bin/bash
#SBATCH -A {account}
#SBATCH --job-name="{jobname}"
#SBATCH --output="{jobname}.%j.%N.out"
#SBATCH --time={walltime}
#SBATCH --partition={partition}
#SBATCH --gres=gpu:{no_gpus}
#SBATCH --nodes={no_nodes}
#SBATCH --ntasks-per-node={ntasks_per_node}
#SBATCH --export=ALL

echo Job start `whoami`@`hostname`, `date`
module load cmake
module load cuda/7.0
module load python
module load scipy

cd {folder}
python ./continue_sim.py {jobname} 0

echo Job end `whoami`@`hostname`, `date`
'''.format(account=account,
           jobname=jobname,
           walltime=walltime,
           partition=partition,
           no_gpus=no_gpus,
           no_nodes=no_nodes,
           ntasks_per_node=ntasks_per_node,
           folder=folder)
    with open(jobname + '-cont.sh', 'w') as file:
        file.write(script)


# Generate scripts for sphere
def generate_simulation_script(jobname, effective_stress):

    script = '''#!/usr/bin/env python
import sphere

cons = sphere.sim('init-1e4')
cons.readlast()
cons.id('{jobname}')

cons.periodicBoundariesXY()

cons.setStiffnessNormal(1.16e7)
cons.setStiffnessTangential(1.16e7)
cons.setStaticFriction(0.5)
cons.setDynamicFriction(0.5)
cons.setDampingNormal(0.0)
cons.setDampingTangential(0.0)

cons.consolidate(normal_stress={effective_stress})
cons.initTemporal(total=3.0, epsilon=0.07)

cons.run(dry=True)
cons.run(device=0)

cons.visualize('energy')
cons.visualize('walls')
'''.format(jobname=jobname,
           effective_stress=effective_stress)

    with open(jobname + '.py', 'w') as file:
        file.write(script)


# Generate scripts
for effective_stress in effective_stresses:

    jobname = jobname_prefix + '{}Pa'.format(effective_stress)

    print(jobname)

    # Generate scripts for slurm, submit with `sbatch <script>`
    generate_slurm_script(jobname)

    generate_slurm_continue_script(jobname)

    generate_simulation_script(jobname, effective_stress)
