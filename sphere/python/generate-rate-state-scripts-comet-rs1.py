#!/usr/bin/env python

# rs1: relative to rs0 set top wall mass to equal total particle mass, disable
# wall viscosity

# Account and cluster information
# https://portal.xsede.org/sdsc-comet
# https://www.sdsc.edu/support/user_guides/comet.html
account = 'csd492'  # from `show_accounts`
jobname_prefix = 'rs1-'
walltime = '2-0'   # hours:minutes:seconds or days-hours
partition = 'gpu-shared'
no_gpus = 1
no_nodes = 1
ntasks_per_node = 1
folder = '~/code/sphere/python'


# Simulation parameter values
effective_stresses = [10e3, 20e3, 100e3, 200e3, 1000e3, 2000e3]
velfacs = [0.1, 1.0, 10.0]
mu_s_vals = [0.5]
mu_d_vals = [0.5]


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
def generate_simulation_script(jobname, effective_stress, velfac, mu_s, mu_d):

    script = '''#!/usr/bin/env python
import sphere
import numpy

# load consolidated granular assemblage
sim = sphere.sim(fluid=False)
cons_jobname = 'cons-1e4-' + '{{}}Pa'.format({effective_stress})
sim = sphere.sim(cons_jobname, fluid=False)
sim.readlast()
sim.id('{jobname}')

sim.checkerboardColors(nx=6, ny=6, nz=6)
sim.cleanup()
sim.adjustUpperWall()
sim.zeroKinematics()

sim.shear(1.0/20.0 * {velfac})

sim.setStiffnessNormal(1.16e7)
sim.setStiffnessTangential(1.16e7)
sim.setStaticFriction({mu_s})
sim.setDynamicFriction({mu_d})
sim.setDampingNormal(0.0)
sim.setDampingTangential(0.0)

sim.w_sigma0[0] = {effective_stress}
sim.w_m[0] = sim.totalMass()
sim.gamma_wn[0] = 0.0

sim.initTemporal(total = 20.0, file_dt = 0.01, epsilon=0.07)

I = numpy.nonzero(sim.fixvel > 0)
sim.fixvel[I] = 8.0 # step-wise velocity change when fixvel in ]5.0; 10.0[

sim.run(dry=True)
sim.run(device=0)
sim.writeVTKall()
sim.visualize('shear')
'''.format(jobname=jobname,
           effective_stress=effective_stress,
           velfac=velfac,
           mu_s=mu_s,
           mu_d=mu_d)

    with open(jobname + '.py', 'w') as file:
        file.write(script)


# Generate scripts
for effective_stress in effective_stresses:
    for velfac in velfacs:
        for mu_s in mu_s_vals:
            for mu_d in mu_s_vals:

                jobname = jobname_prefix + '{}Pa-v={}-mu_s={}-mu_d={}'.format(
                    effective_stress,
                    velfac,
                    mu_s,
                    mu_d)

                print(jobname)

                # Generate scripts for slurm, submit with `sbatch <script>`
                generate_slurm_script(jobname)

                generate_slurm_continue_script(jobname)

                generate_simulation_script(jobname,
                                           effective_stress,
                                           velfac,
                                           mu_s,
                                           mu_d)
