#!/usr/bin/env python
import sphere
import shutil

import seaborn as sns
#sns.set(style='ticks', palette='Set2')
#sns.set(style='ticks', palette='colorblind')
sns.set(style='white', palette='Set2')
sns.despine() # remove chartjunk


sim = sphere.sim(fluid=True)

def plotpressures(sid):
    sim.id(sid)
    outformat = 'pdf'
    sim.visualize('fluid-pressure',outformat=outformat, figsize=[3.74, 3.14])
    filename = sid + '-fluid-pressure.' + outformat
    shutil.copyfile(filename, '/home/adc/articles/own/2/graphics/' + filename)

sids = [
'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-06-velfac=1.0-shear',
'halfshear-darcy-sigma0=20000.0-k_c=3.5e-14-mu=1.797e-06-velfac=1.0-shear',
'halfshear-darcy-sigma0=20000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear',

'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-07-velfac=1.0-shear',
'halfshear-darcy-sigma0=20000.0-k_c=3.5e-15-mu=1.797e-08-velfac=1.0-shear',


'halfshear-darcy-sigma0=80000.0-k_c=3.5e-15-mu=1.797e-06-velfac=1.0-shear',
'halfshear-darcy-sigma0=80000.0-k_c=3.5e-14-mu=1.797e-06-velfac=1.0-shear',
'halfshear-darcy-sigma0=80000.0-k_c=3.5e-13-mu=1.797e-06-velfac=1.0-shear',

'halfshear-darcy-sigma0=80000.0-k_c=3.5e-15-mu=1.797e-07-velfac=1.0-shear',
'halfshear-darcy-sigma0=80000.0-k_c=3.5e-15-mu=1.797e-08-velfac=1.0-shear'#,
]

for sid in sids:
    plotpressures(sid)
