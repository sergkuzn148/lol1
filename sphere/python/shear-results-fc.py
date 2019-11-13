#!/usr/bin/env python
import sphere
import numpy
import subprocess

sid = 'shear-sigma0=20000.0-hw'
imgformat = 'png'
sim = sphere.sim(sid, fluid=False)
subprocess.call('mkdir -p ' + sid + '-fc', shell=True)

d = 0
for i in numpy.arange(210,300):
#for i in numpy.arange(240,260):
    print("File: %d, output: %s-fc/%05d.png" % (i, sid, d))
    sim.readstep(i, verbose=False)
    #sim.forcechains(lc = 3.0e1, uc=1.0e2)
    sim.forcechains(lc = 30.0, uc=1.0e2)
    subprocess.call('mv shear-sigma0=20000-0-hw-fc.' + imgformat \
            + ' ' + sid + '-fc/%05d.png' % (d), shell=True)
    d += 1

subprocess.call('cd ' + sid + '-fc && sh ./make_video.sh', shell=True)

