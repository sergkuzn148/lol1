#!/usr/bin/env python
import sphere
import numpy

'''
Print a table of peak and ultimate shear strengths of the material
'''

baseid = 'halfshear-sigma0=20000.0'




def print_strengths(sid, fluid=False, c=0.0):
    sim = sphere.sim(sid, fluid=fluid)

    sim.readfirst(verbose=False)
    sim.visualize('shear')

    friction = sim.tau[1:]/sim.sigma_eff[1:]
    tau_peak = numpy.max(friction)
    tau_ultimate = numpy.average(friction[-500:-1])

    if fluid:
        print('%.2f \t %.2f \t %.2f' % (c, tau_peak, tau_ultimate))
    else:
        print('dry \t %.2f \t %.2f' % (tau_peak, tau_ultimate))

    return friction




# print header
print('$c$ [-] \t \\tau/\\sigma\' (peak) [-] \t \\tau/\\sigma\' (ultimate) [-]')
f = print_strengths(baseid + '-shear', fluid=False)
f = print_strengths(baseid + '-c=1.0-shear', fluid=True, c=1.0)
f = print_strengths(baseid + '-c=0.1-shear', fluid=True, c=0.1)



