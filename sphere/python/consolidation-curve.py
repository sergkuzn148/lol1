#!/usr/bin/env python
import matplotlib
matplotlib.use('Agg')
matplotlib.rcParams.update({'font.size': 18, 'font.family': 'serif'})
import os
import shutil

import sphere
import numpy
import matplotlib.pyplot as plt

c_phi = 1.0
#c_grad_p_list = [1.0, 0.1, 0.01, 0.001]
#c_grad_p_list = [1.0, 0.1, 0.01]
c_grad_p_list = [1.0, 0.1]
#c_grad_p_list = [1.0]
sigma0 = 10.0e3
#sigma0 = 5.0e3

t = [[], []]
H = [[], []]
#t = [[], [], []]
#H = [[], [], []]
#t = [[], [], [], []]
#H = [[], [], [], []]

c = 0
for c_grad_p in c_grad_p_list:

    sid = 'cons-sigma0=' + str(sigma0) + '-c_phi=' + \
                     str(c_phi) + '-c_grad_p=' + str(c_grad_p)
    if c_grad_p != 1.0:
        sid += '-tall'

    if os.path.isfile('../output/' + sid + '.status.dat'):
        sim = sphere.sim(sid, fluid=True)
        t[c] = numpy.ones(sim.status()-1)
        H[c] = numpy.ones(sim.status()-1)

        #sim.visualize('walls')
        #sim.writeVTKall()

        #sim.plotLoadCurve()
        #sim.readfirst(verbose=True)
        #for i in numpy.arange(1, sim.status()+1):
        for i in numpy.arange(1, sim.status()):
            sim.readstep(i, verbose=False)
            t[c][i-1] = sim.time_current[0]
            H[c][i-1] = sim.w_x[0]

        '''
        # find consolidation parameters
        self.H0 = H[0]
        self.H100 = H[-1]
        self.H50 = (self.H0 + self.H100)/2.0
        T50 = 0.197 # case I

        # find the time where 50% of the consolidation (H50) has happened by
        # linear interpolation. The values in H are expected to be
        # monotonically decreasing. See Numerical Recipies p. 115
        i_lower = 0
        i_upper = self.status()-1
        while (i_upper - i_lower > 1):
            i_midpoint = int((i_upper + i_lower)/2)
            if (self.H50 < H[i_midpoint]):
                i_lower = i_midpoint
            else:
                i_upper = i_midpoint
        self.t50 = t[i_lower] + (t[i_upper] - t[i_lower]) * \
                (self.H50 - H[i_lower])/(H[i_upper] - H[i_lower])

        self.c_v = T50*self.H50**2.0/(self.t50)
        if self.fluid == True:
            e = numpy.mean(sb.phi[:,:,3:-8]) # ignore boundaries
        else:
            e = sb.voidRatio()
        '''

        H[c] -= H[c][0]

    c += 1

# Normalize the thickness change
#min_H = 0.0
#for c in range(len(c_grad_p_list)):
    #min_H_c = numpy.min(H[c])
    #if min_H_c < min_H:
        #min_H = min_H_c

plt.xlabel('Time [s]')
#plt.ylabel('Normalized thickness change [-]')
plt.ylabel('Thickness change [m]')
#plt.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
#for c in range(len(c_grad_p_list)):
    #H[c] /= -min_H_c
plt.semilogx(t[0], H[0], '-k', label='$c$ = %.2f' % (c_grad_p_list[0]))
plt.semilogx(t[1], H[1], '--k', label='$c$ = %.2f' % (c_grad_p_list[1]))
#plt.grid()

plt.legend(loc='best', prop={'size':18}, fancybox=True, framealpha=0.5)
plt.tight_layout()
filename = 'cons-curves.pdf'
plt.savefig(filename)
#print(os.getcwd() + '/' + filename)
shutil.copyfile(filename, '/home/adc/articles/own/2-org/' + filename)
print(filename)
