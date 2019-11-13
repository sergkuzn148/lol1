#!/usr/bin/env python
import sys
import sphere
import numpy
import matplotlib.pyplot as plt

class PermeabilityCalc:
    ''' Darcy's law: Q = -k*A/mu * dP '''

    def __init__(self, sid, plot_evolution=True, print_results=True,
            verbose=True):
        self.sid = sid
        self.readfile(verbose)
        self.findPermeability()
        self.findConductivity()
        self.findMeanPorosity()
        if print_results:
            self.printResults()
        if plot_evolution:
            self.plotEvolution()

    def readfile(self, verbose=True):
        self.sim = sphere.sim(self.sid, fluid=True)
        self.sim.readlast(verbose=verbose)

    def findPermeability(self):
        self.findCellSpacing()
        self.findCrossSectionalArea()
        self.findCrossSectionalFlux()
        self.findPressureGradient()
        self.k = -self.Q*self.sim.mu/(self.A*self.dPdL) # m^2

    def findConductivity(self):
        # hydraulic conductivity
        self.findCellSpacing()
        self.findCrossSectionalArea()
        self.findCrossSectionalFlux()
        self.findPressureGradient()
        #self.K = self.k/self.sim.mu # m/s
        self.K = -self.Q * self.dL / (self.A * self.dP)

    def conductivity(self):
        return self.K[2]

    def c_grad_p(self):
        return self.sim.c_grad_p[0]

    def findMeanPorosity(self):
        ''' calculate mean porosity in cells beneath the top wall '''

        if (self.sim.nw > 0):
            wall_iz = int(self.sim.w_x[0]/self.dx[2])
            self.phi_bar = numpy.mean(self.sim.phi[:,:,0:wall_iz-1])
        else:
            self.phi_bar = numpy.mean(self.sim.phi[:,:,0:-3])
                

    def findCrossSectionalArea(self):
        ''' Cross sectional area normal to each axis '''
        self.A = numpy.array([
            self.sim.L[1]*self.sim.L[2],
            self.sim.L[0]*self.sim.L[2],
            self.sim.L[0]*self.sim.L[1]])

    def findCellSpacing(self):
        self.dx = numpy.array([
            self.sim.L[0]/self.sim.num[0],
            self.sim.L[1]/self.sim.num[1],
            self.sim.L[2]/self.sim.num[2]])

    def findCrossSectionalFlux(self):
        ''' Flux along each axis, measured at the outer boundaries '''
        #self.Q = numpy.array([
            #numpy.mean(self.sim.v_f[-1,:,:]),
            #numpy.mean(self.sim.v_f[:,-1,:]),
            #numpy.mean(self.sim.v_f[:,:,-1])])*self.A

        self.Q = numpy.zeros(3)

        self.A_cell = numpy.array([
            self.dx[1]*self.dx[2],
            self.dx[0]*self.dx[2],
            self.dx[0]*self.dx[1]])

        # x axis (0)
        for y in numpy.arange(self.sim.num[1]):
            for z in numpy.arange(self.sim.num[2]):
                self.Q[0] += self.sim.v_f[-1,y,z,0] * self.A_cell[0]

        # y axis (1)
        for x in numpy.arange(self.sim.num[0]):
            for z in numpy.arange(self.sim.num[2]):
                self.Q[1] += self.sim.v_f[x,-1,z,1] * self.A_cell[1]

        # z axis (2)
        for x in numpy.arange(self.sim.num[0]):
            for y in numpy.arange(self.sim.num[1]):
                self.Q[2] += self.sim.v_f[x,y,-1,2] * self.A_cell[2]

    def findPressureGradient(self):
        ''' Determine pressure gradient by finite differencing the
        mean values at the outer boundaries '''
        self.dP = numpy.array([
            numpy.mean(self.sim.p_f[-1,:,:]) - numpy.mean(self.sim.p_f[0,:,:]),
            numpy.mean(self.sim.p_f[:,-1,:]) - numpy.mean(self.sim.p_f[:,0,:]),
            numpy.mean(self.sim.p_f[:,:,-1]) - numpy.mean(self.sim.p_f[:,:,0])
            ])
        self.dL = self.sim.L
        self.dPdL = self.dP/self.dL

    def printResults(self):
        print('\n### Permeability resuts for "' + self.sid + '" ###')
        print('Pressure gradient: dPdL = ' + str(self.dPdL) + ' Pa/m')
        print('Flux: Q = ' + str(self.Q) + ' m^3/s')
        print('Intrinsic permeability: k = ' + str(self.k) + ' m^2')
        print('Saturated hydraulic conductivity: K = ' + str(self.K) + ' m/s')
        print('Mean porosity: phi_bar = ' + str(self.phi_bar) + '\n')

    def plotEvolution(self, axis=2, outformat='png'):
        '''
        Plot temporal evolution of parameters on the selected axis.
        Note that the first 5 output files are ignored.
        '''
        skipsteps = 5
        nsteps = self.sim.status() - skipsteps
        self.t_series = numpy.empty(nsteps)
        self.Q_series = numpy.empty((nsteps, 3))
        self.phi_bar_series = numpy.empty(nsteps)
        self.k_series = numpy.empty((nsteps, 3))
        self.K_series = numpy.empty((nsteps, 3))

        print('Reading ' + str(nsteps) + ' output files... '),
        sys.stdout.flush()
        for i in numpy.arange(skipsteps, self.sim.status()):
            self.sim.readstep(i, verbose=False)

            self.t_series[i-skipsteps] = self.sim.time_current[0]

            self.findCrossSectionalFlux()
            self.Q_series[i-skipsteps,:] = self.Q

            self.findMeanPorosity()
            self.phi_bar_series[i-skipsteps] = self.phi_bar

            self.findPermeability()
            self.k_series[i-skipsteps,:] = self.k

            self.findConductivity()
            self.K_series[i-skipsteps,:] = self.K
        print('Done')

        fig = plt.figure()

        plt.subplot(2,2,1)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Flux $Q$ [m^3/s]')
        plt.plot(self.t_series, self.Q_series[:,axis])
        #plt.legend()
        plt.grid()

        plt.subplot(2,2,2)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Porosity $\phi$ [-]')
        plt.plot(self.t_series, self.phi_bar_series)
        plt.grid()

        plt.subplot(2,2,3)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Permeability $k$ [m^2]')
        plt.plot(self.t_series, self.k_series[:,axis])
        plt.grid()

        plt.subplot(2,2,4)
        plt.xlabel('Time $t$ [s]')
        plt.ylabel('Conductivity $K$ [m/s]')
        plt.plot(self.t_series, self.K_series[:,axis])
        plt.grid()

        fig.tight_layout()

        filename = self.sid + '-permeability.' + outformat
        plt.savefig(filename)
        print('Figure saved as "' + filename + '"')
