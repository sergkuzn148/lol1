#!/usr/bin/env python
import math
import os
import subprocess
import pickle as pl
import numpy
try:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import matplotlib.collections
    matplotlib.rcParams.update({'font.size': 7, 'font.family': 'serif'})
    matplotlib.rc('text', usetex=True)
    matplotlib.rcParams['text.latex.preamble'] = [r"\usepackage{amsmath}"]
    from matplotlib.font_manager import FontProperties
    py_mpl = True
except ImportError:
    print('Info: Could not find "matplotlib" python module. ' +
          'Plotting functionality will be unavailable')
    py_mpl = False
try:
    import vtk
    py_vtk = True
except ImportError:
    print('Info: Could not find "vtk" python module. ' +
          'Fluid VTK calls will be unavailable')
    print('Consider installing with `pip install --user vtk`')
    py_vtk = False

numpy.seterr(all='warn', over='raise')

# Sphere version number. This field should correspond to the value in
# `../src/version.h`.
VERSION = 2.15

# Transparency on plot legends
legend_alpha = 0.5


class sim:
    '''
    Class containing all ``sphere`` data.

    Contains functions for reading and writing binaries, as well as simulation
    setup and data analysis. Most arrays are initialized to default values.

    :param np: The number of particles to allocate memory for (default=1)
    :type np: int
    :param nd: The number of spatial dimensions (default=3). Note that 2D and
        1D simulations currently are not possible.
    :type nd: int
    :param nw: The number of dynamic walls (default=1)
    :type nw: int
    :param sid: The simulation id (default='unnamed'). The simulation files
        will be written with this base name.
    :type sid: str
    :param fluid: Setup fluid simulation (default=False)
    :type fluid: bool
    :param cfd_solver: Fluid solver to use if fluid == True. 0: Navier-Stokes
        (default), 1: Darcy.
    :type cfd_solver: int
    '''

    def __init__(self, sid='unnamed', np=0, nd=3, nw=0, fluid=False):

        # Sphere version number
        self.version = numpy.ones(1, dtype=numpy.float64)*VERSION

        # The number of spatial dimensions. Values other that 3 do not work
        self.nd = int(nd)

        # The number of particles
        self.np = int(np)

        # The simulation id (text string)
        self.sid = sid

        ## Time parameters
        # Computational time step length [s]
        self.time_dt = numpy.zeros(1, dtype=numpy.float64)

        # Current time [s]
        self.time_current = numpy.zeros(1, dtype=numpy.float64)

        # Total time [s]
        self.time_total = numpy.zeros(1, dtype=numpy.float64)

        # File output interval [s]
        self.time_file_dt = numpy.zeros(1, dtype=numpy.float64)

        # The number of files written
        self.time_step_count = numpy.zeros(1, dtype=numpy.uint32)

        ## World dimensions and grid data
        # The Euclidean coordinate to the origo of the sorting grid
        self.origo = numpy.zeros(self.nd, dtype=numpy.float64)

        # The sorting grid size (x, y, z)
        self.L = numpy.zeros(self.nd, dtype=numpy.float64)

        # The number of sorting cells in each dimension
        self.num = numpy.zeros(self.nd, dtype=numpy.uint32)

        # Whether to treat the lateral boundaries as periodic (1) or not (0)
        self.periodic = numpy.zeros(1, dtype=numpy.uint32)

        # Adaptively resize grid to assemblage height (0: no, 1: yes)
        self.adaptive = numpy.zeros(1, dtype=numpy.uint32)

        ## Particle data
        # Particle position vectors [m]
        self.x = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # Particle radii [m]
        self.radius = numpy.ones(self.np, dtype=numpy.float64)

        # The sums of x and y movement [m]
        self.xyzsum = numpy.zeros((self.np, 3), dtype=numpy.float64)

        # The linear velocities [m/s]
        self.vel = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # Fix the particle horizontal velocities? 0: No, 1: Yes
        self.fixvel = numpy.zeros(self.np, dtype=numpy.float64)

        # The linear force vectors [N]
        self.force = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # The angular position vectors [rad]
        self.angpos = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # The angular velocity vectors [rad/s]
        self.angvel = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # The torque vectors [N*m]
        self.torque = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        # The shear friction energy dissipation rates [W]
        self.es_dot = numpy.zeros(self.np, dtype=numpy.float64)

        # The total shear energy dissipations [J]
        self.es = numpy.zeros(self.np, dtype=numpy.float64)

        # The viscous energy dissipation rates [W]
        self.ev_dot = numpy.zeros(self.np, dtype=numpy.float64)

        # The total viscois energy dissipation [J]
        self.ev = numpy.zeros(self.np, dtype=numpy.float64)

        # The total particle pressures [Pa]
        self.p = numpy.zeros(self.np, dtype=numpy.float64)

        # The gravitational acceleration vector [N*m/s]
        self.g = numpy.array([0.0, 0.0, 0.0], dtype=numpy.float64)

        # The Hookean coefficient for elastic stiffness normal to the contacts
        # [N/m]
        self.k_n = numpy.ones(1, dtype=numpy.float64) * 1.16e9

        # The Hookean coefficient for elastic stiffness tangential to the
        # contacts [N/m]
        self.k_t = numpy.ones(1, dtype=numpy.float64) * 1.16e9

        # The Hookean coefficient for elastic stiffness opposite of contact
        # rotations. UNUSED
        self.k_r = numpy.zeros(1, dtype=numpy.float64)

        # Young's modulus for contact stiffness [Pa]. This value is used
        # instead of the Hookean stiffnesses (k_n, k_t) when self.E is larger
        # than 0.0.
        self.E = numpy.zeros(1, dtype=numpy.float64)

        # The viscosity normal to the contact [N/(m/s)]
        self.gamma_n = numpy.zeros(1, dtype=numpy.float64)

        # The viscosity tangential to the contact [N/(m/s)]
        self.gamma_t = numpy.zeros(1, dtype=numpy.float64)

        # The viscosity to contact rotation [N/(m/s)]
        self.gamma_r = numpy.zeros(1, dtype=numpy.float64)

        # The coefficient of static friction on the contact [-]
        self.mu_s = numpy.ones(1, dtype=numpy.float64) * 0.5

        # The coefficient of dynamic friction on the contact [-]
        self.mu_d = numpy.ones(1, dtype=numpy.float64) * 0.5

        # The coefficient of rotational friction on the contact [-]
        self.mu_r = numpy.zeros(1, dtype=numpy.float64)

        # The viscosity normal to the walls [N/(m/s)]
        self.gamma_wn = numpy.zeros(1, dtype=numpy.float64)

        # The viscosity tangential to the walls [N/(m/s)]
        self.gamma_wt = numpy.zeros(1, dtype=numpy.float64)

        # The coeffient of static friction of the walls [-]
        self.mu_ws = numpy.ones(1, dtype=numpy.float64) * 0.5

        # The coeffient of dynamic friction of the walls [-]
        self.mu_wd = numpy.ones(1, dtype=numpy.float64) * 0.5

        # The particle density [kg/(m^3)]
        self.rho = numpy.ones(1, dtype=numpy.float64) * 2600.0

        # The contact model to use
        # 1: Normal: elasto-viscous, tangential: visco-frictional
        # 2: Normal: elasto-viscous, tangential: elasto-visco-frictional
        self.contactmodel = numpy.ones(1, dtype=numpy.uint32) * 2 # lin-visc-el

        # Capillary bond prefactor
        self.kappa = numpy.zeros(1, dtype=numpy.float64)

        # Capillary bond debonding distance [m]
        self.db = numpy.zeros(1, dtype=numpy.float64)

        # Capillary bond liquid volume [m^3]
        self.V_b = numpy.zeros(1, dtype=numpy.float64)

        ## Wall data
        # Number of dynamic walls
        # nw=1: Uniaxial (also used for shear experiments)
        # nw=2: Biaxial
        # nw=5: Triaxial
        self.nw = int(nw)

        # Wall modes
        # 0: Fixed
        # 1: Normal stress condition
        # 2: Normal velocity condition
        # 3: Normal stress and shear stress condition
        self.wmode = numpy.zeros(self.nw, dtype=numpy.int32)

        # Wall normals
        self.w_n = numpy.zeros((self.nw, self.nd), dtype=numpy.float64)
        if self.nw >= 1:
            self.w_n[0, 2] = -1.0
        if self.nw >= 2:
            self.w_n[1, 0] = -1.0
        if self.nw >= 3:
            self.w_n[2, 0] = 1.0
        if self.nw >= 4:
            self.w_n[3, 1] = -1.0
        if self.nw >= 5:
            self.w_n[4, 1] = 1.0

        # Wall positions on the axes that are parallel to the wall normal [m]
        self.w_x = numpy.ones(self.nw, dtype=numpy.float64)

        # Wall masses [kg]
        self.w_m = numpy.zeros(self.nw, dtype=numpy.float64)

        # Wall velocities on the axes that are parallel to the wall normal [m/s]
        self.w_vel = numpy.zeros(self.nw, dtype=numpy.float64)

        # Wall forces on the axes that are parallel to the wall normal [m/s]
        self.w_force = numpy.zeros(self.nw, dtype=numpy.float64)

        # Wall stress on the axes that are parallel to the wall normal [Pa]
        self.w_sigma0 = numpy.zeros(self.nw, dtype=numpy.float64)

        # Wall stress modulation amplitude [Pa]
        self.w_sigma0_A = numpy.zeros(1, dtype=numpy.float64)

        # Wall stress modulation frequency [Hz]
        self.w_sigma0_f = numpy.zeros(1, dtype=numpy.float64)

        # Wall shear stress, enforced when wmode == 3
        self.w_tau_x = numpy.zeros(1, dtype=numpy.float64)

        ## Bond parameters
        # Radius multiplier to the parallel-bond radii
        self.lambda_bar = numpy.ones(1, dtype=numpy.float64)

        # Number of bonds
        self.nb0 = 0

        # Bond tensile strength [Pa]
        self.sigma_b = numpy.ones(1, dtype=numpy.float64) * numpy.infty

        # Bond shear strength [Pa]
        self.tau_b = numpy.ones(1, dtype=numpy.float64) * numpy.infty

        # Bond pairs
        self.bonds = numpy.zeros((self.nb0, 2), dtype=numpy.uint32)

        # Parallel bond movement
        self.bonds_delta_n = numpy.zeros(self.nb0, dtype=numpy.float64)

        # Shear bond movement
        self.bonds_delta_t = numpy.zeros((self.nb0, self.nd), dtype=numpy.float64)

        # Twisting bond movement
        self.bonds_omega_n = numpy.zeros(self.nb0, dtype=numpy.float64)

        # Bending bond movement
        self.bonds_omega_t = numpy.zeros((self.nb0, self.nd), dtype=numpy.float64)

        ## Fluid parameters

        # Simulate fluid? True: Yes, False: no
        self.fluid = fluid

        if self.fluid:

            # Fluid solver type
            # 0: Navier Stokes (fluid with inertia)
            # 1: Stokes-Darcy (fluid without inertia)
            self.cfd_solver = numpy.zeros(1, dtype=numpy.int32)

            # Fluid dynamic viscosity [N/(m/s)]
            self.mu = numpy.zeros(1, dtype=numpy.float64)

            # Fluid velocities [m/s]
            self.v_f = numpy.zeros((self.num[0], self.num[1], self.num[2], self.nd),
                                   dtype=numpy.float64)

            # Fluid pressures [Pa]
            self.p_f = numpy.zeros((self.num[0], self.num[1], self.num[2]),
                                   dtype=numpy.float64)

            # Fluid cell porosities [-]
            self.phi = numpy.zeros((self.num[0], self.num[1], self.num[2]),
                                   dtype=numpy.float64)

            # Fluid cell porosity change [1/s]
            self.dphi = numpy.zeros((self.num[0], self.num[1], self.num[2]),
                                    dtype=numpy.float64)

            # Fluid density [kg/(m^3)]
            self.rho_f = numpy.ones(1, dtype=numpy.float64) * 1.0e3

            # Pressure modulation at the top boundary
            self.p_mod_A = numpy.zeros(1, dtype=numpy.float64)  # Amplitude [Pa]
            self.p_mod_f = numpy.zeros(1, dtype=numpy.float64)  # Frequency [Hz]
            self.p_mod_phi = numpy.zeros(1, dtype=numpy.float64) # Shift [rad]

            ## Fluid solver parameters

            if self.cfd_solver[0] == 1:  # Darcy solver
                # Boundary conditions at the sides of the fluid grid
                # 0: Dirichlet
                # 1: Neumann
                # 2: Periodic (default)
                self.bc_xn = numpy.ones(1, dtype=numpy.int32)*2  # Neg. x bc
                self.bc_xp = numpy.ones(1, dtype=numpy.int32)*2  # Pos. x bc
                self.bc_yn = numpy.ones(1, dtype=numpy.int32)*2  # Neg. y bc
                self.bc_yp = numpy.ones(1, dtype=numpy.int32)*2  # Pos. y bc

            # Boundary conditions at the top and bottom of the fluid grid
            # 0: Dirichlet (default)
            # 1: Neumann free slip
            # 2: Neumann no slip (Navier Stokes), Periodic (Darcy)
            # 3: Periodic (Navier-Stokes solver only)
            # 4: Constant flux (Darcy solver only)
            self.bc_bot = numpy.zeros(1, dtype=numpy.int32)
            self.bc_top = numpy.zeros(1, dtype=numpy.int32)
            # Free slip boundaries? 1: yes
            self.free_slip_bot = numpy.ones(1, dtype=numpy.int32)
            self.free_slip_top = numpy.ones(1, dtype=numpy.int32)

            # Boundary-normal flux (in case of bc_*=4)
            self.bc_bot_flux = numpy.zeros(1, dtype=numpy.float64)
            self.bc_top_flux = numpy.zeros(1, dtype=numpy.float64)

            # Hold pressures constant in fluid cell (0: True, 1: False)
            self.p_f_constant = numpy.zeros((self.num[0],
                                             self.num[1],
                                             self.num[2]), dtype=numpy.int32)

            # Navier-Stokes
            if self.cfd_solver[0] == 0:

                # Smoothing parameter, should be in the range [0.0;1.0[.
                # 0.0=no smoothing.
                self.gamma = numpy.array(0.0)

                # Under-relaxation parameter, should be in the range ]0.0;1.0].
                # 1.0=no under-relaxation
                self.theta = numpy.array(1.0)

                # Velocity projection parameter, should be in the range
                # [0.0;1.0]
                self.beta = numpy.array(0.0)

                # Tolerance criteria for the normalized max. residual
                self.tolerance = numpy.array(1.0e-3)

                # The maximum number of iterations to perform per time step
                self.maxiter = numpy.array(1e4)

                # The number of DEM time steps to perform between CFD updates
                self.ndem = numpy.array(1)

                # Porosity scaling factor
                self.c_phi = numpy.ones(1, dtype=numpy.float64)

                # Fluid velocity scaling factor
                self.c_v = numpy.ones(1, dtype=numpy.float64)

                # DEM-CFD time scaling factor
                self.dt_dem_fac = numpy.ones(1, dtype=numpy.float64)

                ## Interaction forces
                self.f_d = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
                self.f_p = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
                self.f_v = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
                self.f_sum = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

            # Darcy
            elif self.cfd_solver[0] == 1:

                # Tolerance criteria for the normalized max. residual
                self.tolerance = numpy.array(1.0e-3)

                # The maximum number of iterations to perform per time step
                self.maxiter = numpy.array(1e4)

                # The number of DEM time steps to perform between CFD updates
                self.ndem = numpy.array(1)

                # Porosity scaling factor
                self.c_phi = numpy.ones(1, dtype=numpy.float64)

                # Interaction forces
                self.f_p = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

                # Adiabatic fluid compressibility [1/Pa].
                # Fluid bulk modulus=1/self.beta_f
                self.beta_f = numpy.ones(1, dtype=numpy.float64)*4.5e-10

                # Hydraulic permeability prefactor [m*m]
                self.k_c = numpy.ones(1, dtype=numpy.float64)*4.6e-10

            else:
                raise Exception('Value of cfd_solver not understood (' + \
                                str(self.cfd_solver[0]) + ')')

        # Particle color marker
        self.color = numpy.zeros(self.np, dtype=numpy.int32)

    def __eq__(self, other):
        '''
        Called when to sim objects are compared. Returns 0 if the values
        are identical.
        '''
        if self.version != other.version:
            print('version')
            return False
        elif self.nd != other.nd:
            print('nd')
            return False
        elif self.np != other.np:
            print('np')
            return False
        elif self.time_dt != other.time_dt:
            print('time_dt')
            return False
        elif self.time_current != other.time_current:
            print('time_current')
            return False
        elif self.time_total != other.time_total:
            print('time_total')
            return False
        elif self.time_file_dt != other.time_file_dt:
            print('time_file_dt')
            return False
        elif self.time_step_count != other.time_step_count:
            print('time_step_count')
            return False
        elif (self.origo != other.origo).any():
            print('origo')
            return False
        elif (self.L != other.L).any():
            print('L')
            return 11
        elif (self.num != other.num).any():
            print('num')
            return False
        elif self.periodic != other.periodic:
            print('periodic')
            return False
        elif self.adaptive != other.adaptive:
            print('adaptive')
            return False
        elif (self.x != other.x).any():
            print('x')
            return False
        elif (self.radius != other.radius).any():
            print('radius')
            return False
        elif (self.xyzsum != other.xyzsum).any():
            print('xyzsum')
            return False
        elif (self.vel != other.vel).any():
            print('vel')
            return False
        elif (self.fixvel != other.fixvel).any():
            print('fixvel')
            return False
        elif (self.force != other.force).any():
            print('force')
            return False
        elif (self.angpos != other.angpos).any():
            print('angpos')
            return False
        elif (self.angvel != other.angvel).any():
            print('angvel')
            return False
        elif (self.torque != other.torque).any():
            print('torque')
            return False
        elif (self.es_dot != other.es_dot).any():
            print('es_dot')
            return False
        elif (self.es != other.es).any():
            print('es')
            return False
        elif (self.ev_dot != other.ev_dot).any():
            print('ev_dot')
            return False
        elif (self.ev != other.ev).any():
            print('ev')
            return False
        elif (self.p != other.p).any():
            print('p')
            return False
        elif (self.g != other.g).any():
            print('g')
            return False
        elif self.k_n != other.k_n:
            print('k_n')
            return False
        elif self.k_t != other.k_t:
            print('k_t')
            return False
        elif self.k_r != other.k_r:
            print('k_r')
            return False
        elif self.E != other.E:
            print('E')
            return False
        elif self.gamma_n != other.gamma_n:
            print('gamma_n')
            return False
        elif self.gamma_t != other.gamma_t:
            print('gamma_t')
            return False
        elif self.gamma_r != other.gamma_r:
            print('gamma_r')
            return False
        elif self.mu_s != other.mu_s:
            print('mu_s')
            return False
        elif self.mu_d != other.mu_d:
            print('mu_d')
            return False
        elif self.mu_r != other.mu_r:
            print('mu_r')
            return False
        elif self.rho != other.rho:
            print('rho')
            return False
        elif self.contactmodel != other.contactmodel:
            print('contactmodel')
            return False
        elif self.kappa != other.kappa:
            print('kappa')
            return False
        elif self.db != other.db:
            print('db')
            return False
        elif self.V_b != other.V_b:
            print('V_b')
            return False
        elif self.nw != other.nw:
            print('nw')
            return False
        elif (self.wmode != other.wmode).any():
            print('wmode')
            return False
        elif (self.w_n != other.w_n).any():
            print('w_n')
            return False
        elif (self.w_x != other.w_x).any():
            print('w_x')
            return False
        elif (self.w_m != other.w_m).any():
            print('w_m')
            return False
        elif (self.w_vel != other.w_vel).any():
            print('w_vel')
            return False
        elif (self.w_force != other.w_force).any():
            print('w_force')
            return False
        elif (self.w_sigma0 != other.w_sigma0).any():
            print('w_sigma0')
            return False
        elif self.w_sigma0_A != other.w_sigma0_A:
            print('w_sigma0_A')
            return False
        elif self.w_sigma0_f != other.w_sigma0_f:
            print('w_sigma0_f')
            return False
        elif self.w_tau_x != other.w_tau_x:
            print('w_tau_x')
            return False
        elif self.gamma_wn != other.gamma_wn:
            print('gamma_wn')
            return False
        elif self.gamma_wt != other.gamma_wt:
            print('gamma_wt')
            return False
        elif self.lambda_bar != other.lambda_bar:
            print('lambda_bar')
            return False
        elif self.nb0 != other.nb0:
            print('nb0')
            return False
        elif self.sigma_b != other.sigma_b:
            print('sigma_b')
            return False
        elif self.tau_b != other.tau_b:
            print('tau_b')
            return False
        elif self.bonds != other.bonds:
            print('bonds')
            return False
        elif self.bonds_delta_n != other.bonds_delta_n:
            print('bonds_delta_n')
            return False
        elif self.bonds_delta_t != other.bonds_delta_t:
            print('bonds_delta_t')
            return False
        elif self.bonds_omega_n != other.bonds_omega_n:
            print('bonds_omega_n')
            return False
        elif self.bonds_omega_t != other.bonds_omega_t:
            print('bonds_omega_t')
            return False
        elif self.fluid != other.fluid:
            print('fluid')
            return False

        if self.fluid:
            if self.cfd_solver != other.cfd_solver:
                print('cfd_solver')
                return False
            elif self.mu != other.mu:
                print('mu')
                return False
            elif (self.v_f != other.v_f).any():
                print('v_f')
                return False
            elif (self.p_f != other.p_f).any():
                print('p_f')
                return False
            #elif self.phi != other.phi).any():
                #print('phi')
                #return False  # Porosities not initialized correctly
            elif (self.dphi != other.dphi).any():
                print('d_phi')
                return False
            elif self.rho_f != other.rho_f:
                print('rho_f')
                return False
            elif self.p_mod_A != other.p_mod_A:
                print('p_mod_A')
                return False
            elif self.p_mod_f != other.p_mod_f:
                print('p_mod_f')
                return False
            elif self.p_mod_phi != other.p_mod_phi:
                print('p_mod_phi')
                return False
            elif self.bc_bot != other.bc_bot:
                print('bc_bot')
                return False
            elif self.bc_top != other.bc_top:
                print('bc_top')
                return False
            elif self.free_slip_bot != other.free_slip_bot:
                print('free_slip_bot')
                return False
            elif self.free_slip_top != other.free_slip_top:
                print('free_slip_top')
                return False
            elif self.bc_bot_flux != other.bc_bot_flux:
                print('bc_bot_flux')
                return False
            elif self.bc_top_flux != other.bc_top_flux:
                print('bc_top_flux')
                return False
            elif (self.p_f_constant != other.p_f_constant).any():
                print('p_f_constant')
                return False

            if self.cfd_solver == 0:
                if self.gamma != other.gamma:
                    print('gamma')
                    return False
                elif self.theta != other.theta:
                    print('theta')
                    return False
                elif self.beta != other.beta:
                    print('beta')
                    return False
                elif self.tolerance != other.tolerance:
                    print('tolerance')
                    return False
                elif self.maxiter != other.maxiter:
                    print('maxiter')
                    return False
                elif self.ndem != other.ndem:
                    print('ndem')
                    return False
                elif self.c_phi != other.c_phi:
                    print('c_phi')
                    return 84
                elif self.c_v != other.c_v:
                    print('c_v')
                elif self.dt_dem_fac != other.dt_dem_fac:
                    print('dt_dem_fac')
                    return 85
                elif (self.f_d != other.f_d).any():
                    print('f_d')
                    return 86
                elif (self.f_p != other.f_p).any():
                    print('f_p')
                    return 87
                elif (self.f_v != other.f_v).any():
                    print('f_v')
                    return 88
                elif (self.f_sum != other.f_sum).any():
                    print('f_sum')
                    return 89

            if self.cfd_solver == 1:
                if self.tolerance != other.tolerance:
                    print('tolerance')
                    return False
                elif self.maxiter != other.maxiter:
                    print('maxiter')
                    return False
                elif self.ndem != other.ndem:
                    print('ndem')
                    return False
                elif self.c_phi != other.c_phi:
                    print('c_phi')
                    return 84
                elif (self.f_p != other.f_p).any():
                    print('f_p')
                    return 86
                elif self.beta_f != other.beta_f:
                    print('beta_f')
                    return 87
                elif self.k_c != other.k_c:
                    print('k_c')
                    return 88
                elif self.bc_xn != other.bc_xn:
                    print('bc_xn')
                    return False
                elif self.bc_xp != other.bc_xp:
                    print('bc_xp')
                    return False
                elif self.bc_yn != other.bc_yn:
                    print('bc_yn')
                    return False
                elif self.bc_yp != other.bc_yp:
                    print('bc_yp')
                    return False

        if (self.color != other.color).any():
            print('color')
            return False

        # All equal
        return True

    def id(self, sid=''):
        '''
        Returns or sets the simulation id/name, which is used to identify
        simulation files in the output folders.

        :param sid: The desired simulation id. If left blank the current
            simulation id will be returned.
        :type sid: str
        :returns: The current simulation id if no new value is set.
        :return type: str
        '''
        if sid == '':
            return self.sid
        else:
            self.sid = sid

    def idAppend(self, string):
        '''
        Append a string to the simulation id/name, which is used to identify
        simulation files in the output folders.

        :param string: The string to append to the simulation id (`self.sid`).
        :type string: str
        '''
        self.sid += string

    def addParticle(self, x, radius, xyzsum=numpy.zeros(3), vel=numpy.zeros(3),
                    fixvel=numpy.zeros(1), force=numpy.zeros(3),
                    angpos=numpy.zeros(3), angvel=numpy.zeros(3),
                    torque=numpy.zeros(3), es_dot=numpy.zeros(1),
                    es=numpy.zeros(1), ev_dot=numpy.zeros(1),
                    ev=numpy.zeros(1), p=numpy.zeros(1), color=0):
        '''
        Add a single particle to the simulation object. The only required
        parameters are the position (x) and the radius (radius).

        :param x: A vector pointing to the particle center coordinate.
        :type x: numpy.array
        :param radius: The particle radius
        :type radius: float
        :param vel: The particle linear velocity (default=[0, 0, 0])
        :type vel: numpy.array
        :param fixvel: 0: Do not fix particle velocity (default), 1: Fix
            horizontal linear velocity, -1: Fix horizontal and vertical linear
            velocity
        :type fixvel: float
        :param angpos: The particle angular position (default=[0, 0, 0])
        :type angpos: numpy.array
        :param angvel: The particle angular velocity (default=[0, 0, 0])
        :type angvel: numpy.array
        :param torque: The particle torque (default=[0, 0, 0])
        :type torque: numpy.array
        :param es_dot: The particle shear energy loss rate (default=0)
        :type es_dot: float
        :param es: The particle shear energy loss (default=0)
        :type es: float
        :param ev_dot: The particle viscous energy rate loss (default=0)
        :type ev_dot: float
        :param ev: The particle viscous energy loss (default=0)
        :type ev: float
        :param p: The particle pressure (default=0)
        :type p: float
        '''

        self.np += 1

        self.x = numpy.append(self.x, [x], axis=0)
        self.radius = numpy.append(self.radius, radius)
        self.vel = numpy.append(self.vel, [vel], axis=0)
        self.xyzsum = numpy.append(self.xyzsum, [xyzsum], axis=0)
        self.fixvel = numpy.append(self.fixvel, fixvel)
        self.force = numpy.append(self.force, [force], axis=0)
        self.angpos = numpy.append(self.angpos, [angpos], axis=0)
        self.angvel = numpy.append(self.angvel, [angvel], axis=0)
        self.torque = numpy.append(self.torque, [torque], axis=0)
        self.es_dot = numpy.append(self.es_dot, es_dot)
        self.es = numpy.append(self.es, es)
        self.ev_dot = numpy.append(self.ev_dot, ev_dot)
        self.ev = numpy.append(self.ev, ev)
        self.p = numpy.append(self.p, p)
        self.color = numpy.append(self.color, color)
        if self.fluid:
            self.f_d = numpy.append(self.f_d, [numpy.zeros(3)], axis=0)
            self.f_p = numpy.append(self.f_p, [numpy.zeros(3)], axis=0)
            self.f_v = numpy.append(self.f_v, [numpy.zeros(3)], axis=0)
            self.f_sum = numpy.append(self.f_sum, [numpy.zeros(3)], axis=0)

    def deleteParticle(self, i):
        '''
        Delete particle(s) with index ``i``.

        :param i: One or more particle indexes to delete
        :type i: int, list or numpy.array
        '''

        # The user wants to delete several particles, indexes in a numpy.array
        if type(i) == numpy.ndarray:
            self.np -= i.size

        # The user wants to delete several particles, indexes in a Python list
        elif type(i) == list:
            self.np -= len(i)

        # The user wants to delete a single particle with a integer index
        else:
            self.np -= 1

        if type(i) == tuple:
            raise Exception('Cannot parse tuples as index value. ' +
                            'Valid types are int, list and numpy.ndarray')


        self.x = numpy.delete(self.x, i, axis=0)
        self.radius = numpy.delete(self.radius, i)
        self.vel = numpy.delete(self.vel, i, axis=0)
        self.xyzsum = numpy.delete(self.xyzsum, i, axis=0)
        self.fixvel = numpy.delete(self.fixvel, i)
        self.force = numpy.delete(self.force, i, axis=0)
        self.angpos = numpy.delete(self.angpos, i, axis=0)
        self.angvel = numpy.delete(self.angvel, i, axis=0)
        self.torque = numpy.delete(self.torque, i, axis=0)
        self.es_dot = numpy.delete(self.es_dot, i)
        self.es = numpy.delete(self.es, i)
        self.ev_dot = numpy.delete(self.ev_dot, i)
        self.ev = numpy.delete(self.ev, i)
        self.p = numpy.delete(self.p, i)
        self.color = numpy.delete(self.color, i)
        if self.fluid:
            # Darcy and Navier-Stokes
            self.f_p = numpy.delete(self.f_p, i, axis=0)
            if self.cfd_solver[0] == 0: # Navier-Stokes
                self.f_d = numpy.delete(self.f_d, i, axis=0)
                self.f_v = numpy.delete(self.f_v, i, axis=0)
                self.f_sum = numpy.delete(self.f_sum, i, axis=0)

    def deleteAllParticles(self):
        '''
        Deletes all particles in the simulation object.
        '''
        self.np = 0
        self.x = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.radius = numpy.ones(self.np, dtype=numpy.float64)
        self.xyzsum = numpy.zeros((self.np, 3), dtype=numpy.float64)
        self.vel = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.fixvel = numpy.zeros(self.np, dtype=numpy.float64)
        self.force = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.angpos = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.angvel = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.torque = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.es_dot = numpy.zeros(self.np, dtype=numpy.float64)
        self.es = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev_dot = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev = numpy.zeros(self.np, dtype=numpy.float64)
        self.p = numpy.zeros(self.np, dtype=numpy.float64)
        self.color = numpy.zeros(self.np, dtype=numpy.int32)
        self.f_d = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.f_p = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.f_v = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
        self.f_sum = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

    def readbin(self, targetbin, verbose=True, bonds=True, sigma0mod=True,
                esysparticle=False):
        '''
        Reads a target ``sphere`` binary file.

        See also :func:`writebin()`, :func:`readfirst()`, :func:`readlast()`,
        :func:`readsecond`, and :func:`readstep`.

        :param targetbin: The path to the binary ``sphere`` file
        :type targetbin: str
        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        :param bonds: The input file contains bond information (default=True).
            This parameter should be true for all recent ``sphere`` versions.
        :type bonds: bool
        :param sigma0mod: The input file contains information about modulating
            stresses at the top wall (default=True). This parameter should be
            true for all recent ``sphere`` versions.
        :type sigma0mod: bool
        :param esysparticle: Stop reading the file after reading the kinematics,
            which is useful for reading output files from other DEM programs.
            (default=False)
        :type esysparticle: bool
        '''

        fh = None
        try:
            if verbose:
                print("Input file: {0}".format(targetbin))
            fh = open(targetbin, "rb")

            # Read the file version
            self.version = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            # Read the number of dimensions and particles
            self.nd = int(numpy.fromfile(fh, dtype=numpy.int32, count=1))
            self.np = int(numpy.fromfile(fh, dtype=numpy.uint32, count=1))

            # Read the time variables
            self.time_dt = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_current = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_total = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_file_dt = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.time_step_count = numpy.fromfile(fh, dtype=numpy.uint32, count=1)

            # Allocate array memory for particles
            self.x = numpy.empty((self.np, self.nd), dtype=numpy.float64)
            self.radius = numpy.empty(self.np, dtype=numpy.float64)
            self.xyzsum = numpy.empty((self.np, 3), dtype=numpy.float64)
            self.vel = numpy.empty((self.np, self.nd), dtype=numpy.float64)
            self.fixvel = numpy.empty(self.np, dtype=numpy.float64)
            self.es_dot = numpy.empty(self.np, dtype=numpy.float64)
            self.es = numpy.empty(self.np, dtype=numpy.float64)
            self.ev_dot = numpy.empty(self.np, dtype=numpy.float64)
            self.ev = numpy.empty(self.np, dtype=numpy.float64)
            self.p = numpy.empty(self.np, dtype=numpy.float64)

            # Read remaining data from binary
            self.origo = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.L = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.num = numpy.fromfile(fh, dtype=numpy.uint32, count=self.nd)
            self.periodic = numpy.fromfile(fh, dtype=numpy.int32, count=1)

            if self.version >= 2.14:
                self.adaptive = numpy.fromfile(fh, dtype=numpy.int32, count=1)
            else:
                self.adaptive = numpy.zeros(1, dtype=numpy.float64)

            # Per-particle vectors
            for i in numpy.arange(self.np):
                self.x[i, :] =\
                        numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.radius[i] =\
                        numpy.fromfile(fh, dtype=numpy.float64, count=1)

            if self.version >= 1.03:
                self.xyzsum = numpy.fromfile(fh, dtype=numpy.float64,\
                                          count=self.np*3).reshape(self.np, 3)
            else:
                self.xyzsum = numpy.fromfile(fh, dtype=numpy.float64,\
                                          count=self.np*2).reshape(self.np, 2)

            for i in numpy.arange(self.np):
                self.vel[i, :] = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.fixvel[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            self.force = numpy.fromfile(fh, dtype=numpy.float64,\
                                     count=self.np*self.nd)\
                                     .reshape(self.np, self.nd)

            self.angpos = numpy.fromfile(fh, dtype=numpy.float64,\
                                      count=self.np*self.nd)\
                                      .reshape(self.np, self.nd)
            self.angvel = numpy.fromfile(fh, dtype=numpy.float64,\
                                      count=self.np*self.nd)\
                                      .reshape(self.np, self.nd)
            self.torque = numpy.fromfile(fh, dtype=numpy.float64,\
                                      count=self.np*self.nd)\
                                      .reshape(self.np, self.nd)

            if esysparticle:
                return

            # Per-particle single-value parameters
            self.es_dot = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.es = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.ev_dot = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.ev = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)
            self.p = numpy.fromfile(fh, dtype=numpy.float64, count=self.np)

            # Constant, global physical parameters
            self.g = numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
            self.k_n = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.k_t = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.k_r = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            if self.version >= 2.13:
                self.E = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            else:
                self.E = numpy.zeros(1, dtype=numpy.float64)
            self.gamma_n = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_t = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_r = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_s = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_d = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_r = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_wn = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.gamma_wt = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_ws = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.mu_wd = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.rho = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.contactmodel = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
            self.kappa = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.db = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            self.V_b = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            # Wall data
            self.nw = int(numpy.fromfile(fh, dtype=numpy.uint32, count=1))
            self.wmode = numpy.empty(self.nw, dtype=numpy.int32)
            self.w_n = numpy.empty(self.nw*self.nd, dtype=numpy.float64)\
                       .reshape(self.nw, self.nd)
            self.w_x = numpy.empty(self.nw, dtype=numpy.float64)
            self.w_m = numpy.empty(self.nw, dtype=numpy.float64)
            self.w_vel = numpy.empty(self.nw, dtype=numpy.float64)
            self.w_force = numpy.empty(self.nw, dtype=numpy.float64)
            self.w_sigma0 = numpy.empty(self.nw, dtype=numpy.float64)

            self.wmode = numpy.fromfile(fh, dtype=numpy.int32, count=self.nw)
            for i in numpy.arange(self.nw):
                self.w_n[i, :] =\
                        numpy.fromfile(fh, dtype=numpy.float64, count=self.nd)
                self.w_x[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            for i in numpy.arange(self.nw):
                self.w_m[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_vel[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_force[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_sigma0[i] = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            if sigma0mod:
                self.w_sigma0_A = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.w_sigma0_f = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            if self.version >= 2.1:
                self.w_tau_x = numpy.fromfile(fh, dtype=numpy.float64, count=1)
            else:
                self.w_tau_x = numpy.zeros(1, dtype=numpy.float64)

            if bonds:
                # Inter-particle bonds
                self.lambda_bar = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.nb0 = int(numpy.fromfile(fh, dtype=numpy.uint32, count=1))
                self.sigma_b = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.tau_b = numpy.fromfile(fh, dtype=numpy.float64, count=1)
                self.bonds = numpy.empty((self.nb0, 2), dtype=numpy.uint32)
                for i in numpy.arange(self.nb0):
                    self.bonds[i, 0] = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
                    self.bonds[i, 1] = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
                self.bonds_delta_n = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=self.nb0)
                self.bonds_delta_t = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=self.nb0*self.nd)\
                                                    .reshape(self.nb0, self.nd)
                self.bonds_omega_n = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=self.nb0)
                self.bonds_omega_t = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=self.nb0*self.nd)\
                                                    .reshape(self.nb0, self.nd)
            else:
                self.nb0 = 0

            if self.fluid:

                if self.version >= 2.0:
                    self.cfd_solver = numpy.fromfile(fh, dtype=numpy.int32, count=1)
                else:
                    self.cfd_solver = numpy.zeros(1, dtype=numpy.int32)

                self.mu = numpy.fromfile(fh, dtype=numpy.float64, count=1)

                self.v_f = numpy.empty((self.num[0],
                                        self.num[1],
                                        self.num[2],
                                        self.nd), dtype=numpy.float64)
                self.p_f = numpy.empty((self.num[0],
                                        self.num[1],
                                        self.num[2]), dtype=numpy.float64)
                self.phi = numpy.empty((self.num[0],
                                        self.num[1],
                                        self.num[2]), dtype=numpy.float64)
                self.dphi = numpy.empty((self.num[0],
                                         self.num[1],
                                         self.num[2]), dtype=numpy.float64)

                for z in numpy.arange(self.num[2]):
                    for y in numpy.arange(self.num[1]):
                        for x in numpy.arange(self.num[0]):
                            self.v_f[x, y, z, 0] = numpy.fromfile(fh,
                                                                  dtype=numpy.float64,
                                                                  count=1)
                            self.v_f[x, y, z, 1] = numpy.fromfile(fh,
                                                                  dtype=numpy.float64,
                                                                  count=1)
                            self.v_f[x, y, z, 2] = numpy.fromfile(fh,
                                                                  dtype=numpy.float64,
                                                                  count=1)
                            self.p_f[x, y, z] = numpy.fromfile(fh,
                                                               dtype=numpy.float64,
                                                               count=1)
                            self.phi[x, y, z] = numpy.fromfile(fh,
                                                               dtype=numpy.float64,
                                                               count=1)
                            self.dphi[x, y, z] = numpy.fromfile(fh,
                                                                dtype=numpy.float64,
                                                                count=1)\
                                                 /(self.time_dt*self.ndem)

                if self.version >= 0.36:
                    self.rho_f = numpy.fromfile(fh, dtype=numpy.float64,
                                                count=1)
                    self.p_mod_A = numpy.fromfile(fh, dtype=numpy.float64,
                                                  count=1)
                    self.p_mod_f = numpy.fromfile(fh, dtype=numpy.float64,
                                                  count=1)
                    self.p_mod_phi = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=1)

                    if self.version >= 2.12 and self.cfd_solver[0] == 1:
                        self.bc_xn = numpy.fromfile(fh, dtype=numpy.int32,
                                                    count=1)
                        self.bc_xp = numpy.fromfile(fh, dtype=numpy.int32,
                                                    count=1)
                        self.bc_yn = numpy.fromfile(fh, dtype=numpy.int32,
                                                    count=1)
                        self.bc_yp = numpy.fromfile(fh, dtype=numpy.int32,
                                                    count=1)

                    self.bc_bot = numpy.fromfile(fh, dtype=numpy.int32, count=1)
                    self.bc_top = numpy.fromfile(fh, dtype=numpy.int32, count=1)
                    self.free_slip_bot = numpy.fromfile(fh, dtype=numpy.int32,
                                                        count=1)
                    self.free_slip_top = numpy.fromfile(fh, dtype=numpy.int32,
                                                        count=1)
                    if self.version >= 2.11:
                        self.bc_bot_flux = numpy.fromfile(fh,
                                                          dtype=numpy.float64,
                                                          count=1)
                        self.bc_top_flux = numpy.fromfile(fh,
                                                          dtype=numpy.float64,
                                                          count=1)
                    else:
                        self.bc_bot_flux = numpy.zeros(1, dtype=numpy.float64)
                        self.bc_top_flux = numpy.zeros(1, dtype=numpy.float64)

                    if self.version >= 2.15:
                        self.p_f_constant = numpy.empty((self.num[0],
                                                         self.num[1],
                                                         self.num[2]),
                                                        dtype=numpy.int32)

                        for z in numpy.arange(self.num[2]):
                            for y in numpy.arange(self.num[1]):
                                for x in numpy.arange(self.num[0]):
                                    self.p_f_constant[x, y, z] = \
                                        numpy.fromfile(fh, dtype=numpy.int32,
                                                       count=1)
                    else:
                        self.p_f_constant = numpy.zeros((self.num[0],
                                                         self.num[1],
                                                         self.num[2]),
                                                        dtype=numpy.int32)

                if self.version >= 2.0 and self.cfd_solver == 0:
                    self.gamma = numpy.fromfile(fh, dtype=numpy.float64,
                                                count=1)
                    self.theta = numpy.fromfile(fh, dtype=numpy.float64,
                                                count=1)
                    self.beta = numpy.fromfile(fh, dtype=numpy.float64,
                                               count=1)
                    self.tolerance = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=1)
                    self.maxiter = numpy.fromfile(fh, dtype=numpy.uint32,
                                                  count=1)
                    if self.version >= 1.01:
                        self.ndem = numpy.fromfile(fh, dtype=numpy.uint32,
                                                   count=1)
                    else:
                        self.ndem = 1

                    if self.version >= 1.04:
                        self.c_phi = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=1)
                        self.c_v = numpy.fromfile(fh, dtype=numpy.float64,
                                                  count=1)
                        if self.version == 1.06:
                            self.c_a = numpy.fromfile(fh, dtype=numpy.float64,
                                                      count=1)
                        elif self.version >= 1.07:
                            self.dt_dem_fac = numpy.fromfile(fh,
                                                             dtype=numpy.float64,
                                                             count=1)
                        else:
                            self.c_a = numpy.ones(1, dtype=numpy.float64)
                    else:
                        self.c_phi = numpy.ones(1, dtype=numpy.float64)
                        self.c_v = numpy.ones(1, dtype=numpy.float64)

                    if self.version >= 1.05:
                        self.f_d = numpy.empty_like(self.x)
                        self.f_p = numpy.empty_like(self.x)
                        self.f_v = numpy.empty_like(self.x)
                        self.f_sum = numpy.empty_like(self.x)

                        for i in numpy.arange(self.np):
                            self.f_d[i, :] = numpy.fromfile(fh,
                                                            dtype=numpy.float64,
                                                            count=self.nd)
                        for i in numpy.arange(self.np):
                            self.f_p[i, :] = numpy.fromfile(fh,
                                                            dtype=numpy.float64,
                                                            count=self.nd)
                        for i in numpy.arange(self.np):
                            self.f_v[i, :] = numpy.fromfile(fh,
                                                            dtype=numpy.float64,
                                                            count=self.nd)
                        for i in numpy.arange(self.np):
                            self.f_sum[i, :] = numpy.fromfile(fh,
                                                              dtype=numpy.float64,
                                                              count=self.nd)
                    else:
                        self.f_d = numpy.zeros((self.np, self.nd),
                                               dtype=numpy.float64)
                        self.f_p = numpy.zeros((self.np, self.nd),
                                               dtype=numpy.float64)
                        self.f_v = numpy.zeros((self.np, self.nd),
                                               dtype=numpy.float64)
                        self.f_sum = numpy.zeros((self.np, self.nd),
                                                 dtype=numpy.float64)

                elif self.version >= 2.0 and self.cfd_solver == 1:

                    self.tolerance = numpy.fromfile(fh, dtype=numpy.float64,
                                                    count=1)
                    self.maxiter = numpy.fromfile(fh, dtype=numpy.uint32,
                                                  count=1)
                    self.ndem = numpy.fromfile(fh, dtype=numpy.uint32, count=1)
                    self.c_phi = numpy.fromfile(fh, dtype=numpy.float64,
                                                count=1)
                    self.f_p = numpy.empty_like(self.x)
                    for i in numpy.arange(self.np):
                        self.f_p[i, :] = numpy.fromfile(fh, dtype=numpy.float64,
                                                        count=self.nd)
                    self.beta_f = numpy.fromfile(fh, dtype=numpy.float64,
                                                 count=1)
                    self.k_c = numpy.fromfile(fh, dtype=numpy.float64, count=1)

            if self.version >= 1.02:
                self.color = numpy.fromfile(fh, dtype=numpy.int32,
                                            count=self.np)
            else:
                self.color = numpy.zeros(self.np, dtype=numpy.int32)

        finally:
            self.version[0] = VERSION
            if fh is not None:
                fh.close()

    def writebin(self, folder="../input/", verbose=True):
        '''
        Writes a ``sphere`` binary file to the ``../input/`` folder by default.
        The file name will be in the format ``<self.sid>.bin``.

        See also :func:`readbin()`.

        :param folder: The folder where to place the output binary file
        :type folder: str
        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        '''
        fh = None
        try:
            targetbin = folder + "/" + self.sid + ".bin"
            if verbose:
                print("Output file: {0}".format(targetbin))

            fh = open(targetbin, "wb")

            # Write the current version number
            fh.write(self.version.astype(numpy.float64))

            # Write the number of dimensions and particles
            fh.write(numpy.array(self.nd).astype(numpy.int32))
            fh.write(numpy.array(self.np).astype(numpy.uint32))

            # Write the time variables
            fh.write(self.time_dt.astype(numpy.float64))
            fh.write(self.time_current.astype(numpy.float64))
            fh.write(self.time_total.astype(numpy.float64))
            fh.write(self.time_file_dt.astype(numpy.float64))
            fh.write(self.time_step_count.astype(numpy.uint32))

            # Read remaining data from binary
            fh.write(self.origo.astype(numpy.float64))
            fh.write(self.L.astype(numpy.float64))
            fh.write(self.num.astype(numpy.uint32))
            fh.write(self.periodic.astype(numpy.uint32))
            fh.write(self.adaptive.astype(numpy.uint32))

            # Per-particle vectors
            for i in numpy.arange(self.np):
                fh.write(self.x[i, :].astype(numpy.float64))
                fh.write(self.radius[i].astype(numpy.float64))

            if self.np > 0:
                fh.write(self.xyzsum.astype(numpy.float64))

            for i in numpy.arange(self.np):
                fh.write(self.vel[i, :].astype(numpy.float64))
                fh.write(self.fixvel[i].astype(numpy.float64))

            if self.np > 0:
                fh.write(self.force.astype(numpy.float64))

                fh.write(self.angpos.astype(numpy.float64))
                fh.write(self.angvel.astype(numpy.float64))
                fh.write(self.torque.astype(numpy.float64))

                # Per-particle single-value parameters
                fh.write(self.es_dot.astype(numpy.float64))
                fh.write(self.es.astype(numpy.float64))
                fh.write(self.ev_dot.astype(numpy.float64))
                fh.write(self.ev.astype(numpy.float64))
                fh.write(self.p.astype(numpy.float64))

            fh.write(self.g.astype(numpy.float64))
            fh.write(self.k_n.astype(numpy.float64))
            fh.write(self.k_t.astype(numpy.float64))
            fh.write(self.k_r.astype(numpy.float64))
            fh.write(self.E.astype(numpy.float64))
            fh.write(self.gamma_n.astype(numpy.float64))
            fh.write(self.gamma_t.astype(numpy.float64))
            fh.write(self.gamma_r.astype(numpy.float64))
            fh.write(self.mu_s.astype(numpy.float64))
            fh.write(self.mu_d.astype(numpy.float64))
            fh.write(self.mu_r.astype(numpy.float64))
            fh.write(self.gamma_wn.astype(numpy.float64))
            fh.write(self.gamma_wt.astype(numpy.float64))
            fh.write(self.mu_ws.astype(numpy.float64))
            fh.write(self.mu_wd.astype(numpy.float64))
            fh.write(self.rho.astype(numpy.float64))
            fh.write(self.contactmodel.astype(numpy.uint32))
            fh.write(self.kappa.astype(numpy.float64))
            fh.write(self.db.astype(numpy.float64))
            fh.write(self.V_b.astype(numpy.float64))

            fh.write(numpy.array(self.nw).astype(numpy.uint32))
            for i in numpy.arange(self.nw):
                fh.write(self.wmode[i].astype(numpy.int32))
            for i in numpy.arange(self.nw):
                fh.write(self.w_n[i, :].astype(numpy.float64))
                fh.write(self.w_x[i].astype(numpy.float64))

            for i in numpy.arange(self.nw):
                fh.write(self.w_m[i].astype(numpy.float64))
                fh.write(self.w_vel[i].astype(numpy.float64))
                fh.write(self.w_force[i].astype(numpy.float64))
                fh.write(self.w_sigma0[i].astype(numpy.float64))
            fh.write(self.w_sigma0_A.astype(numpy.float64))
            fh.write(self.w_sigma0_f.astype(numpy.float64))
            fh.write(self.w_tau_x.astype(numpy.float64))

            fh.write(self.lambda_bar.astype(numpy.float64))
            fh.write(numpy.array(self.nb0).astype(numpy.uint32))
            fh.write(self.sigma_b.astype(numpy.float64))
            fh.write(self.tau_b.astype(numpy.float64))
            for i in numpy.arange(self.nb0):
                fh.write(self.bonds[i, 0].astype(numpy.uint32))
                fh.write(self.bonds[i, 1].astype(numpy.uint32))
            fh.write(self.bonds_delta_n.astype(numpy.float64))
            fh.write(self.bonds_delta_t.astype(numpy.float64))
            fh.write(self.bonds_omega_n.astype(numpy.float64))
            fh.write(self.bonds_omega_t.astype(numpy.float64))

            if self.fluid:

                fh.write(self.cfd_solver.astype(numpy.int32))
                fh.write(self.mu.astype(numpy.float64))
                for z in numpy.arange(self.num[2]):
                    for y in numpy.arange(self.num[1]):
                        for x in numpy.arange(self.num[0]):
                            fh.write(self.v_f[x, y, z, 0].astype(numpy.float64))
                            fh.write(self.v_f[x, y, z, 1].astype(numpy.float64))
                            fh.write(self.v_f[x, y, z, 2].astype(numpy.float64))
                            fh.write(self.p_f[x, y, z].astype(numpy.float64))
                            fh.write(self.phi[x, y, z].astype(numpy.float64))
                            fh.write(self.dphi[x, y, z].astype(numpy.float64)*
                                     self.time_dt*self.ndem)

                fh.write(self.rho_f.astype(numpy.float64))
                fh.write(self.p_mod_A.astype(numpy.float64))
                fh.write(self.p_mod_f.astype(numpy.float64))
                fh.write(self.p_mod_phi.astype(numpy.float64))

                if self.cfd_solver[0] == 1:  # Sides only adjustable with Darcy
                    fh.write(self.bc_xn.astype(numpy.int32))
                    fh.write(self.bc_xp.astype(numpy.int32))
                    fh.write(self.bc_yn.astype(numpy.int32))
                    fh.write(self.bc_yp.astype(numpy.int32))

                fh.write(self.bc_bot.astype(numpy.int32))
                fh.write(self.bc_top.astype(numpy.int32))
                fh.write(self.free_slip_bot.astype(numpy.int32))
                fh.write(self.free_slip_top.astype(numpy.int32))
                fh.write(self.bc_bot_flux.astype(numpy.float64))
                fh.write(self.bc_top_flux.astype(numpy.float64))

                for z in numpy.arange(self.num[2]):
                    for y in numpy.arange(self.num[1]):
                        for x in numpy.arange(self.num[0]):
                            fh.write(self.p_f_constant[x, y, z].astype(
                                numpy.int32))

                # Navier Stokes
                if self.cfd_solver[0] == 0:
                    fh.write(self.gamma.astype(numpy.float64))
                    fh.write(self.theta.astype(numpy.float64))
                    fh.write(self.beta.astype(numpy.float64))
                    fh.write(self.tolerance.astype(numpy.float64))
                    fh.write(self.maxiter.astype(numpy.uint32))
                    fh.write(self.ndem.astype(numpy.uint32))

                    fh.write(self.c_phi.astype(numpy.float64))
                    fh.write(self.c_v.astype(numpy.float64))
                    fh.write(self.dt_dem_fac.astype(numpy.float64))

                    for i in numpy.arange(self.np):
                        fh.write(self.f_d[i, :].astype(numpy.float64))
                    for i in numpy.arange(self.np):
                        fh.write(self.f_p[i, :].astype(numpy.float64))
                    for i in numpy.arange(self.np):
                        fh.write(self.f_v[i, :].astype(numpy.float64))
                    for i in numpy.arange(self.np):
                        fh.write(self.f_sum[i, :].astype(numpy.float64))

                # Darcy
                elif self.cfd_solver[0] == 1:

                    fh.write(self.tolerance.astype(numpy.float64))
                    fh.write(self.maxiter.astype(numpy.uint32))
                    fh.write(self.ndem.astype(numpy.uint32))
                    fh.write(self.c_phi.astype(numpy.float64))
                    for i in numpy.arange(self.np):
                        fh.write(self.f_p[i, :].astype(numpy.float64))
                    fh.write(self.beta_f.astype(numpy.float64))
                    fh.write(self.k_c.astype(numpy.float64))

                else:
                    raise Exception('Value of cfd_solver not understood (' + \
                            str(self.cfd_solver[0]) + ')')


            fh.write(self.color.astype(numpy.int32))

        finally:
            if fh is not None:
                fh.close()

    def writeVTKall(self, cell_centered=True, verbose=True, forces=False):
        '''
        Writes a VTK file for each simulation output file with particle
        information and the fluid grid to the ``../output/`` folder by default.
        The file name will be in the format ``<self.sid>.vtu`` and
        ``fluid-<self.sid>.vti``. The vtu files can be used to visualize the
        particles, and the vti files for visualizing the fluid in ParaView.

        After opening the vtu files, the particle fields will show up in the
        "Properties" list. Press "Apply" to import all fields into the ParaView
        session. The particles are visualized by selecting the imported data in
        the "Pipeline Browser". Afterwards, click the "Glyph" button in the
        "Common" toolbar, or go to the "Filters" menu, and press "Glyph" from
        the "Common" list. Choose "Sphere" as the "Glyph Type", set "Radius" to
        1.0, choose "scalar" as the "Scale Mode". Check the "Edit" checkbox, and
        set the "Set Scale Factor" to 1.0. The field "Maximum Number of Points"
        may be increased if the number of particles exceed the default value.
        Finally press "Apply", and the particles will appear in the main window.

        The sphere resolution may be adjusted ("Theta resolution", "Phi
        resolution") to increase the quality and the computational requirements
        of the rendering.

        The fluid grid is visualized by opening the vti files, and pressing
        "Apply" to import all fluid field properties. To visualize the scalar
        fields, such as the pressure, the porosity, the porosity change or the
        velocity magnitude, choose "Surface" or "Surface With Edges" as the
        "Representation". Choose the desired property as the "Coloring" field.
        It may be desirable to show the color bar by pressing the "Show" button,
        and "Rescale" to fit the color range limits to the current file. The
        coordinate system can be displayed by checking the "Show Axis" field.
        All adjustments by default require the "Apply" button to be pressed
        before regenerating the view.

        The fluid vector fields (e.g. the fluid velocity) can be visualizing by
        e.g. arrows. To do this, select the fluid data in the "Pipeline
        Browser". Press "Glyph" from the "Common" toolbar, or go to the
        "Filters" mennu, and press "Glyph" from the "Common" list. Make sure
        that "Arrow" is selected as the "Glyph type", and "Velocity" as the
        "Vectors" value. Adjust the "Maximum Number of Points" to be at least as
        big as the number of fluid cells in the grid. Press "Apply" to visualize
        the arrows.

        If several data files are generated for the same simulation (e.g. using
        the :func:`writeVTKall()` function), it is able to step the
        visualization through time by using the ParaView controls.

        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        :param cell_centered: Write fluid values to cell centered positions
            (default=true)
        :type cell_centered: bool
        :param forces: Write contact force files (slow) (default=False)
        :type forces: bool
        '''
        lastfile = status(self.sid)
        sb = sim(fluid=self.fluid)
        for i in range(lastfile+1):
            fn = "../output/{0}.output{1:0=5}.bin".format(self.sid, i)

            # check if output VTK file exists and if it is newer than spherebin
            fn_vtk = "../output/{0}.{1:0=5}.vtu".format(self.sid, i)
            if os.path.isfile(fn_vtk) and \
                (os.path.getmtime(fn) < os.path.getmtime(fn_vtk)):
                if verbose:
                    print('skipping ' + fn_vtk +
                          ': file exists and is newer than ' + fn)
                if self.fluid:
                    fn_vtk = "../output/fluid-{0}.{1:0=5}.vti" \
                             .format(self.sid, i)
                    if os.path.isfile(fn_vtk) and \
                        (os.path.getmtime(fn) < os.path.getmtime(fn_vtk)):
                        if verbose:
                            print('skipping ' + fn_vtk +
                                  ': file exists and is newer than ' + fn)
                        continue
                else:
                    continue

            sb.sid = self.sid + ".{:0=5}".format(i)
            sb.readbin(fn, verbose=False)
            if sb.np > 0:
                if i == 0 or i == lastfile:
                    if i == lastfile:
                        if verbose:
                            print("\tto")
                    sb.writeVTK(verbose=verbose)
                    if forces:
                        sb.findContactStresses()
                        sb.writeVTKforces(verbose=verbose)
                else:
                    sb.writeVTK(verbose=False)
                    if forces:
                        sb.findContactStresses()
                        sb.writeVTKforces(verbose=False)
            if self.fluid:
                if i == 0 or i == lastfile:
                    if i == lastfile:
                        if verbose:
                            print("\tto")
                    sb.writeFluidVTK(verbose=verbose,
                                     cell_centered=cell_centered)
                else:
                    sb.writeFluidVTK(verbose=False, cell_centered=cell_centered)

    def writeVTK(self, folder='../output/', verbose=True):
        '''
        Writes a VTK file with particle information to the ``../output/`` folder
        by default. The file name will be in the format ``<self.sid>.vtu``.
        The vtu files can be used to visualize the particles in ParaView.

        After opening the vtu files, the particle fields will show up in the
        "Properties" list. Press "Apply" to import all fields into the ParaView
        session. The particles are visualized by selecting the imported data in
        the "Pipeline Browser". Afterwards, click the "Glyph" button in the
        "Common" toolbar, or go to the "Filters" menu, and press "Glyph" from
        the "Common" list. Choose "Sphere" as the "Glyph Type", choose "scalar"
        as the "Scale Mode". Check the "Edit" checkbox, and set the "Set Scale
        Factor" to 1.0. The field "Maximum Number of Points" may be increased if
        the number of particles exceed the default value. Finally press "Apply",
        and the particles will appear in the main window.

        The sphere resolution may be adjusted ("Theta resolution", "Phi
        resolution") to increase the quality and the computational requirements
        of the rendering. All adjustments by default require the "Apply" button
        to be pressed before regenerating the view.

        If several vtu files are generated for the same simulation (e.g. using
        the :func:`writeVTKall()` function), it is able to step the
        visualization through time by using the ParaView controls.

        :param folder: The folder where to place the output binary file (default
            (default='../output/')
        :type folder: str
        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        '''

        fh = None
        try:
            targetbin = folder + '/' + self.sid + '.vtu' # unstructured grid
            if verbose:
                print('Output file: ' + targetbin)

            fh = open(targetbin, 'w')

            # the VTK data file format is documented in
            # http://www.vtk.org/VTK/img/file-formats.pdf

            fh.write('<?xml version="1.0"?>\n') # XML header
            fh.write('<VTKFile type="UnstructuredGrid" version="0.1" '
                     + 'byte_order="LittleEndian">\n') # VTK header
            fh.write('  <UnstructuredGrid>\n')
            fh.write('    <Piece NumberOfPoints="%d" NumberOfCells="0">\n' \
                     % (self.np))

            # Coordinates for each point (positions)
            fh.write('      <Points>\n')
            fh.write('        <DataArray name="Position [m]" type="Float32" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % (self.x[i, 0], self.x[i, 1], self.x[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')
            fh.write('      </Points>\n')

            ### Data attributes
            fh.write('      <PointData Scalars="Diameter [m]" Vectors="vector">\n')

            # Radii
            fh.write('        <DataArray type="Float32" Name="Diameter" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.radius[i]*2.0))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Displacements (xyzsum)
            fh.write('        <DataArray type="Float32" Name="Displacement [m]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % \
                         (self.xyzsum[i, 0], self.xyzsum[i, 1], self.xyzsum[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Velocity
            fh.write('        <DataArray type="Float32" Name="Velocity [m/s]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % \
                         (self.vel[i, 0], self.vel[i, 1], self.vel[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            if self.fluid:

                if self.cfd_solver == 0:  # Navier Stokes
                    # Fluid interaction force
                    fh.write('        <DataArray type="Float32" '
                             + 'Name="Fluid force total [N]" '
                             + 'NumberOfComponents="3" format="ascii">\n')
                    fh.write('          ')
                    for i in range(self.np):
                        fh.write('%f %f %f ' % \
                                 (self.f_sum[i, 0], self.f_sum[i, 1], \
                                  self.f_sum[i, 2]))
                    fh.write('\n')
                    fh.write('        </DataArray>\n')

                    # Fluid drag force
                    fh.write('        <DataArray type="Float32" '
                             + 'Name="Fluid drag force [N]" '
                             + 'NumberOfComponents="3" format="ascii">\n')
                    fh.write('          ')
                    for i in range(self.np):
                        fh.write('%f %f %f ' % \
                                 (self.f_d[i, 0],
                                  self.f_d[i, 1],
                                  self.f_d[i, 2]))
                    fh.write('\n')
                    fh.write('        </DataArray>\n')

                # Fluid pressure force
                fh.write('        <DataArray type="Float32" '
                         + 'Name="Fluid pressure force [N]" '
                         + 'NumberOfComponents="3" format="ascii">\n')
                fh.write('          ')
                for i in range(self.np):
                    fh.write('%f %f %f ' % \
                             (self.f_p[i, 0], self.f_p[i, 1], self.f_p[i, 2]))
                fh.write('\n')
                fh.write('        </DataArray>\n')

                if self.cfd_solver == 0:  # Navier Stokes
                    # Fluid viscous force
                    fh.write('        <DataArray type="Float32" '
                             + 'Name="Fluid viscous force [N]" '
                             + 'NumberOfComponents="3" format="ascii">\n')
                    fh.write('          ')
                    for i in range(self.np):
                        fh.write('%f %f %f ' % \
                                 (self.f_v[i, 0],
                                  self.f_v[i, 1],
                                  self.f_v[i, 2]))
                    fh.write('\n')
                    fh.write('        </DataArray>\n')

            # fixvel
            fh.write('        <DataArray type="Float32" Name="FixedVel" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.fixvel[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Force
            fh.write('        <DataArray type="Float32" Name="Force [N]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % (self.force[i, 0],
                                        self.force[i, 1],
                                        self.force[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Angular Position
            fh.write('        <DataArray type="Float32" Name="Angular position'
                     + '[rad]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % (self.angpos[i, 0],
                                        self.angpos[i, 1],
                                        self.angpos[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Angular Velocity
            fh.write('        <DataArray type="Float32" Name="Angular velocity'
                     + ' [rad/s]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % (self.angvel[i, 0],
                                        self.angvel[i, 1],
                                        self.angvel[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Torque
            fh.write('        <DataArray type="Float32" Name="Torque [Nm]" '
                     + 'NumberOfComponents="3" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f %f %f ' % (self.torque[i, 0],
                                        self.torque[i, 1],
                                        self.torque[i, 2]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Shear energy rate
            fh.write('        <DataArray type="Float32" Name="Shear Energy '
                     + 'Rate [J/s]" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.es_dot[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Shear energy
            fh.write('        <DataArray type="Float32" Name="Shear Energy [J]"'
                     + ' format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.es[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Viscous energy rate
            fh.write('        <DataArray type="Float32" '
                     + 'Name="Viscous Energy Rate [J/s]" format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.ev_dot[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Shear energy
            fh.write('        <DataArray type="Float32" '
                     + 'Name="Viscous Energy [J]" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.ev[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Pressure
            fh.write('        <DataArray type="Float32" Name="Pressure [Pa]" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%f ' % (self.p[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Color
            fh.write('        <DataArray type="Int32" Name="Type color" '
                     + 'format="ascii">\n')
            fh.write('          ')
            for i in range(self.np):
                fh.write('%d ' % (self.color[i]))
            fh.write('\n')
            fh.write('        </DataArray>\n')

            # Footer
            fh.write('      </PointData>\n')
            fh.write('      <Cells>\n')
            fh.write('        <DataArray type="Int32" Name="connectivity" '
                     + 'format="ascii">\n')
            fh.write('        </DataArray>\n')
            fh.write('        <DataArray type="Int32" Name="offsets" '
                     + 'format="ascii">\n')
            fh.write('        </DataArray>\n')
            fh.write('        <DataArray type="UInt8" Name="types" '
                     + 'format="ascii">\n')
            fh.write('        </DataArray>\n')
            fh.write('      </Cells>\n')
            fh.write('    </Piece>\n')
            fh.write('  </UnstructuredGrid>\n')
            fh.write('</VTKFile>')

        finally:
            if fh is not None:
                fh.close()

    def writeVTKforces(self, folder='../output/', verbose=True):
        '''
        Writes a VTK file with particle-interaction information to the
        ``../output/`` folder by default. The file name will be in the format
        ``<self.sid>.vtp``.  The vtp files can be used to visualize the
        particle interactions in ParaView.  First use the "Cell Data to Point
        Data" filter, and afterwards show the contact network with the "Tube"
        filter.

        :param folder: The folder where to place the output file (default
            (default='../output/')
        :type folder: str
        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        '''

        if not py_vtk:
            print('Error: vtk module not found, cannot writeVTKforces.')
            return

        filename = folder + '/forces-' + self.sid + '.vtp' # Polygon data

        # points mark the particle centers
        points = vtk.vtkPoints()

        # lines mark the particle connectivity
        lines = vtk.vtkCellArray()

        # colors
        #colors = vtk.vtkUnsignedCharArray()
        #colors.SetNumberOfComponents(3)
        #colors.SetName('Colors')
        #colors.SetNumberOfTuples(self.overlaps.size)

        # scalars
        forces = vtk.vtkDoubleArray()
        forces.SetName("Force [N]")
        forces.SetNumberOfComponents(1)
        #forces.SetNumberOfTuples(self.overlaps.size)
        forces.SetNumberOfValues(self.overlaps.size)

        stresses = vtk.vtkDoubleArray()
        stresses.SetName("Stress [Pa]")
        stresses.SetNumberOfComponents(1)
        stresses.SetNumberOfValues(self.overlaps.size)

        for i in numpy.arange(self.overlaps.size):
            points.InsertNextPoint(self.x[self.pairs[0, i], :])
            points.InsertNextPoint(self.x[self.pairs[1, i], :])
            line = vtk.vtkLine()
            line.GetPointIds().SetId(0, 2*i)      # index of particle 1
            line.GetPointIds().SetId(1, 2*i + 1)  # index of particle 2
            lines.InsertNextCell(line)
            #colors.SetTupleValue(i, [100, 100, 100])
            forces.SetValue(i, self.f_n_magn[i])
            stresses.SetValue(i, self.sigma_contacts[i])

        # initalize VTK data structure
        polydata = vtk.vtkPolyData()

        polydata.SetPoints(points)
        polydata.SetLines(lines)
        #polydata.GetCellData().SetScalars(colors)
        #polydata.GetCellData().SetScalars(forces)  # default scalar
        polydata.GetCellData().SetScalars(forces)  # default scalar
        #polydata.GetCellData().AddArray(forces)
        polydata.GetCellData().AddArray(stresses)
        #polydata.GetPointData().AddArray(stresses)
        #polydata.GetPointData().SetScalars(stresses)  # default scalar

        # write VTK XML image data file
        writer = vtk.vtkXMLPolyDataWriter()
        writer.SetFileName(filename)
        if vtk.VTK_MAJOR_VERSION <= 5:
            writer.SetInput(polydata)
        else:
            writer.SetInputData(polydata)
        writer.Write()
        #writer.Update()
        if verbose:
            print('Output file: ' + filename)


    def writeFluidVTK(self, folder='../output/', cell_centered=True,
                      verbose=True):
        '''
        Writes a VTK file for the fluid grid to the ``../output/`` folder by
        default. The file name will be in the format ``fluid-<self.sid>.vti``.
        The vti files can be used for visualizing the fluid in ParaView.

        The scalars (pressure, porosity, porosity change) and the velocity
        vectors are either placed in a grid where the grid corners correspond to
        the computational grid center (cell_centered=False). This results in a
        grid that doesn't appears to span the simulation domain, and values are
        smoothly interpolated on the cell faces. Alternatively, the
        visualization grid is equal to the computational grid, and cells face
        colors are not interpolated (cell_centered=True, default behavior).

        The fluid grid is visualized by opening the vti files, and pressing
        "Apply" to import all fluid field properties. To visualize the scalar
        fields, such as the pressure, the porosity, the porosity change or the
        velocity magnitude, choose "Surface" or "Surface With Edges" as the
        "Representation". Choose the desired property as the "Coloring" field.
        It may be desirable to show the color bar by pressing the "Show" button,
        and "Rescale" to fit the color range limits to the current file. The
        coordinate system can be displayed by checking the "Show Axis" field.
        All adjustments by default require the "Apply" button to be pressed
        before regenerating the view.

        The fluid vector fields (e.g. the fluid velocity) can be visualizing by
        e.g. arrows. To do this, select the fluid data in the "Pipeline
        Browser". Press "Glyph" from the "Common" toolbar, or go to the
        "Filters" mennu, and press "Glyph" from the "Common" list. Make sure
        that "Arrow" is selected as the "Glyph type", and "Velocity" as the
        "Vectors" value. Adjust the "Maximum Number of Points" to be at least as
        big as the number of fluid cells in the grid. Press "Apply" to visualize
        the arrows.

        To visualize the cell-centered data with smooth interpolation, and in
        order to visualize fluid vector fields, the cell-centered mesh is
        selected in the "Pipeline Browser", and is filtered using "Filters" ->
        "Alphabetical" -> "Cell Data to Point Data".

        If several data files are generated for the same simulation (e.g. using
        the :func:`writeVTKall()` function), it is able to step the
        visualization through time by using the ParaView controls.

        :param folder: The folder where to place the output binary file (default
            (default='../output/')
        :type folder: str
        :param cell_centered: put scalars and vectors at cell centers (True) or
            cell corners (False), (default=True)
        :type cell_centered: bool
        :param verbose: Show diagnostic information (default=True)
        :type verbose: bool
        '''
        if not py_vtk:
            print('Error: vtk module not found, cannot writeFluidVTK.')
            return

        filename = folder + '/fluid-' + self.sid + '.vti' # image grid

        # initalize VTK data structure
        grid = vtk.vtkImageData()
        dx = (self.L-self.origo)/self.num   # cell center spacing
        if cell_centered:
            grid.SetOrigin(self.origo)
        else:
            grid.SetOrigin(self.origo + 0.5*dx)
        grid.SetSpacing(dx)
        if cell_centered:
            grid.SetDimensions(self.num + 1) # no. of points in each direction
        else:
            grid.SetDimensions(self.num)    # no. of points in each direction

        # array of scalars: hydraulic pressures
        pres = vtk.vtkDoubleArray()
        pres.SetName("Pressure [Pa]")
        pres.SetNumberOfComponents(1)
        if cell_centered:
            pres.SetNumberOfTuples(grid.GetNumberOfCells())
        else:
            pres.SetNumberOfTuples(grid.GetNumberOfPoints())

        # array of vectors: hydraulic velocities
        vel = vtk.vtkDoubleArray()
        vel.SetName("Velocity [m/s]")
        vel.SetNumberOfComponents(3)
        if cell_centered:
            vel.SetNumberOfTuples(grid.GetNumberOfCells())
        else:
            vel.SetNumberOfTuples(grid.GetNumberOfPoints())

        # array of scalars: porosities
        poros = vtk.vtkDoubleArray()
        poros.SetName("Porosity [-]")
        poros.SetNumberOfComponents(1)
        if cell_centered:
            poros.SetNumberOfTuples(grid.GetNumberOfCells())
        else:
            poros.SetNumberOfTuples(grid.GetNumberOfPoints())

        # array of scalars: porosity change
        dporos = vtk.vtkDoubleArray()
        dporos.SetName("Porosity change [1/s]")
        dporos.SetNumberOfComponents(1)
        if cell_centered:
            dporos.SetNumberOfTuples(grid.GetNumberOfCells())
        else:
            dporos.SetNumberOfTuples(grid.GetNumberOfPoints())

        # array of scalars: Reynold's number
        Re_values = self.ReynoldsNumber()
        Re = vtk.vtkDoubleArray()
        Re.SetName("Reynolds number [-]")
        Re.SetNumberOfComponents(1)
        if cell_centered:
            Re.SetNumberOfTuples(grid.GetNumberOfCells())
        else:
            Re.SetNumberOfTuples(grid.GetNumberOfPoints())

        # Find permeabilities if the Darcy solver is used
        if self.cfd_solver[0] == 1:
            self.findPermeabilities()
            k = vtk.vtkDoubleArray()
            k.SetName("Permeability [m*m]")
            k.SetNumberOfComponents(1)
            if cell_centered:
                k.SetNumberOfTuples(grid.GetNumberOfCells())
            else:
                k.SetNumberOfTuples(grid.GetNumberOfPoints())

            self.findHydraulicConductivities()
            K = vtk.vtkDoubleArray()
            K.SetName("Conductivity [m/s]")
            K.SetNumberOfComponents(1)
            if cell_centered:
                K.SetNumberOfTuples(grid.GetNumberOfCells())
            else:
                K.SetNumberOfTuples(grid.GetNumberOfPoints())

            p_f_constant = vtk.vtkDoubleArray()
            p_f_constant.SetName("Constant pressure [-]")
            p_f_constant.SetNumberOfComponents(1)
            if cell_centered:
                p_f_constant.SetNumberOfTuples(grid.GetNumberOfCells())
            else:
                p_f_constant.SetNumberOfTuples(grid.GetNumberOfPoints())

        # insert values
        for z in range(self.num[2]):
            for y in range(self.num[1]):
                for x in range(self.num[0]):
                    idx = x + self.num[0]*y + self.num[0]*self.num[1]*z
                    pres.SetValue(idx, self.p_f[x, y, z])
                    vel.SetTuple(idx, self.v_f[x, y, z, :])
                    poros.SetValue(idx, self.phi[x, y, z])
                    dporos.SetValue(idx, self.dphi[x, y, z])
                    Re.SetValue(idx, Re_values[x, y, z])
                    if self.cfd_solver[0] == 1:
                        k.SetValue(idx, self.k[x, y, z])
                        K.SetValue(idx, self.K[x, y, z])
                        p_f_constant.SetValue(idx, self.p_f_constant[x, y, z])

        # add pres array to grid
        if cell_centered:
            grid.GetCellData().AddArray(pres)
            grid.GetCellData().AddArray(vel)
            grid.GetCellData().AddArray(poros)
            grid.GetCellData().AddArray(dporos)
            grid.GetCellData().AddArray(Re)
            if self.cfd_solver[0] == 1:
                grid.GetCellData().AddArray(k)
                grid.GetCellData().AddArray(K)
                grid.GetCellData().AddArray(p_f_constant)
        else:
            grid.GetPointData().AddArray(pres)
            grid.GetPointData().AddArray(vel)
            grid.GetPointData().AddArray(poros)
            grid.GetPointData().AddArray(dporos)
            grid.GetPointData().AddArray(Re)
            if self.cfd_solver[0] == 1:
                grid.GetPointData().AddArray(k)
                grid.GetPointData().AddArray(K)
                grid.GetPointData().AddArray(p_f_constant)

        # write VTK XML image data file
        writer = vtk.vtkXMLImageDataWriter()
        writer.SetFileName(filename)
        #writer.SetInput(grid) # deprecated from VTK 6
        writer.SetInputData(grid)
        writer.Update()
        if verbose:
            print('Output file: ' + filename)

    def show(self, coloring=numpy.array([]), resolution=6):
        '''
        Show a rendering of all particles in a window.

        :param coloring: Color the particles from red to white to blue according
            to the values in this array.
        :type coloring: numpy.array
        :param resolution: The resolution of the rendered spheres. Larger values
            increase the performance requirements.
        :type resolution: int
        '''

        if not py_vtk:
            print('Error: vtk module not found, cannot show scene.')
            return

        # create a rendering window and renderer
        ren = vtk.vtkRenderer()
        renWin = vtk.vtkRenderWindow()
        renWin.AddRenderer(ren)

        # create a renderwindowinteractor
        iren = vtk.vtkRenderWindowInteractor()
        iren.SetRenderWindow(renWin)

        if coloring.any():
            #min_value = numpy.min(coloring)
            max_value = numpy.max(coloring)
            #min_rgb = numpy.array([50, 50, 50])
            #max_rgb = numpy.array([255, 255, 255])
            #def color(value):
                #return (max_rgb - min_rgb) * (value - min_value)

            def red(ratio):
                return numpy.fmin(1.0, 0.209*ratio**3. - 2.49*ratio**2. + 3.0*ratio
                                  + 0.0109)
            def green(ratio):
                return numpy.fmin(1.0, -2.44*ratio**2. + 2.15*ratio + 0.369)
            def blue(ratio):
                return numpy.fmin(1.0, -2.21*ratio**2. + 1.61*ratio + 0.573)

        for i in numpy.arange(self.np):

            # create source
            source = vtk.vtkSphereSource()
            source.SetCenter(self.x[i, :])
            source.SetRadius(self.radius[i])
            source.SetThetaResolution(resolution)
            source.SetPhiResolution(resolution)

            # mapper
            mapper = vtk.vtkPolyDataMapper()
            if vtk.VTK_MAJOR_VERSION <= 5:
                mapper.SetInput(source.GetOutput())
            else:
                mapper.SetInputConnection(source.GetOutputPort())

            # actor
            actor = vtk.vtkActor()
            actor.SetMapper(mapper)

            # color
            if coloring.any():
                ratio = coloring[i]/max_value
                r, g, b = red(ratio), green(ratio), blue(ratio)
                actor.GetProperty().SetColor(r, g, b)

            # assign actor to the renderer
            ren.AddActor(actor)

        ren.SetBackground(0.3, 0.3, 0.3)

        # enable user interface interactor
        iren.Initialize()
        renWin.Render()
        iren.Start()

    def readfirst(self, verbose=True):
        '''
        Read the first output file from the ``../output/`` folder, corresponding
        to the object simulation id (``self.sid``).

        :param verbose: Display diagnostic information (default=True)
        :type verbose: bool

        See also :func:`readbin()`, :func:`readlast()`, :func:`readsecond`, and
        :func:`readstep`.
        '''

        fn = '../output/' + self.sid + '.output00000.bin'
        self.readbin(fn, verbose)

    def readsecond(self, verbose=True):
        '''
        Read the second output file from the ``../output/`` folder,
        corresponding to the object simulation id (``self.sid``).

        :param verbose: Display diagnostic information (default=True)
        :type verbose: bool

        See also :func:`readbin()`, :func:`readfirst()`, :func:`readlast()`,
        and :func:`readstep`.
        '''
        fn = '../output/' + self.sid + '.output00001.bin'
        self.readbin(fn, verbose)

    def readstep(self, step, verbose=True):
        '''
        Read a output file from the ``../output/`` folder, corresponding
        to the object simulation id (``self.sid``).

        :param step: The output file number to read, starting from 0.
        :type step: int
        :param verbose: Display diagnostic information (default=True)
        :type verbose: bool

        See also :func:`readbin()`, :func:`readfirst()`, :func:`readlast()`,
        and :func:`readsecond`.
        '''
        fn = "../output/{0}.output{1:0=5}.bin".format(self.sid, step)
        self.readbin(fn, verbose)

    def readlast(self, verbose=True):
        '''
        Read the last output file from the ``../output/`` folder, corresponding
        to the object simulation id (``self.sid``).

        :param verbose: Display diagnostic information (default=True)
        :type verbose: bool

        See also :func:`readbin()`, :func:`readfirst()`, :func:`readsecond`, and
        :func:`readstep`.
        '''
        lastfile = status(self.sid)
        fn = "../output/{0}.output{1:0=5}.bin".format(self.sid, lastfile)
        self.readbin(fn, verbose)

    def readTime(self, time, verbose=True):
        '''
        Read the output file most closely corresponding to the time given as an
        argument.

        :param time: The desired current time [s]
        :type time: float

        See also :func:`readbin()`, :func:`readfirst()`, :func:`readsecond`, and
        :func:`readstep`.
        '''

        self.readfirst(verbose=False)
        t_first = self.currentTime()
        n_first = self.time_step_count[0]

        self.readlast(verbose=False)
        t_last = self.currentTime()
        n_last = self.time_step_count[0]

        if time < t_first or time > t_last:
            raise Exception('Error: The specified time {} s is outside the ' +
                            'range of output files [{}; {}] s.'
                            .format(time, t_first, t_last))

        dt_dn = (t_last - t_first)/(n_last - n_first)
        step = int((time - t_first)/dt_dn) + n_first + 1
        self.readstep(step, verbose=verbose)

    def generateRadii(self, psd='logn', mean=440e-6, variance=8.8e-9,
                      histogram=False):
        '''
        Draw random particle radii from a selected probability distribution.
        The larger the variance of radii is, the slower the computations will
        run. The reason is two-fold: The smallest particle dictates the time
        step length, where smaller particles cause shorter time steps. At the
        same time, the largest particle determines the sorting cell size, where
        larger particles cause larger cells. Larger cells are likely to contain
        more particles, causing more contact checks.

        :param psd: The particle side distribution. One possible value is
            ``logn``, which is a log-normal probability distribution, suitable
            for approximating well-sorted, coarse sediments. The other possible
            value is ``uni``, which is a uniform distribution from
            ``mean - variance`` to ``mean + variance``.
        :type psd: str
        :param mean: The mean radius [m] (default=440e-6 m)
        :type mean: float
        :param variance: The variance in the probability distribution
            [m].
        :type variance: float

        See also: :func:`generateBimodalRadii()`.
        '''

        if psd == 'logn': # Log-normal probability distribution
            mu = math.log((mean**2)/math.sqrt(variance+mean**2))
            sigma = math.sqrt(math.log(variance/(mean**2)+1))
            self.radius = numpy.random.lognormal(mu, sigma, self.np)
        elif psd == 'uni':  # Uniform distribution
            radius_min = mean - variance
            radius_max = mean + variance
            self.radius = numpy.random.uniform(radius_min, radius_max, self.np)
        else:
            raise Exception('Particle size distribution type not understood ('
                            + str(psd) + '). '
                            + 'Valid values are \'uni\' or \'logn\'')

        # Show radii as histogram
        if histogram and py_mpl:
            fig = plt.figure(figsize=(8, 8))
            figtitle = 'Particle size distribution, {0} particles'\
                       .format(self.np)
            fig.text(0.5, 0.95, figtitle, horizontalalignment='center',
                     fontproperties=FontProperties(size=18))
            bins = 20

            # Create histogram
            plt.hist(self.radius, bins)

            # Plot
            plt.xlabel('Radii [m]')
            plt.ylabel('Count')
            plt.axis('tight')
            fig.savefig(self.sid + '-psd.png')
            fig.clf()

    def generateBimodalRadii(self, r_small=0.005, r_large=0.05, ratio=0.2,
                             verbose=True):
        '''
        Draw random radii from two distinct sizes.

        :param r_small: Radii of small population [m], in ]0;r_large[
        :type r_small: float
        :param r_large: Radii of large population [m], in ]r_small;inf[
        :type r_large: float
        :param ratio: Approximate volumetric ratio between the two
            populations (large/small).
        :type ratio: float

        See also: :func:`generateRadii()`.
        '''
        if r_small >= r_large:
            raise Exception("r_large should be larger than r_small")

        V_small = V_sphere(r_small)
        V_large = V_sphere(r_large)
        nlarge = int(V_small/V_large * ratio * self.np)  # ignore void volume

        self.radius[:] = r_small
        self.radius[0:nlarge] = r_large
        numpy.random.shuffle(self.radius)

        # Test volumetric ratio
        V_small_total = V_small * (self.np - nlarge)
        V_large_total = V_large * nlarge
        if abs(V_large_total/V_small_total - ratio) > 1.0e5:
            raise Exception("Volumetric ratio seems wrong")

        if verbose:
            print("generateBimodalRadii created " + str(nlarge)
                  + " large particles, and " + str(self.np - nlarge)
                  + " small")

    def checkerboardColors(self, nx=6, ny=6, nz=6):
        '''
        Assign checkerboard color values to the particles in an orthogonal grid.

        :param nx: Number of color values along the x axis
        :type nx: int
        :param ny: Number of color values along the y ayis
        :type ny: int
        :param nz: Number of color values along the z azis
        :type nz: int
        '''
        x_min = numpy.min(self.x[:, 0])
        x_max = numpy.max(self.x[:, 0])
        y_min = numpy.min(self.x[:, 1])
        y_max = numpy.max(self.x[:, 1])
        z_min = numpy.min(self.x[:, 2])
        z_max = numpy.max(self.x[:, 2])
        for i in numpy.arange(self.np):
            ix = numpy.floor((self.x[i, 0] - x_min)/(x_max/nx))
            iy = numpy.floor((self.x[i, 1] - y_min)/(y_max/ny))
            iz = numpy.floor((self.x[i, 2] - z_min)/(z_max/nz))
            self.color[i] = (-1)**ix + (-1)**iy + (-1)**iz

    def contactModel(self, contactmodel):
        '''
        Define which contact model to use for the tangential component of
        particle-particle interactions. The elastic-viscous-frictional contact
        model (2) is considered to be the most realistic contact model, while
        the viscous-frictional contact model is significantly faster.

        :param contactmodel: The type of tangential contact model to use
            (visco-frictional=1, elasto-visco-frictional=2)
        :type contactmodel: int
        '''
        self.contactmodel[0] = contactmodel

    def wall0iz(self):
        '''
        Returns the cell index of wall 0 along z.

        :returns: z cell index
        :return type: int
        '''
        if self.nw > 0:
            return int(self.w_x[0]/(self.L[2]/self.num[2]))
        else:
            raise Exception('No dynamic top wall present!')

    def normalBoundariesXY(self):
        '''
        Set the x and y boundary conditions to be static walls.

        See also :func:`periodicBoundariesXY()` and
        :func:`periodicBoundariesX()`
        '''
        self.periodic[0] = 0

    def periodicBoundariesXY(self):
        '''
        Set the x and y boundary conditions to be periodic.

        See also :func:`normalBoundariesXY()` and
        :func:`periodicBoundariesX()`
        '''
        self.periodic[0] = 1

    def periodicBoundariesX(self):
        '''
        Set the x boundary conditions to be periodic.

        See also :func:`normalBoundariesXY()` and
        :func:`periodicBoundariesXY()`
        '''
        self.periodic[0] = 2

    def adaptiveGrid(self):
        '''
        Set the height of the fluid grid to automatically readjust to the
        height of the granular assemblage, as dictated by the position of the
        top wall.  This will readjust `self.L[2]` during the simulation to
        equal the position of the top wall `self.w_x[0]`.

        See also :func:`staticGrid()`
        '''
        self.adaptive[0] = 1

    def staticGrid(self):
        '''
        Set the height of the fluid grid to be constant as set in `self.L[2]`.

        See also :func:`adaptiveGrid()`
        '''
        self.adaptive[0] = 0

    def initRandomPos(self, gridnum=numpy.array([12, 12, 36]), dx=-1.0):
        '''
        Initialize particle positions in completely random configuration. Radii
        *must* be set beforehand. If the x and y boundaries are set as periodic,
        the particle centers will be placed all the way to the edge. On regular,
        non-periodic boundaries, the particles are restrained at the edges to
        make space for their radii within the bounding box.

        :param gridnum: The number of sorting cells in each spatial direction
            (default=[12, 12, 36])
        :type gridnum: numpy.array
        :param dx: The cell width in any direction. If the default value is used
            (-1), the cell width is calculated to fit the largest particle.
        :type dx: float
        '''

        # Calculate cells in grid
        self.num = gridnum
        r_max = numpy.max(self.radius)

        # Cell configuration
        if dx > 0.0:
            cellsize = dx
        else:
            cellsize = 2.1 * numpy.amax(self.radius)

        # World size
        self.L = self.num * cellsize

        # Particle positions randomly distributed without overlap
        for i in range(self.np):
            overlaps = True
            while overlaps:
                overlaps = False

                # Draw random position
                for d in range(self.nd):
                    self.x[i, d] = (self.L[d] - self.origo[d] - 2*r_max) \
                            * numpy.random.random_sample() \
                            + self.origo[d] + r_max

                # Check other particles for overlaps
                for j in range(i-1):
                    delta = self.x[i] - self.x[j]
                    delta_len = math.sqrt(numpy.dot(delta, delta)) \
                                - (self.radius[i] + self.radius[j])
                    if delta_len < 0.0:
                        overlaps = True
            print("\rFinding non-overlapping particle positions, "
                  + "{0} % complete".format(numpy.ceil(i/self.np*100)))

        # Print newline
        print()


    def defineWorldBoundaries(self, L, origo=[0.0, 0.0, 0.0], dx=-1):
        '''
        Set the boundaries of the world. Particles will only be able to interact
        within this domain. With dynamic walls, allow space for expansions.
        *Important*: The particle radii have to be set beforehand. The world
        edges act as static walls.

        :param L: The upper boundary of the domain [m]
        :type L: numpy.array
        :param origo: The lower boundary of the domain [m]. Negative values
            won't work. Default=[0.0, 0.0, 0.0].
        :type origo: numpy.array
        :param dx: The cell width in any direction. If the default value is used
            (-1), the cell width is calculated to fit the largest particle.
        :type dx: float
        '''

        # Cell configuration
        if dx > 0.0:
            cellsize_min = dx
        else:
            if self.np < 1:
                raise Exception('Error: You need to define dx in ' +
                                'defineWorldBoundaries if there are no ' +
                                'particles in the simulation.')
            cellsize_min = 2.1 * numpy.amax(self.radius)

        # Lower boundary of the sorting grid
        self.origo[:] = origo[:]

        # Upper boundary of the sorting grid
        self.L[:] = L[:]

        # Adjust the number of sorting cells along each axis to fit the largest
        # particle size and the world size
        self.num[0] = numpy.ceil((self.L[0]-self.origo[0])/cellsize_min)
        self.num[1] = numpy.ceil((self.L[1]-self.origo[1])/cellsize_min)
        self.num[2] = numpy.ceil((self.L[2]-self.origo[2])/cellsize_min)

        #if (self.num.any() < 4):
        #if (self.num[0] < 4 or self.num[1] < 4 or self.num[2] < 4):
        if self.num[0] < 3 or self.num[1] < 3 or self.num[2] < 3:
            raise Exception("Error: The grid must be at least 3 cells in each "
                            + "direction\nGrid: x={}, y={}, z={}\n"
                            .format(self.num[0], self.num[1], self.num[2])
                            + "Please increase the world size.")

    def initGrid(self, dx=-1):
        '''
        Initialize grid suitable for the particle positions set previously.
        The margin parameter adjusts the distance (in no. of max. radii)
        from the particle boundaries.
        *Important*: The particle radii have to be set beforehand if the cell
        width isn't specified by `dx`.

        :param dx: The cell width in any direction. If the default value is used
            (-1), the cell width is calculated to fit the largest particle.
        :type dx: float
        '''

        # Cell configuration
        if dx > 0.0:
            cellsize_min = dx
        else:
            cellsize_min = 2.1 * numpy.amax(self.radius)
        self.num[0] = numpy.ceil((self.L[0]-self.origo[0])/cellsize_min)
        self.num[1] = numpy.ceil((self.L[1]-self.origo[1])/cellsize_min)
        self.num[2] = numpy.ceil((self.L[2]-self.origo[2])/cellsize_min)

        if self.num[0] < 4 or self.num[1] < 4 or self.num[2] < 4:
            raise Exception("Error: The grid must be at least 3 cells in each "
                            + "direction\nGrid: x={}, y={}, z={}"
                            .format(self.num[0], self.num[1], self.num[2]))

        # Put upper wall at top boundary
        if self.nw > 0:
            self.w_x[0] = self.L[0]

    def initGridAndWorldsize(self, margin=2.0):
        '''
        Initialize grid suitable for the particle positions set previously.
        The margin parameter adjusts the distance (in no. of max. radii)
        from the particle boundaries. If the upper wall is dynamic, it is placed
        at the top boundary of the world.

        :param margin: Distance to world boundary in no. of max. particle radii
        :type margin: float
        '''

        # Cell configuration
        r_max = numpy.amax(self.radius)

        # Max. and min. coordinates of world
        self.origo = numpy.array([numpy.amin(self.x[:, 0] - self.radius[:]),
                                  numpy.amin(self.x[:, 1] - self.radius[:]),
                                  numpy.amin(self.x[:, 2] - self.radius[:])]) \
                     - margin*r_max
        self.L = numpy.array([numpy.amax(self.x[:, 0] + self.radius[:]),
                              numpy.amax(self.x[:, 1] + self.radius[:]),
                              numpy.amax(self.x[:, 2] + self.radius[:])]) \
                 + margin*r_max

        cellsize_min = 2.1 * r_max
        self.num[0] = numpy.ceil((self.L[0]-self.origo[0])/cellsize_min)
        self.num[1] = numpy.ceil((self.L[1]-self.origo[1])/cellsize_min)
        self.num[2] = numpy.ceil((self.L[2]-self.origo[2])/cellsize_min)

        if self.num[0] < 4 or self.num[1] < 4 or self.num[2] < 4:
            raise Exception("Error: The grid must be at least 3 cells in each "
                            + "direction, num=" + str(self.num))

        # Put upper wall at top boundary
        if self.nw > 0:
            self.w_x[0] = self.L[0]

    def initGridPos(self, gridnum=numpy.array([12, 12, 36])):
        '''
        Initialize particle positions in loose, cubic configuration.
        ``gridnum`` is the number of cells in the x, y and z directions.
        *Important*: The particle radii and the boundary conditions (periodic or
        not) for the x and y boundaries have to be set beforehand.

        :param gridnum: The number of particles in x, y and z directions
        :type gridnum: numpy.array
        '''

        # Calculate cells in grid
        self.num = numpy.asarray(gridnum)

        # World size
        r_max = numpy.amax(self.radius)
        cellsize = 2.1 * r_max
        self.L = self.num * cellsize

        # Check whether there are enough grid cells
        if (self.num[0]*self.num[1]*self.num[2]-(2**3)) < self.np:
            print("Error! The grid is not sufficiently large.")
            raise NameError('Error! The grid is not sufficiently large.')

        gridpos = numpy.zeros(self.nd, dtype=numpy.uint32)

        # Make sure grid is sufficiently large if every second level is moved
        if self.periodic[0] == 1:
            self.num[0] -= 1
            self.num[1] -= 1

        # Check whether there are enough grid cells
        if (self.num[0]*self.num[1]*self.num[2]-(2*3*3)) < self.np:
            print("Error! The grid is not sufficiently large.")
            raise NameError('Error! The grid is not sufficiently large.')

        # Particle positions randomly distributed without overlap
        for i in range(self.np):

            # Find position in 3d mesh from linear index
            gridpos[0] = (i % (self.num[0]))
            gridpos[1] = numpy.floor(i/(self.num[0])) % (self.num[0])
            gridpos[2] = numpy.floor(i/((self.num[0])*(self.num[1]))) #\
                    #% ((self.num[0])*(self.num[1]))

            for d in range(self.nd):
                self.x[i, d] = gridpos[d] * cellsize + 0.5*cellsize

            # Allow pushing every 2.nd level out of lateral boundaries
            if self.periodic[0] == 1:
                # Offset every second level
                if gridpos[2] % 2:
                    self.x[i, 0] += 0.5*cellsize
                    self.x[i, 1] += 0.5*cellsize

        # Readjust grid to correct size
        if self.periodic[0] == 1:
            self.num[0] += 1
            self.num[1] += 1

    def initRandomGridPos(self, gridnum=numpy.array([12, 12, 32]),
                          padding=2.1):
        '''
        Initialize particle positions in loose, cubic configuration with some
        variance. ``gridnum`` is the number of cells in the x, y and z
        directions.  *Important*: The particle radii and the boundary conditions
        (periodic or not) for the x and y boundaries have to be set beforehand.
        The world size and grid height (in the z direction) is readjusted to fit
        the particle positions.

        :param gridnum: The number of particles in x, y and z directions
        :type gridnum: numpy.array
        :param padding: Increase distance between particles in x, y and z
            directions with this multiplier. Large values create more random
            packings.
        :type padding: float
        '''

        # Calculate cells in grid
        coarsegrid = numpy.floor(numpy.asarray(gridnum)/2)

        # World size
        r_max = numpy.amax(self.radius)

        # Cells in grid 2*size to make space for random offset
        cellsize = padding * r_max * 2

        # Check whether there are enough grid cells
        if ((coarsegrid[0]-1)*(coarsegrid[1]-1)*(coarsegrid[2]-1)) < self.np:
            print("Error! The grid is not sufficiently large.")
            raise NameError('Error! The grid is not sufficiently large.')

        gridpos = numpy.zeros(self.nd, dtype=numpy.uint32)

        # Particle positions randomly distributed without overlap
        for i in range(self.np):

            # Find position in 3d mesh from linear index
            gridpos[0] = (i % (coarsegrid[0]))
            gridpos[1] = numpy.floor(i/(coarsegrid[0]))%(coarsegrid[1]) # Thanks Horacio!
            gridpos[2] = numpy.floor(i/((coarsegrid[0])*(coarsegrid[1])))

            # Place particles in grid structure, and randomly adjust the
            # positions within the oversized cells (uniform distribution)
            for d in range(self.nd):
                r = self.radius[i]*1.05
                self.x[i, d] = gridpos[d] * cellsize \
                               + ((cellsize-r) - r) \
                               * numpy.random.random_sample() + r

        # Calculate new grid with cell size equal to max. particle diameter
        x_max = numpy.max(self.x[:, 0] + self.radius)
        y_max = numpy.max(self.x[:, 1] + self.radius)
        z_max = numpy.max(self.x[:, 2] + self.radius)

        # Adjust size of world
        self.num[0] = numpy.ceil(x_max/cellsize)
        self.num[1] = numpy.ceil(y_max/cellsize)
        self.num[2] = numpy.ceil(z_max/cellsize)
        self.L = self.num * cellsize

    def createBondPair(self, i, j, spacing=-0.1):
        '''
        Bond particles i and j. Particle j is moved adjacent to particle i,
        and oriented randomly.

        :param i: Index of first particle in bond
        :type i: int
        :param j: Index of second particle in bond
        :type j: int
        :param spacing: The inter-particle distance prescribed. Positive
            values result in a inter-particle distance, negative equal an
            overlap. The value is relative to the sum of the two radii.
        :type spacing: float
        '''

        x_i = self.x[i]
        r_i = self.radius[i]
        r_j = self.radius[j]
        dist_ij = (r_i + r_j)*(1.0 + spacing)

        dazi = numpy.random.rand(1) * 360.0  # azimuth
        azi = numpy.radians(dazi)
        dang = numpy.random.rand(1) * 180.0 - 90.0 # angle
        ang = numpy.radians(dang)

        x_j = numpy.copy(x_i)
        x_j[0] = x_j[0] + dist_ij * numpy.cos(azi) * numpy.cos(ang)
        x_j[1] = x_j[1] + dist_ij * numpy.sin(azi) * numpy.cos(ang)
        x_j[2] = x_j[2] + dist_ij * numpy.sin(ang) * numpy.cos(azi)
        self.x[j] = x_j

        if self.x[j, 0] < self.origo[0]:
            self.x[j, 0] += x_i[0] - x_j[0]
        if self.x[j, 1] < self.origo[1]:
            self.x[j, 1] += x_i[1] - x_j[1]
        if self.x[j, 2] < self.origo[2]:
            self.x[j, 2] += x_i[2] - x_j[2]

        if self.x[j, 0] > self.L[0]:
            self.x[j, 0] -= abs(x_j[0] - x_i[0])
        if self.x[j, 1] > self.L[1]:
            self.x[j, 1] -= abs(x_j[1] - x_i[1])
        if self.x[j, 2] > self.L[2]:
            self.x[j, 2] -= abs(x_j[2] - x_i[2])

        self.bond(i, j)     # register bond

        # Check that the spacing is correct
        x_ij = self.x[i] - self.x[j]
        x_ij_length = numpy.sqrt(x_ij.dot(x_ij))
        if (x_ij_length - dist_ij) > dist_ij*0.01:
            print(x_i); print(r_i)
            print(x_j); print(r_j)
            print(x_ij_length); print(dist_ij)
            raise Exception("Error, something went wrong in createBondPair")


    def randomBondPairs(self, ratio=0.3, spacing=-0.1):
        '''
        Bond an amount of particles in two-particle clusters. The particles
        should be initialized beforehand.  Note: The actual number of bonds is
        likely to be somewhat smaller than specified, due to the random
        selection algorithm.

        :param ratio: The amount of particles to bond, values in ]0.0;1.0]
        :type ratio: float
        :param spacing: The distance relative to the sum of radii between bonded
                particles, neg. values denote an overlap. Values in ]0.0,inf[.
        :type spacing: float
        '''

        bondparticles = numpy.unique(numpy.random.random_integers(0, high=self.np-1,
                                                                  size=int(self.np*ratio)))
        if bondparticles.size % 2 > 0:
            bondparticles = bondparticles[:-1].copy()
        bondparticles = bondparticles.reshape(int(bondparticles.size/2),
                                              2).copy()

        for n in numpy.arange(bondparticles.shape[0]):
            self.createBondPair(bondparticles[n, 0], bondparticles[n, 1],
                                spacing)

    def zeroKinematics(self):
        '''
        Zero all kinematic parameters of the particles. This function is useful
        when output from one simulation is reused in another simulation.
        '''

        self.force = numpy.zeros((self.np, self.nd))
        self.torque = numpy.zeros((self.np, self.nd))
        self.vel = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                   .reshape(self.np, self.nd)
        self.angvel = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                      .reshape(self.np, self.nd)
        self.angpos = numpy.zeros(self.np*self.nd, dtype=numpy.float64)\
                      .reshape(self.np, self.nd)
        self.es = numpy.zeros(self.np, dtype=numpy.float64)
        self.ev = numpy.zeros(self.np, dtype=numpy.float64)
        self.xyzsum = numpy.zeros(self.np*3, dtype=numpy.float64).reshape(self.np, 3)

    def adjustUpperWall(self, z_adjust=1.1):
        '''
        Included for legacy purposes, calls :func:`adjustWall()` with ``idx=0``.

        :param z_adjust: Increase the world and grid size by this amount to
            allow for wall movement.
        :type z_adjust: float
        '''

        # Initialize upper wall
        self.nw = 1
        self.wmode = numpy.zeros(1) # fixed BC
        self.w_n = numpy.zeros(self.nw*self.nd, dtype=numpy.float64)\
                   .reshape(self.nw, self.nd)
        self.w_n[0, 2] = -1.0
        self.w_vel = numpy.zeros(1)
        self.w_force = numpy.zeros(1)
        self.w_sigma0 = numpy.zeros(1)

        self.w_x = numpy.zeros(1)
        self.w_m = numpy.zeros(1)
        self.adjustWall(idx=0, adjust=z_adjust)

    def adjustWall(self, idx, adjust=1.1):
        '''
        Adjust grid and dynamic wall to max. particle position. The wall
        thickness will by standard equal the maximum particle diameter. The
        density equals the particle density, and the wall size is equal to the
        width and depth of the simulation domain (`self.L[0]` and `self.L[1]`).

        :param: idx: The wall to adjust. 0=+z, upper wall (default), 1=-x,
            left wall, 2=+x, right wall, 3=-y, front wall, 4=+y, back
            wall.
        :type idx: int
        :param adjust: Increase the world and grid size by this amount to
            allow for wall movement.
        :type adjust: float
        '''

        if idx == 0:
            dim = 2
        elif idx == 1 or idx == 2:
            dim = 0
        elif idx == 3 or idx == 4:
            dim = 1
        else:
            print("adjustWall: idx value not understood")

        xmin = numpy.min(self.x[:, dim] - self.radius)
        xmax = numpy.max(self.x[:, dim] + self.radius)

        cellsize = self.L[0] / self.num[0]
        self.num[dim] = numpy.ceil(((xmax-xmin)*adjust + xmin)/cellsize)
        self.L[dim] = (xmax-xmin)*adjust + xmin

        # Initialize upper wall
        if idx == 0 or idx == 1 or idx == 3:
            self.w_x[idx] = numpy.array([xmax])
        else:
            self.w_x[idx] = numpy.array([xmin])
        self.w_m[idx] = self.totalMass()

    def consolidate(self, normal_stress=10e3):
        '''
        Setup consolidation experiment. Specify the upper wall normal stress in
        Pascal, default value is 10 kPa.

        :param normal_stress: The normal stress to apply from the upper wall
        :type normal_stress: float
        '''

        self.nw = 1

        if normal_stress <= 0.0:
            raise Exception('consolidate() error: The normal stress should be '
                            'a positive value, but is ' + str(normal_stress) +
                            ' Pa')

        # Zero the kinematics of all particles
        self.zeroKinematics()

        # Adjust grid and placement of upper wall
        self.adjustUpperWall()

        # Set the top wall BC to a value of normal stress
        self.wmode = numpy.array([1])
        self.w_sigma0 = numpy.ones(1) * normal_stress

        # Set top wall to a certain mass corresponding to the selected normal
        # stress
        #self.w_sigma0 = numpy.zeros(1)
        #self.w_m[0] = numpy.abs(normal_stress*self.L[0]*self.L[1]/self.g[2])
        self.w_m[0] = self.totalMass()

    def uniaxialStrainRate(self, wvel=-0.001):
        '''
        Setup consolidation experiment. Specify the upper wall velocity in m/s,
        default value is -0.001 m/s (i.e. downwards).

        :param wvel: Upper wall velocity. Negative values mean that the wall
            moves downwards.
        :type wvel: float
        '''

        # zero kinematics
        self.zeroKinematics()

        # Initialize upper wall
        self.adjustUpperWall()
        self.wmode = numpy.array([2]) # strain rate BC
        self.w_vel = numpy.array([wvel])

    def triaxial(self, wvel=-0.001, normal_stress=10.0e3):
        '''
        Setup triaxial experiment. The upper wall is moved at a fixed velocity
        in m/s, default values is -0.001 m/s (i.e. downwards). The side walls
        are exerting a defined normal stress.

        :param wvel: Upper wall velocity. Negative values mean that the wall
            moves downwards.
        :type wvel: float
        :param normal_stress: The normal stress to apply from the upper wall.
        :type normal_stress: float
        '''

        # zero kinematics
        self.zeroKinematics()

        # Initialize walls
        self.nw = 5  # five dynamic walls
        self.wmode = numpy.array([2, 1, 1, 1, 1]) # BCs (vel, stress, stress, ...)
        self.w_vel = numpy.array([1, 0, 0, 0, 0]) * wvel
        self.w_sigma0 = numpy.array([0, 1, 1, 1, 1]) * normal_stress
        self.w_n = numpy.array(([0, 0, -1], [-1, 0, 0],
                                [1, 0, 0], [0, -1, 0], [0, 1, 0]),
                               dtype=numpy.float64)
        self.w_x = numpy.zeros(5)
        self.w_m = numpy.zeros(5)
        self.w_force = numpy.zeros(5)
        for i in range(5):
            self.adjustWall(idx=i)

    def shear(self, shear_strain_rate=1.0, shear_stress=False):
        '''
        Setup shear experiment either by a constant shear rate or a constant
        shear stress.  The shear strain rate is the shear velocity divided by
        the initial height per second. The shear movement is along the positive
        x axis. The function zeroes the tangential wall viscosity (gamma_wt) and
        the wall friction coefficients (mu_ws, mu_wn).

        :param shear_strain_rate: The shear strain rate [-] to use if
            shear_stress isn't False.
        :type shear_strain_rate: float
        :param shear_stress: The shear stress value to use [Pa].
        :type shear_stress: float or bool
        '''

        self.nw = 1

        # Find lowest and heighest point
        z_min = numpy.min(self.x[:, 2] - self.radius)
        z_max = numpy.max(self.x[:, 2] + self.radius)

        # the grid cell size is equal to the max. particle diameter
        cellsize = self.L[0] / self.num[0]

        # make grid one cell heigher to allow dilation
        self.num[2] += 1
        self.L[2] = self.num[2] * cellsize

        # zero kinematics
        self.zeroKinematics()

        # Adjust grid and placement of upper wall
        self.wmode = numpy.array([1])

        # Fix horizontal velocity to 0.0 of lowermost particles
        d_max_below = numpy.max(self.radius[numpy.nonzero(self.x[:, 2] <
                                                          (z_max-z_min)*0.3)])*2.0
        I = numpy.nonzero(self.x[:, 2] < (z_min + d_max_below))
        self.fixvel[I] = 1
        self.angvel[I, 0] = 0.0
        self.angvel[I, 1] = 0.0
        self.angvel[I, 2] = 0.0
        self.vel[I, 0] = 0.0 # x-dim
        self.vel[I, 1] = 0.0 # y-dim
        self.color[I] = -1

        # Fix horizontal velocity to specific value of uppermost particles
        d_max_top = numpy.max(self.radius[numpy.nonzero(self.x[:, 2] >
                                                        (z_max-z_min)*0.7)])*2.0
        I = numpy.nonzero(self.x[:, 2] > (z_max - d_max_top))
        self.fixvel[I] = 1
        self.angvel[I, 0] = 0.0
        self.angvel[I, 1] = 0.0
        self.angvel[I, 2] = 0.0
        if not shear_stress:
            self.vel[I, 0] = (z_max-z_min)*shear_strain_rate
        else:
            self.vel[I, 0] = 0.0
            self.wmode[0] = 3
            self.w_tau_x[0] = float(shear_stress)
        self.vel[I, 1] = 0.0 # y-dim
        self.color[I] = -1

        # Set wall tangential viscosity to zero
        self.gamma_wt[0] = 0.0

        # Set wall friction coefficients to zero
        self.mu_ws[0] = 0.0
        self.mu_wd[0] = 0.0

    def largestFluidTimeStep(self, safety=0.5, v_max=-1.0):
        '''
        Finds and returns the largest time step in the fluid phase by von
        Neumann and Courant-Friedrichs-Lewy analysis given the current
        velocities. This ensures stability in the diffusive and advective parts
        of the momentum equation.

        The value of the time step decreases with increasing fluid viscosity
        (`self.mu`), and increases with fluid cell size (`self.L/self.num`)
        and fluid velocities (`self.v_f`).

        NOTE: The fluid time step with the Darcy solver is an arbitrarily
        large value. In practice, this is not a problem since the short
        DEM time step is stable for fluid computations.

        :param safety: Safety factor which is multiplied to the largest time
            step.
        :type safety: float
        :param v_max: The largest anticipated absolute fluid velocity [m/s]
        :type v_max: float

        :returns: The largest timestep stable for the current fluid state.
        :return type: float
        '''

        if self.fluid:

            # Normalized velocities
            v_norm = numpy.empty(self.num[0]*self.num[1]*self.num[2])
            idx = 0
            for x in numpy.arange(self.num[0]):
                for y in numpy.arange(self.num[1]):
                    for z in numpy.arange(self.num[2]):
                        v_norm[idx] = numpy.sqrt(self.v_f[x, y, z, :]\
                                              .dot(self.v_f[x, y, z, :]))
                        idx += 1

            v_max_obs = numpy.amax(v_norm)
            if v_max_obs == 0:
                v_max_obs = 1.0e-7
            if v_max < 0.0:
                v_max = v_max_obs

            dx_min = numpy.min(self.L/self.num)
            dt_min_cfl = dx_min/v_max

            # Navier-Stokes
            if self.cfd_solver[0] == 0:
                dt_min_von_neumann = 0.5*dx_min**2/(self.mu[0] + 1.0e-16)

                return numpy.min([dt_min_von_neumann, dt_min_cfl])*safety

            # Darcy
            elif self.cfd_solver[0] == 1:

                return dt_min_cfl

                '''
                # Determine on the base of the diffusivity coefficient
                # components
                #self.hydraulicPermeability()
                #alpha_max = numpy.max(self.k/(self.beta_f*0.9*self.mu))
                k_max = 2.7e-10  # hardcoded in darcy.cuh
                phi_min = 0.1    # hardcoded in darcy.cuh
                alpha_max = k_max/(self.beta_f*phi_min*self.mu)
                print(alpha_max)
                return safety * 1.0/(2.0*alpha_max)*1.0/(
                        1.0/(self.dx[0]**2) + \
                        1.0/(self.dx[1]**2) + \
                        1.0/(self.dx[2]**2))
                        '''

                '''
                # Determine value on the base of the hydraulic conductivity
                g = numpy.max(numpy.abs(self.g))

                # Bulk modulus of fluid
                K = 1.0/self.beta_f[0]

                self.hydraulicDiffusivity()

                return safety * 1.0/(2.0*self.D)*1.0/( \
                        1.0/(self.dx[0]**2) + \
                        1.0/(self.dx[1]**2) + \
                        1.0/(self.dx[2]**2))
                '''

    def hydraulicConductivity(self, phi=0.35):
        '''
        Determine the hydraulic conductivity (K) [m/s] from the permeability
        prefactor and a chosen porosity.  This value is stored in `self.K_c`.
        This function only works for the Darcy solver (`self.cfd_solver == 1`)

        :param phi: The porosity to use in the Kozeny-Carman relationship
        :type phi: float
        :returns: The hydraulic conductivity [m/s]
        :return type: float
        '''
        if self.cfd_solver[0] == 1:
            k = self.k_c * phi**3/(1.0 - phi**2)
            self.K_c = k*self.rho_f*numpy.abs(self.g[2])/self.mu
            return self.K_c[0]
        else:
            raise Exception('This function only works for the Darcy solver')

    def hydraulicPermeability(self):
        '''
        Determine the hydraulic permeability (k) [m*m] from the Kozeny-Carman
        relationship, using the permeability prefactor (`self.k_c`), and the
        range of valid porosities set in `src/darcy.cuh`, by default in the
        range 0.1 to 0.9.

        This function is only valid for the Darcy solver (`self.cfd_solver ==
        1`).
        '''
        if self.cfd_solver[0] == 1:
            self.findPermeabilities()
        else:
            raise Exception('This function only works for the Darcy solver')

    def hydraulicDiffusivity(self):
        '''
        Determine the hydraulic diffusivity (D) [m*m/s]. The result is stored in
        `self.D`. This function only works for the Darcy solver
        (`self.cfd_solver[0] == 1`)
        '''
        if self.cfd_solver[0] == 1:
            self.hydraulicConductivity()
            phi_bar = numpy.mean(self.phi)
            self.D = self.K_c/(self.rho_f*self.g[2]
                               *(self.k_n[0] + phi_bar*self.K))
        else:
            raise Exception('This function only works for the Darcy solver')

    def initTemporal(self, total, current=0.0, file_dt=0.05, step_count=0,
                     dt=-1, epsilon=0.01):
        '''
        Set temporal parameters for the simulation. *Important*: Particle radii,
        physical parameters, and the optional fluid grid need to be set prior to
        these if the computational time step (dt) isn't set explicitly. If the
        parameter `dt` is the default value (-1), the function will estimate the
        best time step length. The value of the computational time step for the
        DEM is checked for stability in the CFD solution if fluid simulation is
        included.

        :param total: The time at which to end the simulation [s]
        :type total: float
        :param current: The current time [s] (default=0.0 s)
        :type total: float
        :param file_dt: The interval between output files [s] (default=0.05 s)
        :type total: float
        :step_count: The number of the first output file (default=0)
        :type step_count: int
        :param dt: The computational time step length [s]
        :type total: float
        :param epsilon: Time step multiplier (default=0.01)
        :type epsilon: float
        '''

        if dt > 0.0:
            self.time_dt[0] = dt
            if self.np > 0:
                print("Warning: Manually specifying the time step length when "
                      "simulating particles may produce instabilities.")

        elif self.np > 0:

            r_min = numpy.min(self.radius)
            m_min = self.rho * 4.0/3.0*numpy.pi*r_min**3

            if self.E > 0.001:
                k_max = numpy.max(numpy.pi/2.0*self.E*self.radius)
            else:
                k_max = numpy.max([self.k_n[:], self.k_t[:]])

            # Radjaii et al 2011
            self.time_dt[0] = epsilon/(numpy.sqrt(k_max/m_min))

            # Zhang and Campbell, 1992
            #self.time_dt[0] = 0.075*math.sqrt(m_min/k_max)

            # Computational time step (O'Sullivan et al, 2003)
            #self.time_dt[0] = 0.17*math.sqrt(m_min/k_max)

        elif not self.fluid:
            raise Exception('Error: Could not automatically set a time step.')

        # Check numerical stability of the fluid phase, by criteria derived
        # by von Neumann stability analysis of the diffusion and advection
        # terms
        if self.fluid:
            fluid_time_dt = self.largestFluidTimeStep()
            self.time_dt[0] = numpy.min([fluid_time_dt, self.time_dt[0]])

        # Time at start
        self.time_current[0] = current
        self.time_total[0] = total
        self.time_file_dt[0] = file_dt
        self.time_step_count[0] = step_count

    def dry(self):
        '''
        Set the simulation to be dry (no fluids).

        See also :func:`wet()`
        '''
        self.fluid = False

    def wet(self):
        '''
        Set the simulation to be wet (total fluid saturation).

        See also :func:`dry()`
        '''
        self.fluid = True
        self.initFluid()

    def initFluid(self, mu=8.9e-4, rho=1.0e3, p=0.0, hydrostatic=False,
                  cfd_solver=0):
        '''
        Initialize the fluid arrays and the fluid viscosity. The default value
        of ``mu`` equals the dynamic viscosity of water at 25 degrees Celcius.
        The value for water at 0 degrees Celcius is 17.87e-4 kg/(m*s).

        :param mu: The fluid dynamic viscosity [kg/(m*s)]
        :type mu: float
        :param rho: The fluid density [kg/(m^3)]
        :type rho: float
        :param p: The hydraulic pressure to initialize the cells to. If the
            parameter `hydrostatic` is set to `True`, this value will apply to
            the fluid cells at the top
        :param hydrostatic: Initialize the fluid pressures to the hydrostatic
            pressure distribution. A pressure gradient with depth is only
            created if a gravitational acceleration along :math:`z` previously
            has been specified
        :type hydrostatic: bool
        :param cfd_solver: Solver to use for the computational fluid dynamics.
            Accepted values: 0 (Navier Stokes, default) and 1 (Darcy).
        :type cfd_solver: int
        '''
        self.fluid = True
        self.mu = numpy.ones(1, dtype=numpy.float64) * mu
        self.rho_f = numpy.ones(1, dtype=numpy.float64) * rho

        self.p_f = numpy.ones((self.num[0], self.num[1], self.num[2]),
                              dtype=numpy.float64) * p

        if hydrostatic:

            dz = self.L[2]/self.num[2]
            # Zero pressure gradient from grid top to top wall, linear pressure
            # distribution from top wall to grid bottom
            if self.nw == 1:
                wall0_iz = int(self.w_x[0]/(self.L[2]/self.num[2]))
                self.p_f[:, :, wall0_iz:] = p

                for iz in numpy.arange(wall0_iz - 1):
                    z = dz*iz + 0.5*dz
                    depth = self.w_x[0] - z
                    self.p_f[:, :, iz] = p + (depth-dz) * rho * -self.g[2]

            # Linear pressure distribution from grid top to grid bottom
            else:
                for iz in numpy.arange(self.num[2] - 1):
                    z = dz*iz + 0.5*dz
                    depth = self.L[2] - z
                    self.p_f[:, :, iz] = p + (depth-dz) * rho * -self.g[2]


        self.v_f = numpy.zeros((self.num[0], self.num[1], self.num[2], self.nd),
                               dtype=numpy.float64)
        self.phi = numpy.ones((self.num[0], self.num[1], self.num[2]),
                              dtype=numpy.float64)
        self.dphi = numpy.zeros((self.num[0], self.num[1], self.num[2]),
                                dtype=numpy.float64)

        self.p_mod_A = numpy.zeros(1, dtype=numpy.float64)  # Amplitude [Pa]
        self.p_mod_f = numpy.zeros(1, dtype=numpy.float64)  # Frequency [Hz]
        self.p_mod_phi = numpy.zeros(1, dtype=numpy.float64) # Shift [rad]

        self.bc_bot = numpy.zeros(1, dtype=numpy.int32)
        self.bc_top = numpy.zeros(1, dtype=numpy.int32)
        self.free_slip_bot = numpy.ones(1, dtype=numpy.int32)
        self.free_slip_top = numpy.ones(1, dtype=numpy.int32)
        self.bc_bot_flux = numpy.zeros(1, dtype=numpy.float64)
        self.bc_top_flux = numpy.zeros(1, dtype=numpy.float64)

        self.p_f_constant = numpy.zeros((self.num[0], self.num[1], self.num[2]),
                                        dtype=numpy.int32)

        # Fluid solver type
        # 0: Navier Stokes (fluid with inertia)
        # 1: Stokes-Darcy (fluid without inertia)
        self.cfd_solver = numpy.ones(1)*cfd_solver

        if self.cfd_solver[0] == 0:
            self.gamma = numpy.array(0.0)
            self.theta = numpy.array(1.0)
            self.beta = numpy.array(0.0)
            self.tolerance = numpy.array(1.0e-3)
            self.maxiter = numpy.array(1e4)
            self.ndem = numpy.array(1)

            self.c_phi = numpy.ones(1, dtype=numpy.float64)
            self.c_v = numpy.ones(1, dtype=numpy.float64)
            self.dt_dem_fac = numpy.ones(1, dtype=numpy.float64)

            self.f_d = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.f_p = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.f_v = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.f_sum = numpy.zeros((self.np, self.nd), dtype=numpy.float64)

        elif self.cfd_solver[0] == 1:
            self.tolerance = numpy.array(1.0e-3)
            self.maxiter = numpy.array(1e4)
            self.ndem = numpy.array(1)
            self.c_phi = numpy.ones(1, dtype=numpy.float64)
            self.f_d = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.beta_f = numpy.ones(1, dtype=numpy.float64)*4.5e-10
            self.f_p = numpy.zeros((self.np, self.nd), dtype=numpy.float64)
            self.k_c = numpy.ones(1, dtype=numpy.float64)*4.6e-10

            self.bc_xn = numpy.ones(1, dtype=numpy.int32)*2
            self.bc_xp = numpy.ones(1, dtype=numpy.int32)*2
            self.bc_yn = numpy.ones(1, dtype=numpy.int32)*2
            self.bc_yp = numpy.ones(1, dtype=numpy.int32)*2

        else:
            raise Exception('Value of cfd_solver not understood (' + \
                    str(self.cfd_solver[0]) + ')')

    def currentTime(self, value=-1):
        '''
        Get or set the current time. If called without arguments the current
        time is returned. If a new time is passed in the 'value' argument, the
        time is written to the object.

        :param value: The new current time
        :type value: float

        :returns: The current time
        :return type: float
        '''
        if value != -1:
            self.time_current[0] = value
        else:
            return self.time_current[0]

    def setFluidBottomNoFlow(self):
        '''
        Set the lower boundary of the fluid domain to follow the no-flow
        (Neumann) boundary condition with free slip parallel to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidBottomFixedPressure()`.
        '''
        self.bc_bot[0] = 1

    def setFluidBottomNoFlowNoSlip(self):
        '''
        Set the lower boundary of the fluid domain to follow the no-flow
        (Neumann) boundary condition with no slip parallel to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidBottomFixedPressure()`.
        '''
        self.bc_bot[0] = 2

    def setFluidBottomFixedPressure(self):
        '''
        Set the lower boundary of the fluid domain to follow the fixed pressure
        value (Dirichlet) boundary condition.

        This is the default behavior for the boundary. See also
        :func:`setFluidBottomNoFlow()`
        '''
        self.bc_bot[0] = 0

    def setFluidBottomFixedFlux(self, specific_flux):
        '''
        Define a constant fluid flux normal to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidBottomFixedPressure()`.

        :param specific_flux: Specific flux values across boundary (positive
            values upwards), [m/s]
        '''
        self.bc_bot[0] = 4
        self.bc_bot_flux[0] = specific_flux

    def setFluidTopNoFlow(self):
        '''
        Set the upper boundary of the fluid domain to follow the no-flow
        (Neumann) boundary condition with free slip parallel to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidTopFixedPressure()`.
        '''
        self.bc_top[0] = 1

    def setFluidTopNoFlowNoSlip(self):
        '''
        Set the upper boundary of the fluid domain to follow the no-flow
        (Neumann) boundary condition with no slip parallel to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidTopFixedPressure()`.
        '''
        self.bc_top[0] = 2

    def setFluidTopFixedPressure(self):
        '''
        Set the upper boundary of the fluid domain to follow the fixed pressure
        value (Dirichlet) boundary condition.

        This is the default behavior for the boundary. See also
        :func:`setFluidTopNoFlow()`
        '''
        self.bc_top[0] = 0

    def setFluidTopFixedFlux(self, specific_flux):
        '''
        Define a constant fluid flux normal to the boundary.

        The default behavior for the boundary is fixed value (Dirichlet), see
        :func:`setFluidBottomFixedPressure()`.

        :param specific_flux: Specific flux values across boundary (positive
            values upwards), [m/s]
        '''
        self.bc_top[0] = 4
        self.bc_top_flux[0] = specific_flux

    def setFluidXFixedPressure(self):
        '''
        Set the X boundaries of the fluid domain to follow the fixed pressure
        value (Dirichlet) boundary condition.

        This is not the default behavior for the boundary. See also
        :func:`setFluidXFixedPressure()`,
        :func:`setFluidXNoFlow()`, and
        :func:`setFluidXPeriodic()` (default)
        '''
        self.bc_xn[0] = 0
        self.bc_xp[0] = 0

    def setFluidXNoFlow(self):
        '''
        Set the X boundaries of the fluid domain to follow the no-flow
        (Neumann) boundary condition.

        This is not the default behavior for the boundary. See also
        :func:`setFluidXFixedPressure()`,
        :func:`setFluidXNoFlow()`, and
        :func:`setFluidXPeriodic()` (default)
        '''
        self.bc_xn[0] = 1
        self.bc_xp[0] = 1

    def setFluidXPeriodic(self):
        '''
        Set the X boundaries of the fluid domain to follow the periodic
        (cyclic) boundary condition.

        This is the default behavior for the boundary. See also
        :func:`setFluidXFixedPressure()` and
        :func:`setFluidXNoFlow()`
        '''
        self.bc_xn[0] = 2
        self.bc_xp[0] = 2

    def setFluidYFixedPressure(self):
        '''
        Set the Y boundaries of the fluid domain to follow the fixed pressure
        value (Dirichlet) boundary condition.

        This is not the default behavior for the boundary. See also
        :func:`setFluidYNoFlow()` and
        :func:`setFluidYPeriodic()` (default)
        '''
        self.bc_yn[0] = 0
        self.bc_yp[0] = 0

    def setFluidYNoFlow(self):
        '''
        Set the Y boundaries of the fluid domain to follow the no-flow
        (Neumann) boundary condition.

        This is not the default behavior for the boundary. See also
        :func:`setFluidYFixedPressure()` and
        :func:`setFluidYPeriodic()` (default)
        '''
        self.bc_yn[0] = 1
        self.bc_yp[0] = 1

    def setFluidYPeriodic(self):
        '''
        Set the Y boundaries of the fluid domain to follow the periodic
        (cyclic) boundary condition.

        This is the default behavior for the boundary. See also
        :func:`setFluidYFixedPressure()` and
        :func:`setFluidYNoFlow()`
        '''
        self.bc_yn[0] = 2
        self.bc_yp[0] = 2

    def setPermeabilityGrainSize(self, verbose=True):
        '''
        Set the permeability prefactor based on the mean grain size (Damsgaard
        et al., 2015, eq. 10).

        :param verbose: Print information about the realistic permeabilities
            hydraulic conductivities to expect with the chosen permeability
            prefactor.
        :type verbose: bool
        '''
        self.setPermeabilityPrefactor(k_c=numpy.mean(self.radius*2.0)**2.0/180.0,
                                      verbose=verbose)

    def setPermeabilityPrefactor(self, k_c, verbose=True):
        '''
        Set the permeability prefactor from Goren et al 2011, eq. 24. The
        function will print the limits of permeabilities to be simulated. This
        parameter is only used in the Darcy solver.

        :param k_c: Permeability prefactor value [m*m]
        :type k_c: float
        :param verbose: Print information about the realistic permeabilities and
            hydraulic conductivities to expect with the chosen permeability
            prefactor.
        :type verbose: bool
        '''
        if self.cfd_solver[0] == 1:
            self.k_c[0] = k_c
            if verbose:
                phi = numpy.array([0.1, 0.35, 0.9])
                k = self.k_c * phi**3/(1.0 - phi**2)
                K = k * self.rho_f*numpy.abs(self.g[2])/self.mu
                print('Hydraulic permeability limits for porosity phi=' + \
                        str(phi) + ':')
                print('\tk=' + str(k) + ' m*m')
                print('Hydraulic conductivity limits for porosity phi=' + \
                        str(phi) + ':')
                print('\tK=' + str(K) + ' m/s')
        else:
            raise Exception('setPermeabilityPrefactor() only relevant for the '
                            'Darcy solver (cfd_solver=1)')

    def findPermeabilities(self):
        '''
        Calculates the hydrological permeabilities from the Kozeny-Carman
        relationship. These values are only relevant when the Darcy solver is
        used (`self.cfd_solver=1`). The permeability pre-factor `self.k_c`
        and the assemblage porosities must be set beforehand. The former values
        are set if a file from the `output/` folder is read using
        `self.readbin`.
        '''
        if self.cfd_solver[0] == 1:
            phi = numpy.clip(self.phi, 0.1, 0.9)
            self.k = self.k_c * phi**3/(1.0 - phi**2)
        else:
            raise Exception('findPermeabilities() only relevant for the '
                            'Darcy solver (cfd_solver=1)')

    def findHydraulicConductivities(self):
        '''
        Calculates the hydrological conductivities from the Kozeny-Carman
        relationship. These values are only relevant when the Darcy solver is
        used (`self.cfd_solver=1`). The permeability pre-factor `self.k_c`
        and the assemblage porosities must be set beforehand. The former values
        are set if a file from the `output/` folder is read using
        `self.readbin`.
        '''
        if self.cfd_solver[0] == 1:
            self.findPermeabilities()
            self.K = self.k*self.rho_f*numpy.abs(self.g[2])/self.mu
        else:
            raise Exception('findPermeabilities() only relevant for the '
                            'Darcy solver (cfd_solver=1)')

    def defaultParams(self, mu_s=0.5, mu_d=0.5, mu_r=0.0, rho=2600, k_n=1.16e9,
                      k_t=1.16e9, k_r=0, gamma_n=0.0, gamma_t=0.0, gamma_r=0.0,
                      gamma_wn=0.0, gamma_wt=0.0, capillaryCohesion=0):
        '''
        Initialize particle parameters to default values.

        :param mu_s: The coefficient of static friction between particles [-]
        :type mu_s: float
        :param mu_d: The coefficient of dynamic friction between particles [-]
        :type mu_d: float
        :param rho: The density of the particle material [kg/(m^3)]
        :type rho: float
        :param k_n: The normal stiffness of the particles [N/m]
        :type k_n: float
        :param k_t: The tangential stiffness of the particles [N/m]
        :type k_t: float
        :param k_r: The rolling stiffness of the particles [N/rad] *Parameter
            not used*
        :type k_r: float
        :param gamma_n: Particle-particle contact normal viscosity [Ns/m]
        :type gamma_n: float
        :param gamma_t: Particle-particle contact tangential viscosity [Ns/m]
        :type gamma_t: float
        :param gamma_r: Particle-particle contact rolling viscosity *Parameter
            not used*
        :type gamma_r: float
        :param gamma_wn: Wall-particle contact normal viscosity [Ns/m]
        :type gamma_wn: float
        :param gamma_wt: Wall-particle contact tangential viscosity [Ns/m]
        :type gamma_wt: float
        :param capillaryCohesion: Enable particle-particle capillary cohesion
            interaction model (0=no (default), 1=yes)
        :type capillaryCohesion: int
        '''

        # Particle material density, kg/m^3
        self.rho = numpy.ones(1, dtype=numpy.float64) * rho


        ### Dry granular material parameters

        # Contact normal elastic stiffness, N/m
        self.k_n = numpy.ones(1, dtype=numpy.float64) * k_n

        # Contact shear elastic stiffness (for contactmodel=2), N/m
        self.k_t = numpy.ones(1, dtype=numpy.float64) * k_t

        # Contact rolling elastic stiffness (for contactmodel=2), N/m
        self.k_r = numpy.ones(1, dtype=numpy.float64) * k_r

        # Contact normal viscosity. Critical damping: 2*sqrt(m*k_n).
        # Normal force component elastic if nu=0.0.
        #self.gamma_n=numpy.ones(self.np, dtype=numpy.float64) \
                #          * nu_frac * 2.0 * math.sqrt(4.0/3.0 * math.pi \
                #          * numpy.amin(self.radius)**3 \
                #          * self.rho[0] * self.k_n[0])
        self.gamma_n = numpy.ones(1, dtype=numpy.float64) * gamma_n

        # Contact shear viscosity, Ns/m
        self.gamma_t = numpy.ones(1, dtype=numpy.float64) * gamma_t

        # Contact rolling viscosity, Ns/m?
        self.gamma_r = numpy.ones(1, dtype=numpy.float64) * gamma_r

        # Contact static shear friction coefficient
        #self.mu_s = numpy.ones(1, dtype=numpy.float64) * \
                #numpy.tan(numpy.radians(ang_s))
        self.mu_s = numpy.ones(1, dtype=numpy.float64) * mu_s

        # Contact dynamic shear friction coefficient
        #self.mu_d = numpy.ones(1, dtype=numpy.float64) * \
                #numpy.tan(numpy.radians(ang_d))
        self.mu_d = numpy.ones(1, dtype=numpy.float64) * mu_d

        # Contact rolling friction coefficient
        #self.mu_r = numpy.ones(1, dtype=numpy.float64) * \
                #numpy.tan(numpy.radians(ang_r))
        self.mu_r = numpy.ones(1, dtype=numpy.float64) * mu_r

        # Wall viscosities
        self.gamma_wn[0] = gamma_wn # normal
        self.gamma_wt[0] = gamma_wt # sliding

        # Wall friction coefficients
        self.mu_ws = self.mu_s  # static
        self.mu_wd = self.mu_d  # dynamic

        ### Parameters related to capillary bonds

        # Wettability, 0=perfect
        theta = 0.0
        if capillaryCohesion == 1:
            # Prefactor
            self.kappa[0] = 2.0 * math.pi * gamma_t * numpy.cos(theta)
            self.V_b[0] = 1e-12  # Liquid volume at bond
        else:
            self.kappa[0] = 0.0   # Zero capillary force
            self.V_b[0] = 0.0     # Zero liquid volume at bond

        # Debonding distance
        self.db[0] = (1.0 + theta/2.0) * self.V_b**(1.0/3.0)

    def setStiffnessNormal(self, k_n):
        '''
        Set the elastic stiffness (`k_n`) in the normal direction of the
        contact.

        :param k_n: The elastic stiffness coefficient [N/m]
        :type k_n: float
        '''
        self.k_n[0] = k_n

    def setStiffnessTangential(self, k_t):
        '''
        Set the elastic stiffness (`k_t`) in the tangential direction of the
        contact.

        :param k_t: The elastic stiffness coefficient [N/m]
        :type k_t: float
        '''
        self.k_t[0] = k_t

    def setYoungsModulus(self, E):
        '''
        Set the elastic Young's modulus (`E`) for the contact model.  This
        parameter is used over normal stiffness (`k_n`) and tangential
        stiffness (`k_t`) when its value is greater than zero. Using this
        parameter produces size-invariant behavior.

        Example values are ~70e9 Pa for quartz,
        http://www.engineeringtoolbox.com/young-modulus-d_417.html

        :param E: The elastic modulus [Pa]
        :type E: float
        '''
        self.E[0] = E

    def setDampingNormal(self, gamma, over_damping=False):
        '''
        Set the dampening coefficient (gamma) in the normal direction of the
        particle-particle contact model. The function will print the fraction
        between the chosen damping and the critical damping value.

        :param gamma: The viscous damping constant [N/(m/s)]
        :type gamma: float
        :param over_damping: Accept overdampening
        :type over_damping: boolean

        See also: :func:`setDampingTangential(gamma)`
        '''
        self.gamma_n[0] = gamma
        critical_gamma = 2.0*numpy.sqrt(self.smallestMass()*self.k_n[0])
        damping_ratio = gamma/critical_gamma
        if damping_ratio < 1.0:
            print('Info: The system is under-dampened (ratio='
                  + str(damping_ratio)
                  + ') in the normal component. \nCritical damping='
                  + str(critical_gamma) + '. This is ok.')
        elif damping_ratio > 1.0:
            if over_damping:
                print('Warning: The system is over-dampened (ratio='
                      + str(damping_ratio) + ') in the normal component. '
                      '\nCritical damping=' + str(critical_gamma) + '.')
            else:
                raise Exception('Warning: The system is over-dampened (ratio='
                                + str(damping_ratio) + ') in the normal '
                                'component.\n'
                                'Call this function once more with '
                                '`over_damping=True` if this is what you want.'
                                '\nCritical damping=' + str(critical_gamma) +
                                '.')
        else:
            print('Warning: The system is critically dampened (ratio=' +
                  str(damping_ratio) + ') in the normal component. '
                  '\nCritical damping=' + str(critical_gamma) + '.')

    def setDampingTangential(self, gamma, over_damping=False):
        '''
        Set the dampening coefficient (gamma) in the tangential direction of the
        particle-particle contact model. The function will print the fraction
        between the chosen damping and the critical damping value.

        :param gamma: The viscous damping constant [N/(m/s)]
        :type gamma: float
        :param over_damping: Accept overdampening
        :type over_damping: boolean

        See also: :func:`setDampingNormal(gamma)`
        '''
        self.gamma_t[0] = gamma
        damping_ratio = gamma/(2.0*numpy.sqrt(self.smallestMass()*self.k_t[0]))
        if damping_ratio < 1.0:
            print('Info: The system is under-dampened (ratio='
                  + str(damping_ratio)
                  + ') in the tangential component. This is ok.')
        elif damping_ratio > 1.0:
            if over_damping:
                print('Warning: The system is over-dampened (ratio='
                      + str(damping_ratio) + ') in the tangential component.')
            else:
                raise Exception('Warning: The system is over-dampened (ratio='
                                + str(damping_ratio) + ') in the tangential '
                                'component.\n'
                                'Call this function once more with '
                                '`over_damping=True` if this is what you want.')
        else:
            print('Warning: The system is critically dampened (ratio='
                  + str(damping_ratio) + ') in the tangential component.')

    def setStaticFriction(self, mu_s):
        '''
        Set the static friction coefficient for particle-particle interactions
        (`self.mu_s`). This value describes the resistance to a shearing motion
        while it is not happenind (contact tangential velocity zero).

        :param mu_s: Value of the static friction coefficient, in [0;inf[.
            Usually between 0 and 1.
        :type mu_s: float

        See also: :func:`setDynamicFriction(mu_d)`
        '''
        self.mu_s[0] = mu_s

    def setDynamicFriction(self, mu_d):
        '''
        Set the dynamic friction coefficient for particle-particle interactions
        (`self.mu_d`). This value describes the resistance to a shearing motion
        while it is happening (contact tangential velocity larger than 0).
        Strain softening can be introduced by having a smaller dynamic
        frictional coefficient than the static fricion coefficient. Usually this
        value is identical to the static friction coefficient.

        :param mu_d: Value of the dynamic friction coefficient, in [0;inf[.
            Usually between 0 and 1.
        :type mu_d: float

        See also: :func:`setStaticFriction(mu_s)`
        '''
        self.mu_d[0] = mu_d

    def setFluidCompressibility(self, beta_f):
        '''
        Set the fluid adiabatic compressibility [1/Pa]. This value is equal to
        `1/K` where `K` is the bulk modulus [Pa]. The value for water is 5.1e-10
        for water at 0 degrees Celcius. This parameter is used for the Darcy
        solver exclusively.

        :param beta_f: The fluid compressibility [1/Pa]
        :type beta_f: float

        See also: :func:`setFluidDensity()` and :func:`setFluidViscosity()`
        '''
        if self.cfd_solver[0] == 1:
            self.beta_f[0] = beta_f
        else:
            raise Exception('setFluidCompressibility() only relevant for the '
                            'Darcy solver (cfd_solver=1)')

    def setFluidViscosity(self, mu):
        '''
        Set the fluid dynamic viscosity [Pa*s]. The value for water is
        1.797e-3 at 0 degrees Celcius. This parameter is used for both the Darcy
        and Navier-Stokes fluid solver.

        :param mu: The fluid dynamic viscosity [Pa*s]
        :type mu: float

        See also: :func:`setFluidDensity()` and
            :func:`setFluidCompressibility()`
        '''
        self.mu[0] = mu

    def setFluidDensity(self, rho_f):
        '''
        Set the fluid density [kg/(m*m*m)]. The value for water is 1000. This
        parameter is used for the Navier-Stokes fluid solver exclusively.

        :param rho_f: The fluid density [kg/(m*m*m)]
        :type rho_f: float

        See also: :func:`setFluidViscosity()` and
            :func:`setFluidCompressibility()`
        '''
        self.rho_f[0] = rho_f

    def scaleSize(self, factor):
        '''
        Scale the positions, linear velocities, forces, torques and radii of all
        particles and mobile walls.

        :param factor: Spatial scaling factor ]0;inf[
        :type factor: float
        '''
        self.L *= factor
        self.x *= factor
        self.radius *= factor
        self.xyzsum *= factor
        self.vel *= factor
        self.force *= factor
        self.torque *= factor
        self.w_x *= factor
        self.w_m *= factor
        self.w_vel *= factor
        self.w_force *= factor

    def bond(self, i, j):
        '''
        Create a bond between particles with index i and j

        :param i: Index of first particle in bond
        :type i: int
        :param j: Index of second particle in bond
        :type j: int
        '''

        self.lambda_bar[0] = 1.0 # Radius multiplier to parallel-bond radii

        if not hasattr(self, 'bonds'):
            self.bonds = numpy.array([[i, j]], dtype=numpy.uint32)
        else:
            self.bonds = numpy.vstack((self.bonds, [i, j]))

        if not hasattr(self, 'bonds_delta_n'):
            self.bonds_delta_n = numpy.array([0.0], dtype=numpy.uint32)
        else:
            #self.bonds_delta_n = numpy.vstack((self.bonds_delta_n, [0.0]))
            self.bonds_delta_n = numpy.append(self.bonds_delta_n, [0.0])

        if not hasattr(self, 'bonds_delta_t'):
            self.bonds_delta_t = numpy.array([[0.0, 0.0, 0.0]], dtype=numpy.uint32)
        else:
            self.bonds_delta_t = numpy.vstack((self.bonds_delta_t,
                                               [0.0, 0.0, 0.0]))

        if not hasattr(self, 'bonds_omega_n'):
            self.bonds_omega_n = numpy.array([0.0], dtype=numpy.uint32)
        else:
            #self.bonds_omega_n = numpy.vstack((self.bonds_omega_n, [0.0]))
            self.bonds_omega_n = numpy.append(self.bonds_omega_n, [0.0])

        if not hasattr(self, 'bonds_omega_t'):
            self.bonds_omega_t = numpy.array([[0.0, 0.0, 0.0]],
                                             dtype=numpy.uint32)
        else:
            self.bonds_omega_t = numpy.vstack((self.bonds_omega_t,
                                               [0.0, 0.0, 0.0]))

        # Increment the number of bonds with one
        self.nb0 += 1

    def currentNormalStress(self, type='defined'):
        '''
        Calculates the current magnitude of the defined or effective top wall
        normal stress.

        :param type: Find the 'defined' (default) or 'effective' normal stress
        :type type: str

        :returns: The current top wall normal stress in Pascal
        :return type: float
        '''
        if type == 'defined':
            return self.w_sigma0[0] \
                    + self.w_sigma0_A[0] \
                    *numpy.sin(2.0*numpy.pi*self.w_sigma0_f[0]\
                    *self.time_current[0])
        elif type == 'effective':
            return self.w_force[0]/(self.L[0]*self.L[1])
        else:
            raise Exception('Normal stress type ' + type + ' not understood')

    def surfaceArea(self, idx):
        '''
        Returns the surface area of a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The surface area of the particle [m^2]
        :return type: float
        '''
        return 4.0*numpy.pi*self.radius[idx]**2

    def volume(self, idx):
        '''
        Returns the volume of a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The volume of the particle [m^3]
        :return type: float
        '''
        return V_sphere(self.radius[idx])

    def mass(self, idx):
        '''
        Returns the mass of a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The mass of the particle [kg]
        :return type: float
        '''
        return self.rho[0]*self.volume(idx)

    def totalMass(self):
        '''
        Returns the total mass of all particles.

        :returns: The total mass  in [kg]
        '''
        m = 0.0
        for i in range(self.np):
            m += self.mass(i)
        return m

    def smallestMass(self):
        '''
        Returns the mass of the leightest particle.

        :param idx: Particle index
        :type idx: int
        :returns: The mass of the particle [kg]
        :return type: float
        '''
        return V_sphere(numpy.min(self.radius))

    def largestMass(self):
        '''
        Returns the mass of the heaviest particle.

        :param idx: Particle index
        :type idx: int
        :returns: The mass of the particle [kg]
        :return type: float
        '''
        return V_sphere(numpy.max(self.radius))

    def momentOfInertia(self, idx):
        '''
        Returns the moment of inertia of a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The moment of inertia [kg*m^2]
        :return type: float
        '''
        return 2.0/5.0*self.mass(idx)*self.radius[idx]**2

    def kineticEnergy(self, idx):
        '''
        Returns the (linear) kinetic energy for a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The kinetic energy of the particle [J]
        :return type: float
        '''
        return 0.5*self.mass(idx) \
          *numpy.sqrt(numpy.dot(self.vel[idx, :], self.vel[idx, :]))**2

    def totalKineticEnergy(self):
        '''
        Returns the total linear kinetic energy for all particles.

        :returns: The kinetic energy of all particles [J]
        '''
        esum = 0.0
        for i in range(self.np):
            esum += self.kineticEnergy(i)
        return esum

    def rotationalEnergy(self, idx):
        '''
        Returns the rotational energy for a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The rotational kinetic energy of the particle [J]
        :return type: float
        '''
        return 0.5*self.momentOfInertia(idx) \
          *numpy.sqrt(numpy.dot(self.angvel[idx, :], self.angvel[idx, :]))**2

    def totalRotationalEnergy(self):
        '''
        Returns the total rotational kinetic energy for all particles.

        :returns: The rotational energy of all particles [J]
        '''
        esum = 0.0
        for i in range(self.np):
            esum += self.rotationalEnergy(i)
        return esum

    def viscousEnergy(self, idx):
        '''
        Returns the viscous dissipated energy for a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The energy lost by the particle by viscous dissipation [J]
        :return type: float
        '''
        return self.ev[idx]

    def totalViscousEnergy(self):
        '''
        Returns the total viscous dissipated energy for all particles.

        :returns: The normal viscous energy lost by all particles [J]
        :return type: float
        '''
        esum = 0.0
        for i in range(self.np):
            esum += self.viscousEnergy(i)
        return esum

    def frictionalEnergy(self, idx):
        '''
        Returns the frictional dissipated energy for a particle.

        :param idx: Particle index
        :type idx: int
        :returns: The frictional energy lost of the particle [J]
        :return type: float
        '''
        return self.es[idx]

    def totalFrictionalEnergy(self):
        '''
        Returns the total frictional dissipated energy for all particles.

        :returns: The total frictional energy lost of all particles [J]
        :return type: float
        '''
        esum = 0.0
        for i in range(self.np):
            esum += self.frictionalEnergy(i)
        return esum

    def energy(self, method):
        '''
        Calculates the sum of the energy components of all particles.

        :param method: The type of energy to return. Possible values are 'pot'
            for potential energy [J], 'kin' for kinetic energy [J], 'rot' for
            rotational energy [J], 'shear' for energy lost by friction,
            'shearrate' for the rate of frictional energy loss [W], 'visc_n' for
            viscous losses normal to the contact [J], 'visc_n_rate' for the rate
            of viscous losses normal to the contact [W], and finally 'bondpot'
            for the potential energy stored in bonds [J]
        :type method: str
        :returns: The value of the selected energy type
        :return type: float
        '''

        if method == 'pot':
            m = numpy.ones(self.np)*4.0/3.0*math.pi*self.radius**3*self.rho
            return numpy.sum(m*math.sqrt(numpy.dot(self.g, self.g))*self.x[:, 2])

        elif method == 'kin':
            m = numpy.ones(self.np)*4.0/3.0*math.pi*self.radius**3*self.rho
            esum = 0.0
            for i in range(self.np):
                esum += 0.5*m[i]*math.sqrt(\
                        numpy.dot(self.vel[i, :], self.vel[i, :]))**2
            return esum

        elif method == 'rot':
            m = numpy.ones(self.np)*4.0/3.0*math.pi*self.radius**3*self.rho
            esum = 0.0
            for i in range(self.np):
                esum += 0.5*2.0/5.0*m[i]*self.radius[i]**2 \
                        *math.sqrt(\
                        numpy.dot(self.angvel[i, :], self.angvel[i, :]))**2
            return esum

        elif method == 'shear':
            return numpy.sum(self.es)

        elif method == 'shearrate':
            return numpy.sum(self.es_dot)

        elif method == 'visc_n':
            return numpy.sum(self.ev)

        elif method == 'visc_n_rate':
            return numpy.sum(self.ev_dot)

        elif method == 'bondpot':
            if self.nb0 > 0:
                R_bar = self.lambda_bar*numpy.minimum(\
                        self.radius[self.bonds[:, 0]],\
                        self.radius[self.bonds[:, 1]])
                A = numpy.pi*R_bar**2
                I = 0.25*numpy.pi*R_bar**4
                J = I*2.0
                bondpot_fn = numpy.sum(\
                        0.5*A*self.k_n*numpy.abs(self.bonds_delta_n)**2)
                bondpot_ft = numpy.sum(\
                        0.5*A*self.k_t*numpy.linalg.norm(self.bonds_delta_t)**2)
                bondpot_tn = numpy.sum(\
                        0.5*J*self.k_t*numpy.abs(self.bonds_omega_n)**2)
                bondpot_tt = numpy.sum(\
                        0.5*I*self.k_n*numpy.linalg.norm(self.bonds_omega_t)**2)
                return bondpot_fn + bondpot_ft + bondpot_tn + bondpot_tt
            else:
                return 0.0
        else:
            raise Exception('Unknownw energy() method "' + method + '"')

    def voidRatio(self):
        '''
        Calculates the current void ratio

        :returns: The void ratio (pore volume relative to solid volume)
        :return type: float
        '''

        # Find the bulk volume
        V_t = (self.L[0] - self.origo[0]) \
                *(self.L[1] - self.origo[1]) \
                *(self.w_x[0] - self.origo[2])

        # Find the volume of solids
        V_s = numpy.sum(4.0/3.0 * math.pi * self.radius**3)

        # Return the void ratio
        e = (V_t - V_s)/V_s
        return e

    def bulkPorosity(self, trim=True):
        '''
        Calculates the bulk porosity of the particle assemblage.

        :param trim: Trim the total volume to the smallest axis-parallel cube
            containing all particles.
        :type trim: bool

        :returns: The bulk porosity, in [0:1]
        :return type: float
        '''

        V_total = 0.0
        if trim:
            min_x = numpy.min(self.x[:, 0] - self.radius)
            min_y = numpy.min(self.x[:, 1] - self.radius)
            min_z = numpy.min(self.x[:, 2] - self.radius)
            max_x = numpy.max(self.x[:, 0] + self.radius)
            max_y = numpy.max(self.x[:, 1] + self.radius)
            max_z = numpy.max(self.x[:, 2] + self.radius)
            V_total = (max_x - min_x)*(max_y - min_y)*(max_z - min_z)

        else:
            if self.nw == 0:
                V_total = self.L[0] * self.L[1] * self.L[2]
            elif self.nw == 1:
                V_total = self.L[0] * self.L[1] * self.w_x[0]
                if V_total <= 0.0:
                    raise Exception("Could not determine total volume")

        # Find the volume of solids
        V_solid = numpy.sum(V_sphere(self.radius))
        return (V_total - V_solid) / V_total

    def porosity(self, slices=10, verbose=False):
        '''
        Calculates the porosity as a function of depth, by averaging values in
        horizontal slabs. Returns porosity values and their corresponding depth.
        The values are calculated using the external ``porosity`` program.

        :param slices: The number of vertical slabs to find porosities in.
        :type slices: int
        :param verbose: Show the file name of the temporary file written to
            disk
        :type verbose: bool
        :returns: A 2d array of depths and their averaged porosities
        :return type: numpy.array
        '''

        # Write data as binary
        self.writebin(verbose=False)

        # Run porosity program on binary
        pipe = subprocess.Popen(["../porosity",\
                                 "-s", "{}".format(slices),
                                 "../input/" + self.sid + ".bin"],
                                stdout=subprocess.PIPE)
        output, err = pipe.communicate()

        if err:
            print(err)
            raise Exception("Could not run external 'porosity' program")

        # read one line of output at a time
        s2 = output.split(b'\n')
        depth = []
        porosity = []
        for row in s2:
            if row != '\n' or row != '' or row != ' ': # skip blank lines
                s3 = row.split(b'\t')
                if s3 != '' and len(s3) == 2: # make sure line has two vals
                    depth.append(float(s3[0]))
                    porosity.append(float(s3[1]))

        return numpy.array(porosity), numpy.array(depth)

    def run(self, verbose=True, hideinputfile=False, dry=False, valgrind=False,
            cudamemcheck=False, device=-1):
        '''
        Start ``sphere`` calculations on the ``sim`` object

        :param verbose: Show ``sphere`` output
        :type verbose: bool
        :param hideinputfile: Hide the file name of the ``sphere`` input file
        :type hideinputfile: bool
        :param dry: Perform a dry run. Important parameter values are shown by
            the ``sphere`` program, and it exits afterwards.
        :type dry: bool
        :param valgrind: Run the program with ``valgrind`` in order to check
            memory leaks in the host code. This causes a significant increase in
            computational time.
        :type valgrind: bool
        :param cudamemcheck: Run the program with ``cudamemcheck`` in order to
            check for device memory leaks and errors. This causes a significant
            increase in computational time.
        :type cudamemcheck: bool
        :param device: Specify the GPU device to execute the program on.
            If not specified, sphere will use the device with the most CUDA cores.
            To see a list of devices, run ``nvidia-smi`` in the system shell.
        :type device: int
        '''

        self.writebin(verbose=False)

        quiet = ""
        stdout = ""
        dryarg = ""
        fluidarg = ""
        devicearg = ""
        valgrindbin = ""
        cudamemchk = ""
        binary = "sphere"
        if not verbose:
            quiet = "-q "
        if hideinputfile:
            stdout = " > /dev/null"
        if dry:
            dryarg = "--dry "
        if valgrind:
            valgrindbin = "valgrind -q --track-origins=yes "
        if cudamemcheck:
            cudamemchk = "cuda-memcheck --leak-check full "
        if self.fluid:
            fluidarg = "--fluid "
        if device != -1:
            devicearg = "-d " + str(device) + " "

        cmd = "cd ..; " + valgrindbin + cudamemchk + "./" + binary + " " \
                + quiet + dryarg + fluidarg + devicearg + \
                "input/" + self.sid + ".bin " + stdout
        #print(cmd)
        status = subprocess.call(cmd, shell=True)

        if status != 0:
            print("Warning: the sphere run returned with status " + str(status))

    def cleanup(self):
        '''
        Removes the input/output files and images belonging to the object
        simulation ID from the ``input/``, ``output/`` and ``img_out/`` folders.
        '''
        cleanup(self)

    def torqueScript(self, email='adc@geo.au.dk', email_alerts='ae',
                     walltime='24:00:00', queue='qfermi',
                     cudapath='/com/cuda/4.0.17/cuda',
                     spheredir='/home/adc/code/sphere',
                     use_workdir=False, workdir='/scratch'):
        '''
        Creates a job script for the Torque queue manager for the simulation
        object.

        :param email: The e-mail address that Torque messages should be sent to
        :type email: str
        :param email_alerts: The type of Torque messages to send to the e-mail
            address. The character 'b' causes a mail to be sent when the
            execution begins. The character 'e' causes a mail to be sent when
            the execution ends normally. The character 'a' causes a mail to be
            sent if the execution ends abnormally. The characters can be written
            in any order.
        :type email_alerts: str
        :param walltime: The maximal allowed time for the job, in the format
            'HH:MM:SS'.
        :type walltime: str
        :param queue: The Torque queue to schedule the job for
        :type queue: str
        :param cudapath: The path of the CUDA library on the cluster compute
            nodes
        :type cudapath: str
        :param spheredir: The path to the root directory of sphere on the
            cluster
        :type spheredir: str
        :param use_workdir: Use a different working directory than the sphere
            folder
        :type use_workdir: bool
        :param workdir: The working directory during the calculations, if
            `use_workdir=True`
        :type workdir: str

        '''

        filename = self.sid + ".sh"
        fh = None
        try:
            fh = open(filename, "w")

            fh.write('#!/bin/sh\n')
            fh.write('#PBS -N ' + self.sid + '\n')
            fh.write('#PBS -l nodes=1:ppn=1\n')
            fh.write('#PBS -l walltime=' + walltime + '\n')
            fh.write('#PBS -q ' + queue + '\n')
            fh.write('#PBS -M ' + email + '\n')
            fh.write('#PBS -m ' + email_alerts + '\n')
            fh.write('CUDAPATH=' + cudapath + '\n')
            fh.write('export PATH=$CUDAPATH/bin:$PATH\n')
            fh.write('export LD_LIBRARY_PATH=$CUDAPATH/lib64'
                     + ':$CUDAPATH/lib:$LD_LIBRARY_PATH\n')
            fh.write('echo "`whoami`@`hostname`"\n')
            fh.write('echo "Start at `date`"\n')
            fh.write('ORIGDIR=' + spheredir + '\n')
            if use_workdir:
                fh.write('WORKDIR=' + workdir + "/$PBS_JOBID\n")
                fh.write('cp -r $ORIGDIR/* $WORKDIR\n')
                fh.write('cd $WORKDIR\n')
            else:
                fh.write('cd ' + spheredir + '\n')
            fh.write('cmake . && make\n')
            fh.write('./sphere input/' + self.sid + '.bin > /dev/null &\n')
            fh.write('wait\n')
            if use_workdir:
                fh.write('cp $WORKDIR/output/* $ORIGDIR/output/\n')
            fh.write('echo "End at `date`"\n')

        finally:
            if fh is not None:
                fh.close()

    def render(self, method="pres", max_val=1e3, lower_cutoff=0.0,
               graphics_format="png", verbose=True):
        '''
        Using the built-in ray tracer, render all output files that belong to
        the simulation, determined by the simulation id (``sid``).

        :param method: The color visualization method to use for the particles.
            Possible values are: 'normal': color all particles with the same
            color, 'pres': color by pressure, 'vel': color by translational
            velocity, 'angvel': color by rotational velocity, 'xdisp': color by
            total displacement along the x-axis, 'angpos': color by angular
            position.
        :type method: str
        :param max_val: The maximum value of the color bar
        :type max_val: float
        :param lower_cutoff: Do not render particles with a value below this
            value, of the field selected by ``method``
        :type lower_cutoff: float
        :param graphics_format: Convert the PPM images generated by the ray
            tracer to this image format using Imagemagick
        :type graphics_format: str
        :param verbose: Show verbose information during ray tracing
        :type verbose: bool
        '''

        print("Rendering {} images with the raytracer".format(self.sid))

        quiet = ""
        if not verbose:
            quiet = "-q"

        # Render images using sphere raytracer
        if method == "normal":
            subprocess.call("cd ..; for F in `ls output/" + self.sid
                            + "*.bin`; do ./sphere " + quiet
                            + " --render $F; done", shell=True)
        else:
            subprocess.call("cd ..; for F in `ls output/" + self.sid
                            + "*.bin`; do ./sphere " + quiet
                            + " --method " + method + " {}".format(max_val)
                            + " -l {}".format(lower_cutoff)
                            + " --render $F; done", shell=True)

        # Convert images to compressed format
        if verbose:
            print('converting images to ' + graphics_format)
        convert(graphics_format=graphics_format)

    def video(self, out_folder="./", video_format="mp4",
              graphics_folder="../img_out/", graphics_format="png", fps=25,
              verbose=False):
        '''
        Uses ffmpeg to combine images to animation. All images should be
        rendered beforehand using :func:`render()`.

        :param out_folder: The output folder for the video file
        :type out_folder: str
        :param video_format: The format of the output video
        :type video_format: str
        :param graphics_folder: The folder containing the rendered images
        :type graphics_folder: str
        :param graphics_format: The format of the rendered images
        :type graphics_format: str
        :param fps: The number of frames per second to use in the video
        :type fps: int
        :param qscale: The output video quality, in ]0;1]
        :type qscale: float
        :param bitrate: The bitrate to use in the output video
        :type bitrate: int
        :param verbose: Show ffmpeg output
        :type verbose: bool
        '''

        video(self.sid, out_folder, video_format, graphics_folder,
              graphics_format, fps, verbose)

    def shearDisplacement(self):
        '''
        Calculates and returns the current shear displacement. The displacement
        is found by determining the total x-axis displacement of the upper,
        fixed particles.

        :returns: The total shear displacement [m]
        :return type: float

        See also: :func:`shearStrain()` and :func:`shearVelocity()`
        '''

        # Displacement of the upper, fixed particles in the shear direction
        #xdisp = self.time_current[0] * self.shearVel()
        fixvel = numpy.nonzero(self.fixvel > 0.0)
        return numpy.max(self.xyzsum[fixvel, 0])

    def shearVelocity(self):
        '''
        Calculates and returns the current shear velocity. The displacement
        is found by determining the total x-axis velocity of the upper,
        fixed particles.

        :returns: The shear velocity [m/s]
        :return type: float

        See also: :func:`shearStrainRate()` and :func:`shearDisplacement()`
        '''
        # Displacement of the upper, fixed particles in the shear direction
        #xdisp = self.time_current[0] * self.shearVel()
        fixvel = numpy.nonzero(self.fixvel > 0.0)
        return numpy.max(self.vel[fixvel, 0])

    def shearVel(self):
        '''
        Alias of :func:`shearVelocity()`
        '''
        return self.shearVelocity()

    def shearStrain(self):
        '''
        Calculates and returns the current shear strain (gamma) value of the
        experiment. The shear strain is found by determining the total x-axis
        displacement of the upper, fixed particles.

        :returns: The total shear strain [-]
        :return type: float

        See also: :func:`shearStrainRate()` and :func:`shearVel()`
        '''

        # Current height
        w_x0 = self.w_x[0]

        # Displacement of the upper, fixed particles in the shear direction
        xdisp = self.shearDisplacement()

        # Return shear strain
        return xdisp/w_x0

    def shearStrainRate(self):
        '''
        Calculates the shear strain rate (dot(gamma)) value of the experiment.

        :returns: The value of dot(gamma)
        :return type: float

        See also: :func:`shearStrain()` and :func:`shearVel()`
        '''
        #return self.shearStrain()/self.time_current[1]

        # Current height
        w_x0 = self.w_x[0]
        v = self.shearVelocity()

        # Return shear strain rate
        return v/w_x0

    def inertiaParameterPlanarShear(self):
        '''
        Returns the value of the inertia parameter $I$ during planar shear
        proposed by GDR-MiDi 2004.

        :returns: The value of $I$
        :return type: float

        See also: :func:`shearStrainRate()` and :func:`shearVel()`
        '''
        return self.shearStrainRate() * numpy.mean(self.radius) \
                * numpy.sqrt(self.rho[0]/self.currentNormalStress())

    def findOverlaps(self):
        '''
        Find all particle-particle overlaps by a n^2 contact search, which is
        done in C++. The particle pair indexes and the distance of the overlaps
        is saved in the object itself as the ``.pairs`` and ``.overlaps``
        members.

        See also: :func:`findNormalForces()`
        '''
        self.writebin(verbose=False)
        subprocess.call('cd .. && ./sphere --contacts input/' + self.sid
                        + '.bin > output/' + self.sid + '.contacts.txt',
                        shell=True)
        contactdata = numpy.loadtxt('../output/' + self.sid + '.contacts.txt')
        self.pairs = numpy.array((contactdata[:, 0], contactdata[:, 1]),
                                 dtype=numpy.int32)
        self.overlaps = numpy.array(contactdata[:, 2])

    def findCoordinationNumber(self):
        '''
        Finds the coordination number (the average number of contacts per
        particle). Requires a previous call to :func:`findOverlaps()`. Values
        are stored in ``self.coordinationnumber``.
        '''
        self.coordinationnumber = numpy.zeros(self.np, dtype=numpy.int)
        for i in numpy.arange(self.overlaps.size):
            self.coordinationnumber[self.pairs[0, i]] += 1
            self.coordinationnumber[self.pairs[1, i]] += 1

    def findMeanCoordinationNumber(self):
        '''
        Returns the coordination number (the average number of contacts per
        particle). Requires a previous call to :func:`findOverlaps()`

        :returns: The mean particle coordination number
        :return type: float
        '''
        return numpy.mean(self.coordinationnumber)

    def findNormalForces(self):
        '''
        Finds all particle-particle overlaps (by first calling
        :func:`findOverlaps()`) and calculating the normal magnitude by
        multiplying the overlaps with the elastic stiffness ``self.k_n``.

        The result is saved in ``self.f_n_magn``.

        See also: :func:`findOverlaps()` and :func:`findContactStresses()`
        '''
        self.findOverlaps()
        self.f_n_magn = self.k_n * numpy.abs(self.overlaps)

    def contactSurfaceArea(self, i, j, overlap):
        '''
        Finds the contact surface area of an inter-particle contact.

        :param i: Index of first particle
        :type i: int or array of ints
        :param j: Index of second particle
        :type j: int or array of ints
        :param d: Overlap distance
        :type d: float or array of floats
        :returns: Contact area [m*m]
        :return type: float or array of floats
        '''
        r_i = self.radius[i]
        r_j = self.radius[j]
        d = r_i + r_j + overlap
        contact_radius = 1./(2.*d)*((-d + r_i - r_j)*(-d - r_i + r_j)*
                                    (-d + r_i + r_j)*(d + r_i + r_j)
                                   )**0.5
        return numpy.pi*contact_radius**2.

    def contactParticleArea(self, i, j):
        '''
        Finds the average area of an two particles in an inter-particle contact.

        :param i: Index of first particle
        :type i: int or array of ints
        :param j: Index of second particle
        :type j: int or array of ints
        :param d: Overlap distance
        :type d: float or array of floats
        :returns: Contact area [m*m]
        :return type: float or array of floats
        '''
        r_bar = (self.radius[i] + self.radius[j])*0.5
        return numpy.pi*r_bar**2.

    def findAllContactSurfaceAreas(self):
        '''
        Finds the contact surface area of an inter-particle contact. This
        function requires a prior call to :func:`findOverlaps()` as it reads
        from the ``self.pairs`` and ``self.overlaps`` arrays.

        :returns: Array of contact surface areas
        :return type: array of floats
        '''
        return self.contactSurfaceArea(self.pairs[0, :], self.pairs[1, :],
                                       self.overlaps)

    def findAllAverageParticlePairAreas(self):
        '''
        Finds the average area of an inter-particle contact. This
        function requires a prior call to :func:`findOverlaps()` as it reads
        from the ``self.pairs`` and ``self.overlaps`` arrays.

        :returns: Array of contact surface areas
        :return type: array of floats
        '''
        return self.contactParticleArea(self.pairs[0, :], self.pairs[1, :])

    def findContactStresses(self, area='average'):
        '''
        Finds all particle-particle uniaxial normal stresses (by first calling
        :func:`findNormalForces()`) and calculating the stress magnitudes by
        dividing the normal force magnitude with the average particle area
        ('average') or by the contact surface area ('contact').

        The result is saved in ``self.sigma_contacts``.

        :param area: Area to use: 'average' (default) or 'contact'
        :type area: str

        See also: :func:`findNormalForces()` and :func:`findOverlaps()`
        '''
        self.findNormalForces()
        if area == 'average':
            areas = self.findAllAverageParticlePairAreas()
        elif area == 'contact':
            areas = self.findAllContactSurfaceAreas()
        else:
            raise Exception('Contact area type "' + area + '" not understood')

        self.sigma_contacts = self.f_n_magn/areas

    def findLoadedContacts(self, threshold):
        '''
        Finds the indices of contact pairs where the contact stress magnitude
        exceeds or is equal to a specified threshold value. This function calls
        :func:`findContactStresses()`.

        :param threshold: Threshold contact stress [Pa]
        :type threshold: float
        :returns: Array of contact indices
        :return type: array of ints
        '''
        self.findContactStresses()
        return numpy.nonzero(self.sigma_contacts >= threshold)

    def forcechains(self, lc=200.0, uc=650.0, outformat='png', disp='2d'):
        '''
        Visualizes the force chains in the system from the magnitude of the
        normal contact forces, and produces an image of them. Warning: Will
        segfault if no contacts are found.

        :param lc: Lower cutoff of contact forces. Contacts below are not
            visualized
        :type lc: float
        :param uc: Upper cutoff of contact forces. Contacts above are
            visualized with this value
        :type uc: float
        :param outformat: Format of output image. Possible values are
            'interactive', 'png', 'epslatex', 'epslatex-color'
        :type outformat: str
        :param disp: Display forcechains in '2d' or '3d'
        :type disp: str
        '''

        self.writebin(verbose=False)

        nd = ''
        if disp == '2d':
            nd = '-2d '

        subprocess.call("cd .. && ./forcechains " + nd + "-f " + outformat
                        + " -lc " + str(lc) + " -uc " + str(uc)
                        + " input/" + self.sid + ".bin > python/tmp.gp",
                        shell=True)
        subprocess.call("gnuplot tmp.gp && rm tmp.gp", shell=True)


    def forcechainsRose(self, lower_limit=0.25, graphics_format='pdf'):
        '''
        Visualize trend and plunge angles of the strongest force chains in a
        rose plot. The plots are saved in the current folder with the name
        'fc-<simulation id>-rose.pdf'.

        :param lower_limit: Do not visualize force chains below this relative
            contact force magnitude, in ]0;1[
        :type lower_limit: float
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        '''
        self.writebin(verbose=False)

        subprocess.call("cd .. && ./forcechains -f txt input/" + self.sid \
                + ".bin > python/fc-tmp.txt", shell=True)

        # data will have the shape (numcontacts, 7)
        data = numpy.loadtxt("fc-tmp.txt", skiprows=1)

        # find the max. value of the normal force
        f_n_max = numpy.amax(data[:, 6])

        # specify the lower limit of force chains to do statistics on
        f_n_lim = lower_limit * f_n_max * 0.6

        # find the indexes of these contacts
        I = numpy.nonzero(data[:, 6] > f_n_lim)

        # loop through these contacts and find the strike and dip of the
        # contacts
        strikelist = [] # strike direction of the normal vector, [0:360[
        diplist = [] # dip of the normal vector, [0:90]
        for i in I[0]:

            x1 = data[i, 0]
            y1 = data[i, 1]
            z1 = data[i, 2]
            x2 = data[i, 3]
            y2 = data[i, 4]
            z2 = data[i, 5]

            if z1 < z2:
                xlower = x1; ylower = y1; zlower = z1
                xupper = x2; yupper = y2; zupper = z2
            else:
                xlower = x2; ylower = y2; zlower = z2
                xupper = x1; yupper = y1; zupper = z1

            # Vector pointing downwards
            dx = xlower - xupper
            dy = ylower - yupper
            dhoriz = numpy.sqrt(dx**2 + dy**2)

            # Find dip angle
            diplist.append(math.degrees(math.atan((zupper - zlower)/dhoriz)))

            # Find strike angle
            if ylower >= yupper: # in first two quadrants
                strikelist.append(math.acos(dx/dhoriz))
            else:
                strikelist.append(2.0*numpy.pi - math.acos(dx/dhoriz))


        plt.figure(figsize=[4, 4])
        ax = plt.subplot(111, polar=True)
        ax.scatter(strikelist, diplist, c='k', marker='+')
        ax.set_rmax(90)
        ax.set_rticks([])
        plt.savefig('fc-' + self.sid + '-rose.' + graphics_format,\
                    transparent=True)

        subprocess.call('rm fc-tmp.txt', shell=True)

    def bondsRose(self, graphics_format='pdf'):
        '''
        Visualize the trend and plunge angles of the bond pairs in a rose plot.
        The plot is saved in the current folder as
        'bonds-<simulation id>-rose.<graphics_format>'.

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        '''
        if not py_mpl:
            print('Error: matplotlib module not found, cannot bondsRose.')
            return
        # loop through these contacts and find the strike and dip of the
        # contacts
        strikelist = [] # strike direction of the normal vector, [0:360[
        diplist = [] # dip of the normal vector, [0:90]
        for n in numpy.arange(self.nb0):

            i = self.bonds[n, 0]
            j = self.bonds[n, 1]

            x1 = self.x[i, 0]
            y1 = self.x[i, 1]
            z1 = self.x[i, 2]
            x2 = self.x[j, 0]
            y2 = self.x[j, 1]
            z2 = self.x[j, 2]

            if z1 < z2:
                xlower = x1; ylower = y1; zlower = z1
                xupper = x2; yupper = y2; zupper = z2
            else:
                xlower = x2; ylower = y2; zlower = z2
                xupper = x1; yupper = y1; zupper = z1

            # Vector pointing downwards
            dx = xlower - xupper
            dy = ylower - yupper
            dhoriz = numpy.sqrt(dx**2 + dy**2)

            # Find dip angle
            diplist.append(math.degrees(math.atan((zupper - zlower)/dhoriz)))

            # Find strike angle
            if ylower >= yupper: # in first two quadrants
                strikelist.append(math.acos(dx/dhoriz))
            else:
                strikelist.append(2.0*numpy.pi - math.acos(dx/dhoriz))

        plt.figure(figsize=[4, 4])
        ax = plt.subplot(111, polar=True)
        ax.scatter(strikelist, diplist, c='k', marker='+')
        ax.set_rmax(90)
        ax.set_rticks([])
        plt.savefig('bonds-' + self.sid + '-rose.' + graphics_format,\
                    transparent=True)

    def status(self):
        '''
        Returns the current simulation status by using the simulation id
        (``sid``) as an identifier.

        :returns: The number of the last output file written
        :return type: int
        '''
        return status(self.sid)

    def momentum(self, idx):
        '''
        Returns the momentum (m*v) of a particle.

        :param idx: The particle index
        :type idx: int
        :returns: The particle momentum [N*s]
        :return type: numpy.array
        '''
        return self.rho*V_sphere(self.radius[idx])*self.vel[idx, :]

    def totalMomentum(self):
        '''
        Returns the sum of particle momentums.

        :returns: The sum of particle momentums (m*v) [N*s]
        :return type: numpy.array
        '''
        m_sum = numpy.zeros(3)
        for i in range(self.np):
            m_sum += self.momentum(i)
        return m_sum

    def sheardisp(self, graphics_format='pdf', zslices=32):
        '''
        Plot the particle x-axis displacement against the original vertical
        particle position. The plot is saved in the current directory with the
        file name '<simulation id>-sheardisp.<graphics_format>'.

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        '''
        if not py_mpl:
            print('Error: matplotlib module not found, cannot sheardisp.')
            return

        # Bin data and error bars for alternative visualization
        h_total = numpy.max(self.x[:, 2]) - numpy.min(self.x[:, 2])
        h_slice = h_total / zslices

        zpos = numpy.zeros(zslices)
        xdisp = numpy.zeros(zslices)
        err = numpy.zeros(zslices)

        for iz in range(zslices):

            # Find upper and lower boundaries of bin
            zlower = iz * h_slice
            zupper = zlower + h_slice

            # Save depth
            zpos[iz] = zlower + 0.5*h_slice

            # Find particle indexes within that slice
            I = numpy.nonzero((self.x[:, 2] > zlower) & (self.x[:, 2] < zupper))

            # Save mean x displacement
            xdisp[iz] = numpy.mean(self.xyzsum[I, 0])

            # Save x displacement standard deviation
            err[iz] = numpy.std(self.xyzsum[I, 0])

        plt.figure(figsize=[4, 4])
        ax = plt.subplot(111)
        ax.scatter(self.xyzsum[:, 0], self.x[:, 2], c='gray', marker='+')
        ax.errorbar(xdisp, zpos, xerr=err,
                    c='black', linestyle='-', linewidth=1.4)
        ax.set_xlabel("Horizontal particle displacement, [m]")
        ax.set_ylabel("Vertical position, [m]")
        plt.savefig(self.sid + '-sheardisp.' + graphics_format,
                    transparent=True)

    def porosities(self, graphics_format='pdf', zslices=16):
        '''
        Plot the averaged porosities with depth. The plot is saved in the format
        '<simulation id>-porosity.<graphics_format>'.

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param zslices: The number of points along the vertical axis to sample
            the porosity in
        :type zslices: int
        '''
        if not py_mpl:
            print('Error: matplotlib module not found, cannot sheardisp.')
            return

        porosity, depth = self.porosity(zslices)

        plt.figure(figsize=[4, 4])
        ax = plt.subplot(111)
        ax.plot(porosity, depth, c='black', linestyle='-', linewidth=1.4)
        ax.set_xlabel('Horizontally averaged porosity, [-]')
        ax.set_ylabel('Vertical position, [m]')
        plt.savefig(self.sid + '-porositiy.' + graphics_format,
                    transparent=True)

    def thinsection_x1x3(self, x2='center', graphics_format='png', cbmax=None,
                         arrowscale=0.01, velarrowscale=1.0, slipscale=1.0,
                         verbose=False):
        '''
        Produce a 2D image of particles on a x1,x3 plane, intersecting the
        second axis at x2. Output is saved as '<sid>-ts-x1x3.txt' in the
        current folder.

        An upper limit to the pressure color bar range can be set by the
        cbmax parameter.

        The data can be plotted in gnuplot with:
            gnuplot> set size ratio -1
            gnuplot> set palette defined (0 "blue", 0.5 "gray", 1 "red")
            gnuplot> plot '<sid>-ts-x1x3.txt' with circles palette fs \
                    transparent solid 0.4 noborder

        This function also saves a plot of the inter-particle slip angles.

        :param x2: The position along the second axis of the intersecting plane
        :type x2: foat
        :param graphics_format: Save the slip angle plot in this format
        :type graphics_format: str
        :param cbmax: The maximal value of the pressure color bar range
        :type cbmax: float
        :param arrowscale: Scale the rotational arrows by this value
        :type arrowscale: float
        :param velarrowscale: Scale the translational arrows by this value
        :type velarrowscale: float
        :param slipscale: Scale the slip arrows by this value
        :type slipscale: float
        :param verbose: Show function output during calculations
        :type verbose: bool
        '''

        if not py_mpl:
            print('Error: matplotlib module not found (thinsection_x1x3).')
            return

        if x2 == 'center':
            x2 = (self.L[1] - self.origo[1]) / 2.0

        # Initialize plot circle positionsr, radii and pressures
        ilist = []
        xlist = []
        ylist = []
        rlist = []
        plist = []
        pmax = 0.0
        rmax = 0.0
        axlist = []
        aylist = []
        daxlist = []
        daylist = []
        dvxlist = []
        dvylist = []
        # Black circle at periphery of particles with angvel[:, 1] > 0.0
        cxlist = []
        cylist = []
        crlist = []

        # Loop over all particles, find intersections
        for i in range(self.np):

            delta = abs(self.x[i, 1] - x2)   # distance between centre and plane

            if delta < self.radius[i]: # if the sphere intersects the plane

                # Store particle index
                ilist.append(i)

                # Store position on plane
                xlist.append(self.x[i, 0])
                ylist.append(self.x[i, 2])

                # Store radius of intersection
                r_circ = math.sqrt(self.radius[i]**2 - delta**2)
                if r_circ > rmax:
                    rmax = r_circ
                rlist.append(r_circ)

                # Store pos. and radius if it is spinning around pos. y
                if self.angvel[i, 1] > 0.0:
                    cxlist.append(self.x[i, 0])
                    cylist.append(self.x[i, 2])
                    crlist.append(r_circ)

                # Store pressure
                pval = self.p[i]
                if cbmax != None:
                    if pval > cbmax:
                        pval = cbmax
                plist.append(pval)

                # Store rotational velocity data for arrows
                # Save two arrows per particle
                axlist.append(self.x[i, 0]) # x starting point of arrow
                axlist.append(self.x[i, 0]) # x starting point of arrow

                # y starting point of arrow
                aylist.append(self.x[i, 2] + r_circ*0.5)

                # y starting point of arrow
                aylist.append(self.x[i, 2] - r_circ*0.5)

                # delta x for arrow end point
                daxlist.append(self.angvel[i, 1]*arrowscale)

                # delta x for arrow end point
                daxlist.append(-self.angvel[i, 1]*arrowscale)
                daylist.append(0.0) # delta y for arrow end point
                daylist.append(0.0) # delta y for arrow end point

                # Store linear velocity data

                # delta x for arrow end point
                dvxlist.append(self.vel[i, 0]*velarrowscale)

                # delta y for arrow end point
                dvylist.append(self.vel[i, 2]*velarrowscale)

                if r_circ > self.radius[i]:
                    raise Exception("Error, circle radius is larger than the "
                                    "particle radius")
                if self.p[i] > pmax:
                    pmax = self.p[i]

        if verbose:
            print("Max. pressure of intersecting spheres: " + str(pmax) + " Pa")
            if cbmax != None:
                print("Value limited to: " + str(cbmax) + " Pa")

        # Save circle data
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3.txt'
        fh = None
        try:
            fh = open(filename, 'w')

            for (x, y, r, p) in zip(xlist, ylist, rlist, plist):
                fh.write("{}\t{}\t{}\t{}\n".format(x, y, r, p))

        finally:
            if fh is not None:
                fh.close()

        # Save circle data for articles spinning with pos. y
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-circ.txt'
        fh = None
        try:
            fh = open(filename, 'w')

            for (x, y, r) in zip(cxlist, cylist, crlist):
                fh.write("{}\t{}\t{}\n".format(x, y, r))

        finally:
            if fh is not None:
                fh.close()

        # Save angular velocity data. The arrow lengths are normalized to max.
        # radius
        #   Output format: x, y, deltax, deltay
        #   gnuplot> plot '-' using 1:2:3:4 with vectors head filled lt 2
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-arrows.txt'
        fh = None
        try:
            fh = open(filename, 'w')

            for (ax, ay, dax, day) in zip(axlist, aylist, daxlist, daylist):
                fh.write("{}\t{}\t{}\t{}\n".format(ax, ay, dax, day))

        finally:
            if fh is not None:
                fh.close()

        # Save linear velocity data
        #   Output format: x, y, deltax, deltay
        #   gnuplot> plot '-' using 1:2:3:4 with vectors head filled lt 2
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-velarrows.txt'
        fh = None
        try:
            fh = open(filename, 'w')

            for (x, y, dvx, dvy) in zip(xlist, ylist, dvxlist, dvylist):
                fh.write("{}\t{}\t{}\t{}\n".format(x, y, dvx, dvy))

        finally:
            if fh is not None:
                fh.close()

        # Check whether there are slips between the particles intersecting the
        # plane
        sxlist = []
        sylist = []
        dsxlist = []
        dsylist = []
        anglelist = [] # angle of the slip vector
        slipvellist = [] # velocity of the slip
        for i in ilist:

            # Loop through other particles, and check whether they are in
            # contact
            for j in ilist:
                #if i < j:
                if i != j:

                    # positions
                    x_i = self.x[i, :]
                    x_j = self.x[j, :]

                    # radii
                    r_i = self.radius[i]
                    r_j = self.radius[j]

                    # Inter-particle vector
                    x_ij = x_i - x_j
                    x_ij_length = numpy.sqrt(x_ij.dot(x_ij))

                    # Check for overlap
                    if x_ij_length - (r_i + r_j) < 0.0:

                        # contact plane normal vector
                        n_ij = x_ij / x_ij_length

                        vel_i = self.vel[i, :]
                        vel_j = self.vel[j, :]
                        angvel_i = self.angvel[i, :]
                        angvel_j = self.angvel[j, :]

                        # Determine the tangential contact surface velocity in
                        # the x,z plane
                        dot_delta = (vel_i - vel_j) \
                                + r_i * numpy.cross(n_ij, angvel_i) \
                                + r_j * numpy.cross(n_ij, angvel_j)

                        # Subtract normal component to get tangential velocity
                        dot_delta_n = n_ij * numpy.dot(dot_delta, n_ij)
                        dot_delta_t = dot_delta - dot_delta_n

                        # Save slip velocity data for gnuplot
                        if dot_delta_t[0] != 0.0 or dot_delta_t[2] != 0.0:

                            # Center position of the contact
                            cpos = x_i - x_ij * 0.5

                            sxlist.append(cpos[0])
                            sylist.append(cpos[2])
                            dsxlist.append(dot_delta_t[0] * slipscale)
                            dsylist.append(dot_delta_t[2] * slipscale)
                            #anglelist.append(math.degrees(\
                                    #math.atan(dot_delta_t[2]/dot_delta_t[0])))
                            anglelist.append(\
                                    math.atan(dot_delta_t[2]/dot_delta_t[0]))
                            slipvellist.append(\
                                    numpy.sqrt(dot_delta_t.dot(dot_delta_t)))


        # Write slip lines to text file
        filename = '../gnuplot/data/' + self.sid + '-ts-x1x3-slips.txt'
        fh = None
        try:
            fh = open(filename, 'w')

            for (sx, sy, dsx, dsy) in zip(sxlist, sylist, dsxlist, dsylist):
                fh.write("{}\t{}\t{}\t{}\n".format(sx, sy, dsx, dsy))

        finally:
            if fh is not None:
                fh.close()

        # Plot thinsection with gnuplot script
        gamma = self.shearStrain()
        subprocess.call('''cd ../gnuplot/scripts && gnuplot -e "sid='{}'; ''' \
                + '''gamma='{:.4}'; xmin='{}'; xmax='{}'; ymin='{}'; ''' \
                + '''ymax='{}'" plotts.gp'''.format(\
                self.sid, self.shearStrain(), self.origo[0], self.L[0], \
                self.origo[2], self.L[2]), shell=True)

        # Find all particles who have a slip velocity higher than slipvel
        slipvellimit = 0.01
        slipvels = numpy.nonzero(numpy.array(slipvellist) > slipvellimit)

        # Bin slip angle data for histogram
        binno = 36/2
        hist_ang, bins_ang = numpy.histogram(numpy.array(anglelist)[slipvels],\
                bins=binno, density=False)
        center_ang = (bins_ang[:-1] + bins_ang[1:]) / 2.0

        center_ang_mirr = numpy.concatenate((center_ang, center_ang + math.pi))
        hist_ang_mirr = numpy.tile(hist_ang, 2)

        # Write slip angles to text file
        #numpy.savetxt(self.sid + '-ts-x1x3-slipangles.txt', zip(center_ang,\
                #hist_ang), fmt="%f\t%f")

        fig = plt.figure()
        ax = fig.add_subplot(111, polar=True)
        ax.bar(center_ang_mirr, hist_ang_mirr, width=30.0/180.0)
        fig.savefig('../img_out/' + self.sid + '-ts-x1x3-slipangles.' +
                    graphics_format)
        fig.clf()

    def plotContacts(self, graphics_format='png', figsize=[4, 4], title=None,
                     lower_limit=0.0, upper_limit=1.0, alpha=1.0,
                     return_data=False, outfolder='.',
                     f_min=None, f_max=None, histogram=True,
                     forcechains=True):
        '''
        Plot current contact orientations on polar plot

        :param lower_limit: Do not visualize force chains below this relative
            contact force magnitude, in ]0;1[
        :type lower_limit: float
        :param upper_limit: Visualize force chains above this relative
            contact force magnitude but cap color bar range, in ]0;1[
        :type upper_limit: float
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        '''

        if not py_mpl:
            print('Error: matplotlib module not found (plotContacts).')
            return

        self.writebin(verbose=False)

        subprocess.call("cd .. && ./forcechains -f txt input/" + self.sid \
                + ".bin > python/contacts-tmp.txt", shell=True)

        # data will have the shape (numcontacts, 7)
        data = numpy.loadtxt('contacts-tmp.txt', skiprows=1)

        # find the max. value of the normal force
        f_n_max = numpy.amax(data[:, 6])

        # specify the lower limit of force chains to do statistics on
        f_n_lim = lower_limit * f_n_max

        if f_min:
            f_n_lim = f_min
        if f_max:
            f_n_max = f_max

        # find the indexes of these contacts
        I = numpy.nonzero(data[:, 6] >= f_n_lim)

        # loop through these contacts and find the strike and dip of the
        # contacts

        # strike direction of the normal vector, [0:360[
        strikelist = numpy.empty(len(I[0]))
        diplist = numpy.empty(len(I[0])) # dip of the normal vector, [0:90]
        forcemagnitude = data[I, 6]
        j = 0
        for i in I[0]:

            x1 = data[i, 0]
            y1 = data[i, 1]
            z1 = data[i, 2]
            x2 = data[i, 3]
            y2 = data[i, 4]
            z2 = data[i, 5]

            if z1 < z2:
                xlower = x1; ylower = y1; zlower = z1
                xupper = x2; yupper = y2; zupper = z2
            else:
                xlower = x2; ylower = y2; zlower = z2
                xupper = x1; yupper = y1; zupper = z1

            # Vector pointing downwards
            dx = xlower - xupper
            dy = ylower - yupper
            dhoriz = numpy.sqrt(dx**2 + dy**2)

            # Find dip angle
            diplist[j] = numpy.degrees(numpy.arctan((zupper - zlower)/dhoriz))

            # Find strike angle
            if ylower >= yupper: # in first two quadrants
                strikelist[j] = numpy.arccos(dx/dhoriz)
            else:
                strikelist[j] = 2.0*numpy.pi - numpy.arccos(dx/dhoriz)

            j += 1

        fig = plt.figure(figsize=figsize)
        ax = plt.subplot(111, polar=True)
        cs = ax.scatter(strikelist, 90. - diplist, marker='o',
                        c=forcemagnitude,
                        s=forcemagnitude/f_n_max*40.,
                        alpha=alpha,
                        edgecolors='none',
                        vmin=f_n_max*lower_limit,
                        vmax=f_n_max*upper_limit,
                        cmap=matplotlib.cm.get_cmap('afmhot_r'))
        plt.colorbar(cs, extend='max')

        # plot defined max compressive stress from tau/N ratio
        ax.scatter(0., # prescribed stress
                   numpy.degrees(numpy.arctan(self.shearStress('defined')/
                                              self.currentNormalStress('defined'))),
                   marker='o', c='none', edgecolor='blue', s=300)
        ax.scatter(0., # actual stress
                   numpy.degrees(numpy.arctan(self.shearStress('effective')/
                                              self.currentNormalStress('effective'))),
                   marker='+', color='blue', s=300)

        ax.set_rmax(90)
        ax.set_rticks([])

        if title:
            plt.title(title)
        else:
            plt.title('t={:.2f} s'.format(self.currentTime()))

        #plt.tight_layout()
        plt.savefig(outfolder + '/contacts-' + self.sid + '-' + \
                    str(self.time_step_count[0]) + '.' + \
                graphics_format,\
                transparent=False)

        subprocess.call('rm contacts-tmp.txt', shell=True)

        fig.clf()
        if histogram:
            #hist, bins = numpy.histogram(datadata[:, 6], bins=10)
            _, _, _ = plt.hist(data[:, 6], alpha=0.75, facecolor='gray')
            #plt.xlabel('$\\boldsymbol{f}_\text{n}$ [N]')
            plt.yscale('log', nonposy='clip')
            plt.xlabel('Contact load [N]')
            plt.ylabel('Count $N$')
            plt.grid(True)
            plt.savefig(outfolder + '/contacts-hist-' + self.sid + '-' + \
                        str(self.time_step_count[0]) + '.' + \
                    graphics_format,\
                    transparent=False)
            plt.clf()

            # angle: 0 when vertical, 90 when horizontal
            #hist, bins = numpy.histogram(datadata[:, 6], bins=10)
            _, _, _ = plt.hist(90. - diplist, bins=range(0, 100, 10),
                               alpha=0.75, facecolor='gray')
            theta_sigma1 = numpy.degrees(numpy.arctan(
                self.currentNormalStress('defined')/\
                self.shearStress('defined')))
            plt.axvline(90. - theta_sigma1, color='k', linestyle='dashed',
                        linewidth=1)
            plt.xlim([0, 90.])
            plt.ylim([0, self.np/10])
            #plt.xlabel('$\\boldsymbol{f}_\text{n}$ [N]')
            plt.xlabel('Contact angle [deg]')
            plt.ylabel('Count $N$')
            plt.grid(True)
            plt.savefig(outfolder + '/dip-' + self.sid + '-' + \
                        str(self.time_step_count[0]) + '.' + \
                    graphics_format,\
                    transparent=False)
            plt.clf()

        if forcechains:

            #color = matplotlib.cm.spectral(data[:, 6]/f_n_max)
            for i in I[0]:

                x1 = data[i, 0]
                #y1 = data[i, 1]
                z1 = data[i, 2]
                x2 = data[i, 3]
                #y2 = data[i, 4]
                z2 = data[i, 5]
                f_n = data[i, 6]

                lw_max = 1.0
                if f_n >= f_n_max:
                    lw = lw_max
                else:
                    lw = (f_n - f_n_lim)/(f_n_max - f_n_lim)*lw_max

                #print lw
                plt.plot([x1, x2], [z1, z2], '-k', linewidth=lw)

            axfc1 = plt.gca()
            axfc1.spines['right'].set_visible(False)
            axfc1.spines['left'].set_visible(False)
            # Only show ticks on the left and bottom spines
            axfc1.xaxis.set_ticks_position('none')
            axfc1.yaxis.set_ticks_position('none')
            #axfc1.set_xticklabels([])
            #axfc1.set_yticklabels([])
            axfc1.set_xlim([self.origo[0], self.L[0]])
            axfc1.set_ylim([self.origo[2], self.L[2]])
            axfc1.set_aspect('equal')

            plt.xlabel('$x$ [m]')
            plt.ylabel('$z$ [m]')
            plt.grid(False)
            plt.savefig(outfolder + '/fc-' + self.sid + '-' + \
                        str(self.time_step_count[0]) + '.' + \
                    graphics_format,\
                    transparent=False)

        plt.close()

        if return_data:
            return data, strikelist, diplist, forcemagnitude, alpha, f_n_max

    def plotFluidPressuresY(self, y=-1, graphics_format='png', verbose=True):
        '''
        Plot fluid pressures in a plane normal to the second axis.
        The plot is saved in the current folder with the format
        'p_f-<simulation id>-y<y value>.<graphics_format>'.

        :param y: Plot pressures in fluid cells with these y axis values. If
            this value is -1, the center y position is used.
        :type y: int
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool

        See also: :func:`writeFluidVTK()` and :func:`plotFluidPressuresZ()`
        '''

        if not py_mpl:
            print('Error: matplotlib module not found (plotFluidPressuresY).')
            return

        if y == -1:
            y = self.num[1]/2

        plt.figure(figsize=[8, 8])
        plt.title('Fluid pressures')
        imgplt = plt.imshow(self.p_f[:, y, :].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_3$')
        plt.colorbar()
        filename = 'p_f-' + self.sid + '-y' + str(y) + '.' + graphics_format
        plt.savefig(filename, transparent=False)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close()

    def plotFluidPressuresZ(self, z=-1, graphics_format='png', verbose=True):
        '''
        Plot fluid pressures in a plane normal to the third axis.
        The plot is saved in the current folder with the format
        'p_f-<simulation id>-z<z value>.<graphics_format>'.

        :param z: Plot pressures in fluid cells with these z axis values. If
            this value is -1, the center z position is used.
        :type z: int
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool

        See also: :func:`writeFluidVTK()` and :func:`plotFluidPressuresY()`
        '''

        if not py_mpl:
            print('Error: matplotlib module not found (plotFluidPressuresZ).')
            return

        if z == -1:
            z = self.num[2]/2

        plt.figure(figsize=[8, 8])
        plt.title('Fluid pressures')
        imgplt = plt.imshow(self.p_f[:, :, z].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar()
        filename = 'p_f-' + self.sid + '-z' + str(z) + '.' + graphics_format
        plt.savefig(filename, transparent=False)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close()

    def plotFluidVelocitiesY(self, y=-1, graphics_format='png', verbose=True):
        '''
        Plot fluid velocities in a plane normal to the second axis.
        The plot is saved in the current folder with the format
        'v_f-<simulation id>-z<z value>.<graphics_format>'.

        :param y: Plot velocities in fluid cells with these y axis values. If
            this value is -1, the center y position is used.
        :type y: int
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool

        See also: :func:`writeFluidVTK()` and :func:`plotFluidVelocitiesZ()`
        '''

        if not py_mpl:
            print('Error: matplotlib module not found (plotFluidVelocitiesY).')
            return

        if y == -1:
            y = self.num[1]/2

        plt.title('Fluid velocities')
        plt.figure(figsize=[8, 8])

        plt.subplot(131)
        imgplt = plt.imshow(self.v_f[:, y, :, 0].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_1$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_3$')
        plt.colorbar(orientation='horizontal')

        plt.subplot(132)
        imgplt = plt.imshow(self.v_f[:, y, :, 1].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_2$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_3$')
        plt.colorbar(orientation='horizontal')

        plt.subplot(133)
        imgplt = plt.imshow(self.v_f[:, y, :, 2].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_3$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_3$')
        plt.colorbar(orientation='horizontal')

        filename = 'v_f-' + self.sid + '-y' + str(y) + '.' + graphics_format
        plt.savefig(filename, transparent=False)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close()

    def plotFluidVelocitiesZ(self, z=-1, graphics_format='png', verbose=True):
        '''
        Plot fluid velocities in a plane normal to the third axis.
        The plot is saved in the current folder with the format
        'v_f-<simulation id>-z<z value>.<graphics_format>'.

        :param z: Plot velocities in fluid cells with these z axis values. If
            this value is -1, the center z position is used.
        :type z: int
        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool

        See also: :func:`writeFluidVTK()` and :func:`plotFluidVelocitiesY()`
        '''
        if not py_mpl:
            print('Error: matplotlib module not found (plotFluidVelocitiesZ).')
            return

        if z == -1:
            z = self.num[2]/2

        plt.title("Fluid velocities")
        plt.figure(figsize=[8, 8])

        plt.subplot(131)
        imgplt = plt.imshow(self.v_f[:, :, z, 0].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_1$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar(orientation='horizontal')

        plt.subplot(132)
        imgplt = plt.imshow(self.v_f[:, :, z, 1].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_2$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar(orientation='horizontal')

        plt.subplot(133)
        imgplt = plt.imshow(self.v_f[:, :, z, 2].T, origin='lower')
        imgplt.set_interpolation('nearest')
        #imgplt.set_interpolation('bicubic')
        #imgplt.set_cmap('hot')
        plt.title("$v_3$")
        plt.xlabel('$x_1$')
        plt.ylabel('$x_2$')
        plt.colorbar(orientation='horizontal')

        filename = 'v_f-' + self.sid + '-z' + str(z) + '.' + graphics_format
        plt.savefig(filename, transparent=False)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close()

    def plotFluidDiffAdvPresZ(self, graphics_format='png', verbose=True):
        '''
        Compare contributions to the velocity from diffusion and advection,
        assuming the flow is 1D along the z-axis, phi=1, and dphi=0. This
        solution is analog to the predicted velocity and not constrained by the
        conservation of mass. The plot is saved in the output folder with the
        name format '<simulation id>-diff_adv-t=<current time>s-mu=<dynamic
        viscosity>Pa-s.<graphics_format>'.

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool
        '''
        if not py_mpl:
            print('Error: matplotlib module not found (plotFluidDiffAdvPresZ).')
            return

        # The v_z values are read from self.v_f[0, 0, :, 2]
        dz = self.L[2]/self.num[2]
        rho = self.rho_f

        # Central difference gradients
        dvz_dz = (self.v_f[0, 0, 1:, 2] - self.v_f[0, 0, :-1, 2])/(2.0*dz)
        dvzvz_dz = (self.v_f[0, 0, 1:, 2]**2 - self.v_f[0, 0, :-1, 2]**2)\
                   /(2.0*dz)

        # Diffusive contribution to velocity change
        dvz_diff = 2.0*self.mu/rho*dvz_dz*self.time_dt

        # Advective contribution to velocity change
        dvz_adv = dvzvz_dz*self.time_dt

        # Pressure gradient
        dp_dz = (self.p_f[0, 0, 1:] - self.p_f[0, 0, :-1])/(2.0*dz)

        cellno = numpy.arange(1, self.num[2])

        fig = plt.figure()
        titlesize = 12

        plt.subplot(1, 3, 1)
        plt.title('Pressure', fontsize=titlesize)
        plt.ylabel('$i_z$')
        plt.xlabel('$p_z$')
        plt.plot(self.p_f[0, 0, :], numpy.arange(self.num[2]))
        plt.grid()

        plt.subplot(1, 3, 2)
        plt.title('Pressure gradient', fontsize=titlesize)
        plt.ylabel('$i_z$')
        plt.xlabel('$\Delta p_z$')
        plt.plot(dp_dz, cellno)
        plt.grid()

        plt.subplot(1, 3, 3)
        plt.title('Velocity prediction terms', fontsize=titlesize)
        plt.ylabel('$i_z$')
        plt.xlabel('$\Delta v_z$')
        plt.plot(dvz_diff, cellno, label='Diffusion')
        plt.plot(dvz_adv, cellno, label='Advection')
        plt.plot(dvz_diff+dvz_adv, cellno, '--', label='Sum')
        leg = plt.legend(loc='best', prop={'size':8})
        leg.get_frame().set_alpha(0.5)
        plt.grid()

        plt.tight_layout()
        filename = '../output/{}-diff_adv-t={:.2e}s-mu={:.2e}Pa-s.{}'\
                   .format(self.sid, self.time_current[0], self.mu[0],
                           graphics_format)
        plt.savefig(filename)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close(fig)

    def ReynoldsNumber(self):
        '''
        Estimate the per-cell Reynolds number by: Re=rho * ||v_f|| * dx/mu.
        This value is returned and also stored in `self.Re`.

        :returns: Reynolds number
        :return type: Numpy array with dimensions like the fluid grid
        '''

        # find magnitude of fluid velocity vectors
        self.v_f_magn = numpy.empty_like(self.p_f)
        for z in numpy.arange(self.num[2]):
            for y in numpy.arange(self.num[1]):
                for x in numpy.arange(self.num[0]):
                    self.v_f_magn[x, y, z] = \
                            self.v_f[x, y, z, :].dot(self.v_f[x, y, z, :])

        Re = self.rho_f*self.v_f_magn*self.L[0]/self.num[0]/(self.mu + \
                1.0e-16)
        return Re

    def plotLoadCurve(self, graphics_format='png', verbose=True):
        '''
        Plot the load curve (log time vs. upper wall movement).  The plot is
        saved in the current folder with the file name
        '<simulation id>-loadcurve.<graphics_format>'.
        The consolidation coefficient calculations are done on the base of
        Bowles 1992, p. 129--139, using the "Casagrande" method.
        It is assumed that the consolidation has stopped at the end of the
        simulation (i.e. flat curve).

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool
        '''
        if not py_mpl:
            print('Error: matplotlib module not found (plotLoadCurve).')
            return

        t = numpy.empty(self.status())
        H = numpy.empty_like(t)
        sb = sim(self.sid, fluid=self.fluid)
        sb.readfirst(verbose=False)
        for i in numpy.arange(1, self.status()+1):
            sb.readstep(i, verbose=False)
            if i == 0:
                load = sb.w_sigma0[0]
            t[i-1] = sb.time_current[0]
            H[i-1] = sb.w_x[0]

        # find consolidation parameters
        H0 = H[0]
        H100 = H[-1]
        H50 = (H0 + H100)/2.0
        T50 = 0.197 # case I

        # find the time where 50% of the consolidation (H50) has happened by
        # linear interpolation. The values in H are expected to be
        # monotonically decreasing. See Numerical Recipies p. 115
        i_lower = 0
        i_upper = self.status()-1
        while i_upper - i_lower > 1:
            i_midpoint = int((i_upper + i_lower)/2)
            if H50 < H[i_midpoint]:
                i_lower = i_midpoint
            else:
                i_upper = i_midpoint
        t50 = t[i_lower] + (t[i_upper] - t[i_lower]) * \
                (H50 - H[i_lower])/(H[i_upper] - H[i_lower])

        c_coeff = T50*H50**2.0/(t50)
        if self.fluid:
            e = numpy.mean(sb.phi[:, :, 3:-8]) # ignore boundaries
        else:
            e = sb.voidRatio()

        phi_bar = e
        fig = plt.figure()
        plt.xlabel('Time [s]')
        plt.ylabel('Height [m]')
        plt.title('$c_v$=%.2e m$^2$ s$^{-1}$ at %.1f kPa and $e$=%.2f' \
                % (c_coeff, sb.w_sigma0[0]/1000.0, e))
        plt.semilogx(t, H, '+-')
        plt.axhline(y=H0, color='gray')
        plt.axhline(y=H50, color='gray')
        plt.axhline(y=H100, color='gray')
        plt.axvline(x=t50, color='red')
        plt.grid()
        filename = self.sid + '-loadcurve.' + graphics_format
        plt.savefig(filename)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close(fig)

    def convergence(self):
        '''
        Read the convergence evolution in the CFD solver. The values are stored
        in `self.conv` with iteration number in the first column and iteration
        count in the second column.

        See also: :func:`plotConvergence()`
        '''
        return numpy.loadtxt('../output/' + self.sid + '-conv.log', dtype=numpy.int32)

    def plotConvergence(self, graphics_format='png', verbose=True):
        '''
        Plot the convergence evolution in the CFD solver. The plot is saved
        in the output folder with the file name
        '<simulation id>-conv.<graphics_format>'.

        :param graphics_format: Save the plot in this format
        :type graphics_format: str
        :param verbose: Print output filename after saving
        :type verbose: bool

        See also: :func:`convergence()`
        '''
        if not py_mpl:
            print('Error: matplotlib module not found (plotConvergence).')
            return

        fig = plt.figure()
        conv = self.convergence()

        plt.title('Convergence evolution in CFD solver in "' + self.sid + '"')
        plt.xlabel('Time step')
        plt.ylabel('Jacobi iterations')
        plt.plot(conv[:, 0], conv[:, 1])
        plt.grid()
        filename = self.sid + '-conv.' + graphics_format
        plt.savefig(filename)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close(fig)

    def plotSinFunction(self, baseval, A, f, phi=0.0, xlabel='$t$ [s]',
                        ylabel='$y$', plotstyle='.', outformat='png',
                        verbose=True):
        '''
        Plot the values of a sinusoidal modulated base value. Saves the output
        as a plot in the current folder.
        The time values will range from `self.time_current` to
        `self.time_total`.

        :param baseval: The center value which the sinusoidal fluctuations are
            modulating
        :type baseval: float
        :param A: The fluctuation amplitude
        :type A: float
        :param phi: The phase shift [s]
        :type phi: float
        :param xlabel: The label for the x axis
        :type xlabel: str
        :param ylabel: The label for the y axis
        :type ylabel: str
        :param plotstyle: Matplotlib-string for specifying plotting style
        :type plotstyle: str
        :param outformat: File format of the output plot
        :type outformat: str
        :param verbose: Print output filename after saving
        :type verbose: bool
        '''
        if not py_mpl:
            print('Error: matplotlib module not found (plotSinFunction).')
            return

        fig = plt.figure(figsize=[8, 6])
        steps_left = (self.time_total[0] - self.time_current[0]) \
                /self.time_file_dt[0]
        t = numpy.linspace(self.time_current[0], self.time_total[0], steps_left)
        f = baseval + A*numpy.sin(2.0*numpy.pi*f*t + phi)
        plt.plot(t, f, plotstyle)
        plt.grid()
        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.tight_layout()
        filename = self.sid + '-sin.' + outformat
        plt.savefig(filename)
        if verbose:
            print(filename)
        plt.clf()
        plt.close(fig)

    def setTopWallNormalStressModulation(self, A, f, plot=False):
        '''
        Set the parameters for the sine wave modulating the normal stress
        at the top wall. Note that a cos-wave is obtained with phi=pi/2.

        :param A: Fluctuation amplitude [Pa]
        :type A: float
        :param f: Fluctuation frequency [Hz]
        :type f: float
        :param plot: Show a plot of the resulting modulation
        :type plot: bool

        See also: :func:`setFluidPressureModulation()` and
        :func:`disableTopWallNormalStressModulation()`
        '''
        self.w_sigma0_A[0] = A
        self.w_sigma0_f[0] = f

        if plot and py_mpl:
            self.plotSinFunction(self.w_sigma0[0], A, f, phi=0.0,
                                 xlabel='$t$ [s]', ylabel='$\\sigma_0$ [Pa]')

    def disableTopWallNormalStressModulation(self):
        '''
        Set the parameters for the sine wave modulating the normal stress
        at the top dynamic wall to zero.

        See also: :func:`setTopWallNormalStressModulation()`
        '''
        self.setTopWallNormalStressModulation(A=0.0, f=0.0)

    def setFluidPressureModulation(self, A, f, phi=0.0, plot=False):
        '''
        Set the parameters for the sine wave modulating the fluid pressures
        at the top boundary. Note that a cos-wave is obtained with phi=pi/2.

        :param A: Fluctuation amplitude [Pa]
        :type A: float
        :param f: Fluctuation frequency [Hz]
        :type f: float
        :param phi: Fluctuation phase shift (default=0.0) [rad]
        :type phi: float
        :param plot: Show a plot of the resulting modulation
        :type plot: bool

        See also: :func:`setTopWallNormalStressModulation()` and
        :func:`disableFluidPressureModulation()`
        '''
        self.p_mod_A[0] = A
        self.p_mod_f[0] = f
        self.p_mod_phi[0] = phi

        if plot:
            self.plotSinFunction(self.p_f[0, 0, -1], A, f, phi=0.0,
                                 xlabel='$t$ [s]', ylabel='$p_f$ [kPa]')

    def disableFluidPressureModulation(self):
        '''
        Set the parameters for the sine wave modulating the fluid pressures
        at the top boundary to zero.

        See also: :func:`setFluidPressureModulation()`
        '''
        self.setFluidPressureModulation(A=0.0, f=0.0)

    def plotPrescribedFluidPressures(self, graphics_format='png',
                                     verbose=True):
        '''
        Plot the prescribed fluid pressures through time that may be
        modulated through the class parameters p_mod_A, p_mod_f, and p_mod_phi.
        The plot is saved in the output folder with the file name
        '<simulation id>-pres.<graphics_format>'.
        '''
        if not py_mpl:
            print('Error: matplotlib module not found ' +
                  '(plotPrescribedFluidPressures).')
            return

        fig = plt.figure()

        plt.title('Prescribed fluid pressures at the top in "' + self.sid + '"')
        plt.xlabel('Time [s]')
        plt.ylabel('Pressure [Pa]')
        t = numpy.linspace(0, self.time_total, self.time_total/self.time_file_dt)
        p = self.p_f[0, 0, -1] + self.p_mod_A * \
            numpy.sin(2.0*numpy.pi*self.p_mod_f*t + self.p_mod_phi)
        plt.plot(t, p, '.-')
        plt.grid()
        filename = '../output/' + self.sid + '-pres.' + graphics_format
        plt.savefig(filename)
        if verbose:
            print('saved to ' + filename)
        plt.clf()
        plt.close(fig)

    def acceleration(self, idx=-1):
        '''
        Returns the acceleration of one or more particles, selected by their
        index. If the index is equal to -1 (default value), all accelerations
        are returned.

        :param idx: Index or index range of particles
        :type idx: int, list or numpy.array
        :returns: n-by-3 matrix of acceleration(s)
        :return type: numpy.array
        '''
        if idx == -1:
            idx = range(self.np)
        return self.force[idx, :]/(V_sphere(self.radius[idx])*self.rho[0]) + \
                self.g

    def setGamma(self, gamma):
        '''
        Gamma is a fluid solver parameter, used for smoothing the pressure
        values. The epsilon (pressure) values are smoothed by including the
        average epsilon value of the six closest (face) neighbor cells. This
        parameter should be in the range [0.0;1.0[. The higher the value, the
        more averaging is introduced. A value of 0.0 disables all averaging.

        The default and recommended value is 0.0.

        :param theta: The smoothing parameter value
        :type theta: float

        Other solver parameter setting functions: :func:`setTheta()`,
        :func:`setBeta()`, :func:`setTolerance()`,
        :func:`setDEMstepsPerCFDstep()` and :func:`setMaxIterations()`
        '''
        self.gamma = numpy.asarray(gamma)

    def setTheta(self, theta):
        '''
        Theta is a fluid solver under-relaxation parameter, used in solution of
        Poisson equation. The value should be within the range ]0.0;1.0]. At a
        value of 1.0, the new estimate of epsilon values is used exclusively. At
        lower values, a linear interpolation between new and old values is used.
        The solution typically converges faster with a value of 1.0, but
        instabilities may be avoided with lower values.

        The default and recommended value is 1.0.

        :param theta: The under-relaxation parameter value
        :type theta: float

        Other solver parameter setting functions: :func:`setGamma()`,
        :func:`setBeta()`, :func:`setTolerance()`,
        :func:`setDEMstepsPerCFDstep()` and :func:`setMaxIterations()`
        '''
        self.theta = numpy.asarray(theta)


    def setBeta(self, beta):
        '''
        Beta is a fluid solver parameter, used in velocity prediction and
        pressure iteration 1.0: Use old pressures for fluid velocity prediction
        (see Langtangen et al. 2002) 0.0: Do not use old pressures for fluid
        velocity prediction (Chorin's original projection method, see Chorin
        (1968) and "Projection method (fluid dynamics)" page on Wikipedia.  The
        best results precision and performance-wise are obtained by using a beta
        of 0 and a low tolerance criteria value.

        The default and recommended value is 0.0.

        Other solver parameter setting functions: :func:`setGamma()`,
        :func:`setTheta()`, :func:`setTolerance()`,
        :func:`setDEMstepsPerCFDstep()` and
        :func:`setMaxIterations()`
        '''
        self.beta = numpy.asarray(beta)

    def setTolerance(self, tolerance):
        '''
        A fluid solver parameter, the value of the tolerance parameter denotes
        the required value of the maximum normalized residual for the fluid
        solver.

        The default and recommended value is 1.0e-3.

        :param tolerance: The tolerance criteria for the maximal normalized
            residual
        :type tolerance: float

        Other solver parameter setting functions: :func:`setGamma()`,
        :func:`setTheta()`, :func:`setBeta()`, :func:`setDEMstepsPerCFDstep()` and
        :func:`setMaxIterations()`
        '''
        self.tolerance = numpy.asarray(tolerance)

    def setMaxIterations(self, maxiter):
        '''
        A fluid solver parameter, the value of the maxiter parameter denotes the
        maximal allowed number of fluid solver iterations before ending the
        fluid solver loop prematurely. The residual values are at that point not
        fulfilling the tolerance criteria. The parameter is included to avoid
        infinite hangs.

        The default and recommended value is 1e4.

        :param maxiter: The maximum number of Jacobi iterations in the fluid
            solver
        :type maxiter: int

        Other solver parameter setting functions: :func:`setGamma()`,
        :func:`setTheta()`, :func:`setBeta()`, :func:`setDEMstepsPerCFDstep()`
        and :func:`setTolerance()`
        '''
        self.maxiter = numpy.asarray(maxiter)

    def setDEMstepsPerCFDstep(self, ndem):
        '''
        A fluid solver parameter, the value of the maxiter parameter denotes the
        number of DEM time steps to be performed per CFD time step.

        The default value is 1.

        :param ndem: The DEM/CFD time step ratio
        :type ndem: int

        Other solver parameter setting functions: :func:`setGamma()`,
        :func:`setTheta()`, :func:`setBeta()`, :func:`setTolerance()` and
        :func:`setMaxIterations()`.
        '''
        self.ndem = numpy.asarray(ndem)

    def shearStress(self, type='effective'):
        '''
        Calculates the sum of shear stress values measured on any moving
        particles with a finite and fixed velocity.

        :param type: Find the 'defined' or 'effective' (default) shear stress
        :type type: str

        :returns: The shear stress in Pa
        :return type: numpy.array
        '''

        if type == 'defined':
            return self.w_tau_x[0]

        elif type == 'effective':

            fixvel = numpy.nonzero(self.fixvel > 0.0)
            force = numpy.zeros(3)

            # Summation of shear stress contributions
            for i in fixvel[0]:
                if self.vel[i, 0] > 0.0:
                    force += -self.force[i, :]

            return force[0]/(self.L[0]*self.L[1])

        else:
            raise Exception('Shear stress type ' + type + ' not understood')


    def visualize(self, method='energy', savefig=True, outformat='png',
                  figsize=False, pickle=False, xlim=False, firststep=0,
                  f_min=None, f_max=None, cmap=None, smoothing=0,
                  smoothing_window='hanning'):
        '''
        Visualize output from the simulation, where the temporal progress is
        of interest. The output will be saved in the current folder with a name
        combining the simulation id of the simulation, and the visualization
        method.

        :param method: The type of plot to render. Possible values are 'energy',
            'walls', 'triaxial', 'inertia', 'mean-fluid-pressure',
            'fluid-pressure', 'shear', 'shear-displacement', 'porosity',
            'rate-dependence', 'contacts'
        :type method: str
        :param savefig: Save the image instead of showing it on screen
        :type savefig: bool
        :param outformat: The output format of the plot data. This can be an
            image format, or in text ('txt').
        :param figsize: Specify output figure size in inches
        :type figsize: array
        :param pickle: Save all figure content as a Python pickle file. It can
            be opened later using `fig=pickle.load(open('file.pickle','rb'))`.
        :type pickle: bool
        :param xlim: Set custom limits to the x axis. If not specified, the x
            range will correspond to the entire data interval.
        :type xlim: array
        :param firststep: The first output file step to read (default: 0)
        :type firststep: int
        :param cmap: Choose custom color map, e.g.
            `cmap=matplotlib.cm.get_cmap('afmhot')`
        :type cmap: matplotlib.colors.LinearSegmentedColormap
        :param smoothing: Apply smoothing across a number of output files to the
            `method='shear'` plot. A value of less than 3 means that no
            smoothing occurs.
        :type smoothing: int
        :param smoothing_window: Type of smoothing to use when `smoothing >= 3`.
            Valid values are 'flat', 'hanning' (default), 'hamming', 'bartlett',
            and 'blackman'.
        :type smoothing_window: str
        '''

        lastfile = self.status()
        sb = sim(sid=self.sid, np=self.np, nw=self.nw, fluid=self.fluid)

        if not py_mpl:
            print('Error: matplotlib module not found (visualize).')
            return

        ### Plotting
        if outformat != 'txt':
            if figsize:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure(figsize=(8, 8))

        if method == 'energy':
            if figsize:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure(figsize=(20, 8))

            # Allocate arrays
            t = numpy.zeros(lastfile-firststep + 1)
            Epot = numpy.zeros_like(t)
            Ekin = numpy.zeros_like(t)
            Erot = numpy.zeros_like(t)
            Es = numpy.zeros_like(t)
            Ev = numpy.zeros_like(t)
            Es_dot = numpy.zeros_like(t)
            Ev_dot = numpy.zeros_like(t)
            Ebondpot = numpy.zeros_like(t)
            Esum = numpy.zeros_like(t)

            # Read energy values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                Epot[i] = sb.energy("pot")
                Ekin[i] = sb.energy("kin")
                Erot[i] = sb.energy("rot")
                Es[i] = sb.energy("shear")
                Ev[i] = sb.energy("visc_n")
                Es_dot[i] = sb.energy("shearrate")
                Ev_dot[i] = sb.energy("visc_n_rate")
                Ebondpot[i] = sb.energy("bondpot")
                Esum[i] = Epot[i] + Ekin[i] + Erot[i] + Es[i] + Ev[i] +\
                        Ebondpot[i]
                t[i] = sb.currentTime()


            if outformat != 'txt':
                # Potential energy
                ax1 = plt.subplot2grid((2, 5), (0, 0))
                ax1.set_xlabel('Time [s]')
                ax1.set_ylabel('Total potential energy [J]')
                ax1.plot(t, Epot, '+-')
                ax1.grid()

                # Kinetic energy
                ax2 = plt.subplot2grid((2, 5), (0, 1))
                ax2.set_xlabel('Time [s]')
                ax2.set_ylabel('Total kinetic energy [J]')
                ax2.plot(t, Ekin, '+-')
                ax2.grid()

                # Rotational energy
                ax3 = plt.subplot2grid((2, 5), (0, 2))
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Total rotational energy [J]')
                ax3.plot(t, Erot, '+-')
                ax3.grid()

                # Bond energy
                ax4 = plt.subplot2grid((2, 5), (0, 3))
                ax4.set_xlabel('Time [s]')
                ax4.set_ylabel('Bond energy [J]')
                ax4.plot(t, Ebondpot, '+-')
                ax4.grid()

                # Total energy
                ax5 = plt.subplot2grid((2, 5), (0, 4))
                ax5.set_xlabel('Time [s]')
                ax5.set_ylabel('Total energy [J]')
                ax5.plot(t, Esum, '+-')
                ax5.grid()

                # Shear energy rate
                ax6 = plt.subplot2grid((2, 5), (1, 0))
                ax6.set_xlabel('Time [s]')
                ax6.set_ylabel('Frictional dissipation rate [W]')
                ax6.plot(t, Es_dot, '+-')
                ax6.grid()

                # Shear energy
                ax7 = plt.subplot2grid((2, 5), (1, 1))
                ax7.set_xlabel('Time [s]')
                ax7.set_ylabel('Total frictional dissipation [J]')
                ax7.plot(t, Es, '+-')
                ax7.grid()

                # Visc_n energy rate
                ax8 = plt.subplot2grid((2, 5), (1, 2))
                ax8.set_xlabel('Time [s]')
                ax8.set_ylabel('Viscous dissipation rate [W]')
                ax8.plot(t, Ev_dot, '+-')
                ax8.grid()

                # Visc_n energy
                ax9 = plt.subplot2grid((2, 5), (1, 3))
                ax9.set_xlabel('Time [s]')
                ax9.set_ylabel('Total viscous dissipation [J]')
                ax9.plot(t, Ev, '+-')
                ax9.grid()

                # Combined view
                ax10 = plt.subplot2grid((2, 5), (1, 4))
                ax10.set_xlabel('Time [s]')
                ax10.set_ylabel('Energy [J]')
                ax10.plot(t, Epot, '+-g')
                ax10.plot(t, Ekin, '+-b')
                ax10.plot(t, Erot, '+-r')
                ax10.legend(('$\sum E_{pot}$', '$\sum E_{kin}$',
                             '$\sum E_{rot}$'), 'upper right', shadow=True)
                ax10.grid()

                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)
                    ax3.set_xlim(xlim)
                    ax4.set_xlim(xlim)
                    ax5.set_xlim(xlim)
                    ax6.set_xlim(xlim)
                    ax7.set_xlim(xlim)
                    ax8.set_xlim(xlim)
                    ax9.set_xlim(xlim)
                    ax10.set_xlim(xlim)

                fig.tight_layout()

        elif method == 'walls':

            # Read energy values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                # Allocate arrays on first run
                if i == firststep:
                    wforce = numpy.zeros((lastfile+1)*sb.nw,\
                            dtype=numpy.float64).reshape((lastfile+1), sb.nw)
                    wvel = numpy.zeros((lastfile+1)*sb.nw,\
                            dtype=numpy.float64).reshape((lastfile+1), sb.nw)
                    wpos = numpy.zeros((lastfile+1)*sb.nw,\
                            dtype=numpy.float64).reshape((lastfile+1), sb.nw)
                    wsigma0 = numpy.zeros((lastfile+1)*sb.nw,\
                            dtype=numpy.float64).reshape((lastfile+1), sb.nw)
                    maxpos = numpy.zeros((lastfile+1), dtype=numpy.float64)
                    logstress = numpy.zeros((lastfile+1), dtype=numpy.float64)
                    voidratio = numpy.zeros((lastfile+1), dtype=numpy.float64)

                wforce[i] = sb.w_force[0]
                wvel[i] = sb.w_vel[0]
                wpos[i] = sb.w_x[0]
                wsigma0[i] = sb.w_sigma0[0]
                maxpos[i] = numpy.max(sb.x[:, 2]+sb.radius)
                logstress[i] = numpy.log((sb.w_force[0]/(sb.L[0]*sb.L[1]))/1000.0)
                voidratio[i] = sb.voidRatio()

            t = numpy.linspace(0.0, sb.time_current, lastfile+1)

            # Plotting
            if outformat != 'txt':
                # linear plot of time vs. wall position
                ax1 = plt.subplot2grid((2, 2), (0, 0))
                ax1.set_xlabel('Time [s]')
                ax1.set_ylabel('Position [m]')
                ax1.plot(t, wpos, '+-', label="upper wall")
                ax1.plot(t, maxpos, '+-', label="heighest particle")
                ax1.legend()
                ax1.grid()

                #ax2 = plt.subplot2grid((2, 2), (1, 0))
                #ax2.set_xlabel('Time [s]')
                #ax2.set_ylabel('Force [N]')
                #ax2.plot(t, wforce, '+-')

                # semilog plot of log stress vs. void ratio
                ax2 = plt.subplot2grid((2, 2), (1, 0))
                ax2.set_xlabel('log deviatoric stress [kPa]')
                ax2.set_ylabel('Void ratio [-]')
                ax2.plot(logstress, voidratio, '+-')
                ax2.grid()

                # linear plot of time vs. wall velocity
                ax3 = plt.subplot2grid((2, 2), (0, 1))
                ax3.set_xlabel('Time [s]')
                ax3.set_ylabel('Velocity [m/s]')
                ax3.plot(t, wvel, '+-')
                ax3.grid()

                # linear plot of time vs. deviatoric stress
                ax4 = plt.subplot2grid((2, 2), (1, 1))
                ax4.set_xlabel('Time [s]')
                ax4.set_ylabel('Deviatoric stress [Pa]')
                ax4.plot(t, wsigma0, '+-', label="$\sigma_0$")
                ax4.plot(t, wforce/(sb.L[0]*sb.L[1]), '+-', label="$\sigma'$")
                ax4.legend(loc=4)
                ax4.grid()

                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)
                    ax3.set_xlim(xlim)
                    ax4.set_xlim(xlim)

        elif method == 'triaxial':

            # Read energy values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                vol = (sb.w_x[0]-sb.origo[2]) * (sb.w_x[1]-sb.w_x[2]) \
                        * (sb.w_x[3] - sb.w_x[4])

                # Allocate arrays on first run
                if i == firststep:
                    axial_strain = numpy.zeros(lastfile+1, dtype=numpy.float64)
                    deviatoric_stress =\
                            numpy.zeros(lastfile+1, dtype=numpy.float64)
                    volumetric_strain =\
                            numpy.zeros(lastfile+1, dtype=numpy.float64)

                    w0pos0 = sb.w_x[0]
                    vol0 = vol

                sigma1 = sb.w_force[0]/\
                        ((sb.w_x[1]-sb.w_x[2])*(sb.w_x[3]-sb.w_x[4]))

                axial_strain[i] = (w0pos0 - sb.w_x[0])/w0pos0
                volumetric_strain[i] = (vol0-vol)/vol0
                deviatoric_stress[i] = sigma1 / sb.w_sigma0[1]

            #print(lastfile)
            #print(axial_strain)
            #print(deviatoric_stress)
            #print(volumetric_strain)

            # Plotting
            if outformat != 'txt':

                # linear plot of deviatoric stress
                ax1 = plt.subplot2grid((2, 1), (0, 0))
                ax1.set_xlabel('Axial strain, $\gamma_1$, [-]')
                ax1.set_ylabel('Deviatoric stress, $\sigma_1 - \sigma_3$, [Pa]')
                ax1.plot(axial_strain, deviatoric_stress, '+-')
                #ax1.legend()
                ax1.grid()

                #ax2 = plt.subplot2grid((2, 2), (1, 0))
                #ax2.set_xlabel('Time [s]')
                #ax2.set_ylabel('Force [N]')
                #ax2.plot(t, wforce, '+-')

                # semilog plot of log stress vs. void ratio
                ax2 = plt.subplot2grid((2, 1), (1, 0))
                ax2.set_xlabel('Axial strain, $\gamma_1$ [-]')
                ax2.set_ylabel('Volumetric strain, $\gamma_v$, [-]')
                ax2.plot(axial_strain, volumetric_strain, '+-')
                ax2.grid()

                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)

        elif method == 'shear':

            # Read stress values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                # First iteration: Allocate arrays and find constant values
                if i == firststep:
                    # Shear displacement
                    xdisp = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Normal stress
                    sigma_eff = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Normal stress
                    sigma_def = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Shear stress
                    tau = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Upper wall position
                    dilation = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Peak shear stress
                    tau_p = 0.0

                    # Shear strain value of peak sh. stress
                    tau_p_shearstrain = 0.0

                    fixvel = numpy.nonzero(sb.fixvel > 0.0)
                    #fixvel_upper = numpy.nonzero(sb.vel[fixvel, 0] > 0.0)
                    shearvel = sb.vel[fixvel, 0].max()
                    w_x0 = sb.w_x[0]        # Original height
                    A = sb.L[0] * sb.L[1]   # Upper surface area

                if i == firststep+1:
                    w_x0 = sb.w_x[0]        # Original height

                # Summation of shear stress contributions
                for j in fixvel[0]:
                    if sb.vel[j, 0] > 0.0:
                        tau[i] += -sb.force[j, 0]/A

                if i > 0:
                    xdisp[i] = xdisp[i-1] + sb.time_file_dt[0]*shearvel
                sigma_eff[i] = sb.w_force[0]/A
                sigma_def[i] = sb.w_sigma0[0]

                # dilation in meters
                #dilation[i] = sb.w_x[0] - w_x0

                # dilation in percent
                #dilation[i] = (sb.w_x[0] - w_x0)/w_x0 * 100.0 # dilation in percent

                # dilation in number of mean particle diameters
                d_bar = numpy.mean(self.radius)*2.0
                if numpy.isnan(d_bar):
                    print('No radii in self.radius, attempting to read first '
                          + 'file')
                    self.readfirst()
                    d_bar = numpy.mean(self.radius)*2.0
                dilation[i] = (sb.w_x[0] - w_x0)/d_bar

                # Test if this was the max. shear stress
                if tau[i] > tau_p:
                    tau_p = tau[i]
                    tau_p_shearstrain = xdisp[i]/w_x0

            shear_strain = xdisp/w_x0

            # Copy values so they can be modified during smoothing
            shear_strain_smooth = shear_strain
            tau_smooth = tau
            sigma_def_smooth = sigma_def

            # Optionally smooth the shear stress
            if smoothing > 2:

                if smoothing_window not in ['flat', 'hanning', 'hamming',
                                            'bartlett', 'blackman']:
                    raise ValueError

                s = numpy.r_[2*tau[0]-tau[smoothing:1:-1], tau,
                             2*tau[-1]-tau[-1:-smoothing:-1]]

                if smoothing_window == 'flat': # moving average
                    w = numpy.ones(smoothing, 'd')
                else:
                    w = getattr(self.np, smoothing_window)(smoothing)
                y = numpy.convolve(w/w.sum(), s, mode='same')
                tau_smooth = y[smoothing-1:-smoothing+1]

            # Plot stresses
            if outformat != 'txt':
                shearinfo = "$\\tau_p$={:.3} Pa at $\gamma$={:.3}".format(\
                        tau_p, tau_p_shearstrain)
                fig.text(0.01, 0.01, shearinfo, horizontalalignment='left',
                         fontproperties=FontProperties(size=14))
                ax1 = plt.subplot2grid((2, 1), (0, 0))
                ax1.set_xlabel('Shear strain [-]')
                ax1.set_ylabel('Shear friction $\\tau/\\sigma_0$ [-]')
                if smoothing > 2:
                    ax1.plot(shear_strain_smooth[1:-(smoothing+1)/2],
                             tau_smooth[1:-(smoothing+1)/2] /
                             sigma_def_smooth[1:-(smoothing+1)/2],
                             '-', label="$\\tau/\\sigma_0$")
                else:
                    ax1.plot(shear_strain[1:],\
                             tau[1:]/sigma_def[1:],\
                             '-', label="$\\tau/\\sigma_0$")
                ax1.grid()

                # Plot dilation
                ax2 = plt.subplot2grid((2, 1), (1, 0))
                ax2.set_xlabel('Shear strain [-]')
                ax2.set_ylabel('Dilation, $\Delta h/(2\\bar{r})$ [m]')
                if smoothing > 2:
                    ax2.plot(shear_strain_smooth[1:-(smoothing+1)/2],
                             dilation[1:-(smoothing+1)/2], '-')
                else:
                    ax2.plot(shear_strain, dilation, '-')
                ax2.grid()

                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)

                fig.tight_layout()

            else:
                # Write values to textfile
                filename = "shear-stresses-{0}.txt".format(self.sid)
                #print("Writing stress data to " + filename)
                fh = None
                try:
                    fh = open(filename, "w")
                    for i in numpy.arange(firststep, lastfile+1):
                        # format: shear distance [mm], sigma [kPa], tau [kPa],
                        # Dilation [%]
                        fh.write("{0}\t{1}\t{2}\t{3}\n"
                                 .format(xdisp[i], sigma_eff[i]/1000.0,
                                         tau[i]/1000.0, dilation[i]))
                finally:
                    if fh is not None:
                        fh.close()

        elif method == 'shear-displacement':

            time = numpy.zeros(lastfile+1, dtype=numpy.float64)
            # Read stress values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                # First iteration: Allocate arrays and find constant values
                if i == firststep:

                    # Shear displacement
                    xdisp = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Normal stress
                    sigma_eff = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Normal stress
                    sigma_def = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Shear stress
                    tau_eff = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Upper wall position
                    dilation = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Mean porosity
                    phi_bar = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Mean fluid pressure
                    p_f_bar = numpy.zeros(lastfile+1, dtype=numpy.float64)
                    p_f_top = numpy.zeros(lastfile+1, dtype=numpy.float64)

                    # Upper wall position
                    tau_p = 0.0             # Peak shear stress
                    # Shear strain value of peak sh. stress
                    tau_p_shearstrain = 0.0

                    fixvel = numpy.nonzero(sb.fixvel > 0.0)
                    #fixvel_upper=numpy.nonzero(sb.vel[fixvel, 0] > 0.0)
                    w_x0 = sb.w_x[0]      # Original height
                    A = sb.L[0]*sb.L[1]   # Upper surface area

                    d_bar = numpy.mean(sb.radius)*2.0

                    # Shear velocity
                    v = numpy.zeros(lastfile+1, dtype=numpy.float64)

                time[i] = sb.time_current[0]

                if i == firststep+1:
                    w_x0 = sb.w_x[0] # Original height

                # Summation of shear stress contributions
                for j in fixvel[0]:
                    if sb.vel[j, 0] > 0.0:
                        tau_eff[i] += -sb.force[j, 0]/A

                if i > 0:
                    xdisp[i] = sb.xyzsum[fixvel, 0].max()
                    v[i] = sb.vel[fixvel, 0].max()

                sigma_eff[i] = sb.w_force[0]/A
                sigma_def[i] = sb.currentNormalStress()

                # dilation in number of mean particle diameters
                dilation[i] = (sb.w_x[0] - w_x0)/d_bar

                wall0_iz = int(sb.w_x[0]/(sb.L[2]/sb.num[2]))

                if self.fluid:
                    if i > 0:
                        phi_bar[i] = numpy.mean(sb.phi[:, :, 0:wall0_iz])
                    if i == firststep+1:
                        phi_bar[0] = phi_bar[1]
                    p_f_bar[i] = numpy.mean(sb.p_f[:, :, 0:wall0_iz])
                    p_f_top[i] = sb.p_f[0, 0, -1]

                # Test if this was the max. shear stress
                if tau_eff[i] > tau_p:
                    tau_p = tau_eff[i]
                    tau_p_shearstrain = xdisp[i]/w_x0

            shear_strain = xdisp/w_x0

            # Plot stresses
            if outformat != 'txt':
                if figsize:
                    fig = plt.figure(figsize=figsize)
                else:
                    fig = plt.figure(figsize=(8, 12))

                # Upper plot
                ax1 = plt.subplot(3, 1, 1)
                ax1.plot(time, xdisp, 'k', label='Displacement')
                ax1.set_ylabel('Horizontal displacement [m]')

                ax2 = ax1.twinx()

                #ax2color = '#666666'
                ax2color = 'blue'
                if self.fluid:
                    ax2.plot(time, phi_bar, color=ax2color, label='Porosity')
                    ax2.set_ylabel('Mean porosity $\\bar{\\phi}$ [-]')
                else:
                    ax2.plot(time, dilation, color=ax2color, label='Dilation')
                    ax2.set_ylabel('Dilation, $\Delta h/(2\\bar{r})$ [-]')
                for tl in ax2.get_yticklabels():
                    tl.set_color(ax2color)

                # Middle plot
                ax5 = plt.subplot(3, 1, 2, sharex=ax1)
                ax5.semilogy(time[1:], v[1:], label='Shear velocity')
                ax5.set_ylabel('Shear velocity [ms$^{-1}$]')

                # shade stick periods
                collection = \
                        matplotlib.collections.BrokenBarHCollection.span_where(
                            time, ymin=1.0e-7, ymax=1.0,
                            where=numpy.isclose(v, 0.0),
                            facecolor='black', alpha=0.2,
                            linewidth=0)
                ax5.add_collection(collection)

                # Lower plot
                ax3 = plt.subplot(3, 1, 3, sharex=ax1)
                if sb.w_sigma0_A > 1.0e-3:
                    lns0 = ax3.plot(time, sigma_def/1000.0,
                                    '-k', label="$\\sigma_0$")
                    lns1 = ax3.plot(time, sigma_eff/1000.0,
                                    '--k', label="$\\sigma'$")
                    lns2 = ax3.plot(time, numpy.ones_like(time)*sb.w_tau_x/1000.0,
                                    '-r', label="$\\tau$")
                    lns3 = ax3.plot(time, tau_eff/1000.0,
                                    '--r', label="$\\tau'$")
                    ax3.set_ylabel('Stress [kPa]')
                else:
                    ax3.plot(time, tau_eff/sb.w_sigma0[0],
                             '-k', label="$Shear friction$")
                    ax3.plot([0, time[-1]],
                             [sb.w_tau_x/sigma_def, sb.w_tau_x/sigma_def],
                             '--k', label="$Applied shear friction$")
                    ax3.set_ylabel('Shear friction $\\tau\'/\\sigma_0$ [-]')
                    # axis limits
                    ax3.set_ylim([sb.w_tau_x/sigma_def[0]*0.5,
                                  sb.w_tau_x/sigma_def[0]*1.5])

                if self.fluid:
                    ax4 = ax3.twinx()
                    #ax4color = '#666666'
                    ax4color = ax2color
                    lns4 = ax4.plot(time, p_f_top/1000.0, '-', color=ax4color,
                                    label='$p_\\text{f}^\\text{forcing}$')
                    lns5 = ax4.plot(time, p_f_bar/1000.0, '--', color=ax4color,
                                    label='$\\bar{p}_\\text{f}$')
                    ax4.set_ylabel('Mean fluid pressure '
                                   + '$\\bar{p_\\text{f}}$ [kPa]')
                    for tl in ax4.get_yticklabels():
                        tl.set_color(ax4color)
                    if sb.w_sigma0_A > 1.0e-3:
                        #ax4.legend(loc='upper right')
                        lns = lns0+lns1+lns2+lns3+lns4+lns5
                        labs = [l.get_label() for l in lns]
                        ax4.legend(lns, labs, loc='upper right',
                                   fancybox=True, framealpha=legend_alpha)
                    if xlim:
                        ax4.set_xlim(xlim)

                # aesthetics
                ax3.set_xlabel('Time [s]')

                ax1.grid()
                ax3.grid()
                ax5.grid()

                if xlim:
                    ax1.set_xlim(xlim)
                    ax2.set_xlim(xlim)
                    ax3.set_xlim(xlim)
                    ax5.set_xlim(xlim)

                plt.setp(ax1.get_xticklabels(), visible=False)
                plt.setp(ax5.get_xticklabels(), visible=False)
                fig.tight_layout()
                plt.subplots_adjust(hspace=0.05)

        elif method == 'rate-dependence':

            if figsize:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure(figsize=(8, 6))

            tau = numpy.empty(sb.status())
            N = numpy.empty(sb.status())
            #v = numpy.empty(sb.status())
            shearstrainrate = numpy.empty(sb.status())
            shearstrain = numpy.empty(sb.status())
            for i in numpy.arange(firststep, sb.status()):
                sb.readstep(i+1, verbose=False)
                #tau = sb.shearStress()
                tau[i] = sb.w_tau_x # defined shear stress
                N[i] = sb.currentNormalStress() # defined normal stress
                shearstrainrate[i] = sb.shearStrainRate()
                shearstrain[i] = sb.shearStrain()

            # remove nonzero sliding velocities and their associated values
            idx = numpy.nonzero(shearstrainrate)
            shearstrainrate_nonzero = shearstrainrate[idx]
            tau_nonzero = tau[idx]
            N_nonzero = N[idx]
            shearstrain_nonzero = shearstrain[idx]

            ax1 = plt.subplot(111)
            #ax1.semilogy(N/1000., v)
            #ax1.semilogy(tau_nonzero/N_nonzero, v_nonzero, '+k')
            #ax1.plot(tau/N, v, '.')
            friction = tau_nonzero/N_nonzero
            #CS = ax1.scatter(friction, v_nonzero, c=shearstrain_nonzero,
                    #linewidth=0)
            if cmap:
                CS = ax1.scatter(friction, shearstrainrate_nonzero,
                                 c=shearstrain_nonzero, linewidth=0.1,
                                 cmap=cmap)
            else:
                CS = ax1.scatter(friction, shearstrainrate_nonzero,
                                 c=shearstrain_nonzero, linewidth=0.1,
                                 cmap=matplotlib.cm.get_cmap('afmhot'))
            ax1.set_yscale('log')
            x_min = numpy.floor(numpy.min(friction))
            x_max = numpy.ceil(numpy.max(friction))
            ax1.set_xlim([x_min, x_max])
            y_min = numpy.min(shearstrainrate_nonzero)*0.5
            y_max = numpy.max(shearstrainrate_nonzero)*2.0
            ax1.set_ylim([y_min, y_max])

            cb = plt.colorbar(CS)
            cb.set_label('Shear strain $\\gamma$ [-]')

            ax1.set_xlabel('Friction $\\tau/N$ [-]')
            ax1.set_ylabel('Shear strain rate $\\dot{\\gamma}$ [s$^{-1}$]')

        elif method == 'inertia':

            t = numpy.zeros(sb.status())
            I = numpy.zeros(sb.status())

            for i in numpy.arange(firststep, sb.status()):
                sb.readstep(i, verbose=False)
                t[i] = sb.currentTime()
                I[i] = sb.inertiaParameterPlanarShear()

            # Plotting
            if outformat != 'txt':

                if xlim:
                    ax1.set_xlim(xlim)

                # linear plot of deviatoric stress
                ax1 = plt.subplot2grid((1, 1), (0, 0))
                ax1.set_xlabel('Time $t$ [s]')
                ax1.set_ylabel('Inertia parameter $I$ [-]')
                ax1.semilogy(t, I)
                #ax1.legend()
                ax1.grid()

        elif method == 'mean-fluid-pressure':

            # Read pressure values from simulation binaries
            for i in numpy.arange(firststep, lastfile+1):
                sb.readstep(i, verbose=False)

                # Allocate arrays on first run
                if i == firststep:
                    p_mean = numpy.zeros(lastfile+1, dtype=numpy.float64)

                p_mean[i] = numpy.mean(sb.p_f)

            t = numpy.linspace(0.0, sb.time_current, lastfile+1)

            # Plotting
            if outformat != 'txt':

                if xlim:
                    ax1.set_xlim(xlim)

                # linear plot of deviatoric stress
                ax1 = plt.subplot2grid((1, 1), (0, 0))
                ax1.set_xlabel('Time $t$, [s]')
                ax1.set_ylabel('Mean fluid pressure, $\\bar{p}_f$, [kPa]')
                ax1.plot(t, p_mean/1000.0, '+-')
                #ax1.legend()
                ax1.grid()

        elif method == 'fluid-pressure':

            if figsize:
                fig = plt.figure(figsize=figsize)
            else:
                fig = plt.figure(figsize=(8, 6))

            sb.readfirst(verbose=False)

            # cell midpoint cell positions
            zpos_c = numpy.zeros(sb.num[2])
            dz = sb.L[2]/sb.num[2]
            for i in numpy.arange(sb.num[2]):
                zpos_c[i] = i*dz + 0.5*dz

            shear_strain = numpy.zeros(sb.status())
            pres = numpy.zeros((sb.num[2], sb.status()))

            # Read pressure values from simulation binaries
            for i in numpy.arange(firststep, sb.status()):
                sb.readstep(i, verbose=False)
                pres[:, i] = numpy.average(numpy.average(sb.p_f, axis=0), axis=0)
                shear_strain[i] = sb.shearStrain()
            t = numpy.linspace(0.0, sb.time_current, lastfile+1)

            # Plotting
            if outformat != 'txt':

                ax = plt.subplot(1, 1, 1)

                pres /= 1000.0 # Pa to kPa

                if xlim:
                    sb.readstep(10, verbose=False)
                    gamma_per_i = sb.shearStrain()/10.0
                    i_min = int(xlim[0]/gamma_per_i)
                    i_max = int(xlim[1]/gamma_per_i)
                    pres = pres[:, i_min:i_max]
                else:
                    i_min = 0
                    i_max = sb.status()
                # use largest difference in p from 0 as +/- limit on colormap
                #print i_min, i_max
                p_ext = numpy.max(numpy.abs(pres))

                if sb.wmode[0] == 3:
                    x = t
                else:
                    x = shear_strain
                if xlim:
                    x = x[i_min:i_max]
                if cmap:
                    im1 = ax.pcolormesh(x, zpos_c, pres, cmap=cmap,
                                        vmin=-p_ext, vmax=p_ext,
                                        rasterized=True)
                else:
                    im1 = ax.pcolormesh(x, zpos_c, pres,
                                        cmap=matplotlib.cm.get_cmap('RdBu_r'),
                                        vmin=-p_ext, vmax=p_ext,
                                        rasterized=True)
                ax.set_xlim([0, numpy.max(x)])
                if sb.w_x[0] < sb.L[2]:
                    ax.set_ylim([zpos_c[0], sb.w_x[0]])
                else:
                    ax.set_ylim([zpos_c[0], zpos_c[-1]])
                if sb.wmode[0] == 3:
                    ax.set_xlabel('Time $t$ [s]')
                else:
                    ax.set_xlabel('Shear strain $\\gamma$ [-]')
                ax.set_ylabel('Vertical position $z$ [m]')

                if xlim:
                    ax.set_xlim([x[0], x[-1]])

                # for article2
                ax.set_ylim([zpos_c[0], zpos_c[9]])

                cb = plt.colorbar(im1)
                cb.set_label('$p_\\text{f}$ [kPa]')
                cb.solids.set_rasterized(True)
                plt.tight_layout()

        elif method == 'porosity':

            sb.readfirst(verbose=False)
            if not sb.fluid:
                raise Exception('Porosities can only be visualized in wet ' +
                                'simulations')

            wall0_iz = int(sb.w_x[0]/(sb.L[2]/sb.num[2]))

            # cell midpoint cell positions
            zpos_c = numpy.zeros(sb.num[2])
            dz = sb.L[2]/sb.num[2]
            for i in numpy.arange(firststep, sb.num[2]):
                zpos_c[i] = i*dz + 0.5*dz

            shear_strain = numpy.zeros(sb.status())
            poros = numpy.zeros((sb.num[2], sb.status()))

            # Read pressure values from simulation binaries
            for i in numpy.arange(firststep, sb.status()):
                sb.readstep(i, verbose=False)
                poros[:, i] = numpy.average(numpy.average(sb.phi, axis=0), axis=0)
                shear_strain[i] = sb.shearStrain()
            t = numpy.linspace(0.0, sb.time_current, lastfile+1)

            # Plotting
            if outformat != 'txt':

                ax = plt.subplot(1, 1, 1)

                poros_max = numpy.max(poros[0:wall0_iz-1, 1:])
                poros_min = numpy.min(poros)

                if sb.wmode[0] == 3:
                    x = t
                else:
                    x = shear_strain
                if cmap:
                    im1 = ax.pcolormesh(x, zpos_c, poros,
                                        cmap=cmap,
                                        vmin=poros_min, vmax=poros_max,
                                        rasterized=True)
                else:
                    im1 = ax.pcolormesh(x, zpos_c, poros,
                                        cmap=matplotlib.cm.get_cmap('Blues_r'),
                                        vmin=poros_min, vmax=poros_max,
                                        rasterized=True)
                ax.set_xlim([0, numpy.max(x)])
                if sb.w_x[0] < sb.L[2]:
                    ax.set_ylim([zpos_c[0], sb.w_x[0]])
                else:
                    ax.set_ylim([zpos_c[0], zpos_c[-1]])
                if sb.wmode[0] == 3:
                    ax.set_xlabel('Time $t$ [s]')
                else:
                    ax.set_xlabel('Shear strain $\\gamma$ [-]')
                ax.set_ylabel('Vertical position $z$ [m]')

                if xlim:
                    ax.set_xlim(xlim)

                cb = plt.colorbar(im1)
                cb.set_label('Mean horizontal porosity $\\bar{\phi}$ [-]')
                cb.solids.set_rasterized(True)
                plt.tight_layout()
                plt.subplots_adjust(wspace=.05)

        elif method == 'contacts':

            for i in numpy.arange(sb.status()+1):
                fn = "../output/{0}.output{1:0=5}.bin".format(self.sid, i)
                sb.sid = self.sid + ".{:0=5}".format(i)
                sb.readbin(fn, verbose=True)
                if f_min and f_max:
                    sb.plotContacts(lower_limit=0.25, upper_limit=0.75,
                                    outfolder='../img_out/',
                                    f_min=f_min, f_max=f_max,
                                    title="t={:.2f} s, $N$={:.0f} kPa"
                                    .format(sb.currentTime(),
                                            sb.currentNormalStress('defined')
                                            /1000.))
                else:
                    sb.plotContacts(lower_limit=0.25, upper_limit=0.75,
                                    title="t={:.2f} s, $N$={:.0f} kPa"
                                    .format(sb.currentTime(),
                                            sb.currentNormalStress('defined')
                                            /1000.), outfolder='../img_out/')

            # render images to movie
            subprocess.call('cd ../img_out/ && ' +
                            'ffmpeg -sameq -i {}.%05d-contacts.png '
                            .format(self.sid) +
                            '{}-contacts.mp4'.format(self.sid),
                            shell=True)

        else:
            print("Visualization type '" + method + "' not understood")
            return

        # Optional save of figure content
        filename = ''
        if xlim:
            filename = '{0}-{1}-{3}.{2}'.format(self.sid, method, outformat,
                                                xlim[-1])
        else:
            filename = '{0}-{1}.{2}'.format(self.sid, method, outformat)
        if pickle:
            pl.dump(fig, file(filename + '.pickle', 'w'))

        # Optional save of figure
        if outformat != 'txt':
            if savefig:
                fig.savefig(filename)
                print(filename)
                fig.clf()
                plt.close()
            else:
                plt.show()


def convert(graphics_format='png', folder='../img_out', remove_ppm=False):
    '''
    Converts all PPM images in img_out to graphics_format using ImageMagick. All
    PPM images are subsequently removed if `remove_ppm` is `True`.

    :param graphics_format: Convert the images to this format
    :type graphics_format: str
    :param folder: The folder containing the PPM images to convert
    :type folder: str
    :param remove_ppm: Remove ALL ppm files in `folder` after conversion
    :type remove_ppm: bool
    '''

    #quiet = ' > /dev/null'
    quiet = ''
    # Convert images
    subprocess.call('for F in ' + folder \
            + '/*.ppm ; do BASE=`basename $F .ppm`; convert $F ' \
            + folder + '/$BASE.' + graphics_format + ' ' \
            + quiet + ' ; done', shell=True)

    # Remove PPM files
    if remove_ppm:
        subprocess.call('rm ' + folder + '/*.ppm', shell=True)

def render(binary, method='pres', max_val=1e3, lower_cutoff=0.0,
           graphics_format='png', verbose=True):
    '''
    Render target binary using the ``sphere`` raytracer.

    :param method: The color visualization method to use for the particles.
        Possible values are: 'normal': color all particles with the same
        color, 'pres': color by pressure, 'vel': color by translational
        velocity, 'angvel': color by rotational velocity, 'xdisp': color by
        total displacement along the x-axis, 'angpos': color by angular
        position.
    :type method: str
    :param max_val: The maximum value of the color bar
    :type max_val: float
    :param lower_cutoff: Do not render particles with a value below this
        value, of the field selected by ``method``
    :type lower_cutoff: float
    :param graphics_format: Convert the PPM images generated by the ray
        tracer to this image format using Imagemagick
    :type graphics_format: str
    :param verbose: Show verbose information during ray tracing
    :type verbose: bool
    '''
    quiet = ''
    if not verbose:
        quiet = '-q'

    # Render images using sphere raytracer
    if method == 'normal':
        subprocess.call('cd .. ; ./sphere ' + quiet + \
                ' --render ' + binary, shell=True)
    else:
        subprocess.call('cd .. ; ./sphere ' + quiet + \
                ' --method ' + method + ' {}'.format(max_val) + \
                ' -l {}'.format(lower_cutoff) + \
                ' --render ' + binary, shell=True)

    # Convert images to compressed format
    if verbose:
        print('converting to ' + graphics_format)
    convert(graphics_format)

def video(project, out_folder='./', video_format='mp4',
          graphics_folder='../img_out/', graphics_format='png', fps=25,
          verbose=True):
    '''
    Uses ffmpeg to combine images to animation. All images should be
    rendered beforehand using :func:`render()`.

    :param project: The simulation id of the project to render
    :type project: str
    :param out_folder: The output folder for the video file
    :type out_folder: str
    :param video_format: The format of the output video
    :type video_format: str
    :param graphics_folder: The folder containing the rendered images
    :type graphics_folder: str
    :param graphics_format: The format of the rendered images
    :type graphics_format: str
    :param fps: The number of frames per second to use in the video
    :type fps: int
    :param qscale: The output video quality, in ]0;1]
    :type qscale: float
    :param bitrate: The bitrate to use in the output video
    :type bitrate: int
    :param verbose: Show ffmpeg output
    :type verbose: bool
    '''
    # Possible loglevels:
    # quiet, panic, fatal, error, warning, info, verbose, debug
    loglevel = 'info'
    if not verbose:
        loglevel = 'error'

    outfile = out_folder + '/' + project + '.' + video_format
    subprocess.call('ffmpeg -loglevel ' + loglevel + ' '
                    + '-i ' + graphics_folder + project + '.output%05d.'
                    + graphics_format
                    + ' -c:v libx264 -profile:v high -pix_fmt yuv420p -g 30'
                    + ' -r {} -y '.format(fps)
                    + outfile, shell=True)
    if verbose:
        print('saved to ' + outfile)

def thinsectionVideo(project, out_folder="./", video_format="mp4", fps=25,
                     qscale=1, bitrate=1800, verbose=False):
    '''
    Uses ffmpeg to combine thin section images to an animation. This function
    will implicity render the thin section images beforehand.

    :param project: The simulation id of the project to render
    :type project: str
    :param out_folder: The output folder for the video file
    :type out_folder: str
    :param video_format: The format of the output video
    :type video_format: str
    :param fps: The number of frames per second to use in the video
    :type fps: int
    :param qscale: The output video quality, in ]0;1]
    :type qscale: float
    :param bitrate: The bitrate to use in the output video
    :type bitrate: int
    :param verbose: Show ffmpeg output
    :type verbose: bool
    '''
    ''' Use ffmpeg to combine thin section images to animation.
        This function will start off by rendering the images.
    '''

    # Render thin section images (png)
    lastfile = status(project)
    sb = sim(fluid=False)
    for i in range(lastfile+1):
        fn = "../output/{0}.output{1:0=5}.bin".format(project, i)
        sb.sid = project + ".output{:0=5}".format(i)
        sb.readbin(fn, verbose=False)
        sb.thinsection_x1x3(cbmax=sb.w_sigma0[0]*4.0)

    # Combine images to animation
    # Possible loglevels:
    # quiet, panic, fatal, error, warning, info, verbose, debug
    loglevel = "info"
    if not verbose:
        loglevel = "error"

    subprocess.call("ffmpeg -qscale {0} -r {1} -b {2} -y ".format(\
                    qscale, fps, bitrate)
                    + "-loglevel " + loglevel + " "
                    + "-i ../img_out/" + project + ".output%05d-ts-x1x3.png "
                    + "-vf 'crop=((in_w/2)*2):((in_h/2)*2)' " \
                    + out_folder + "/" + project + "-ts-x1x3." + video_format,
                    shell=True)

def run(binary, verbose=True, hideinputfile=False):
    '''
    Execute ``sphere`` with target binary file as input.

    :param binary: Input file for ``sphere``
    :type binary: str
    :param verbose: Show ``sphere`` output
    :type verbose: bool
    :param hideinputfile: Hide the input file
    :type hideinputfile: bool
    '''

    quiet = ''
    stdout = ''
    if not verbose:
        quiet = '-q'
    if hideinputfile:
        stdout = ' > /dev/null'
    subprocess.call('cd ..; ./sphere ' + quiet + ' ' + binary + ' ' + stdout, \
            shell=True)

def torqueScriptParallel3(obj1, obj2, obj3, email='adc@geo.au.dk',
                          email_alerts='ae', walltime='24:00:00',
                          queue='qfermi', cudapath='/com/cuda/4.0.17/cuda',
                          spheredir='/home/adc/code/sphere',
                          use_workdir=False,
                          workdir='/scratch'):
    '''
    Create job script for the Torque queue manager for three binaries,
    executed in parallel, ideally on three GPUs.

    :param email: The e-mail address that Torque messages should be sent to
    :type email: str
    :param email_alerts: The type of Torque messages to send to the e-mail
        address. The character 'b' causes a mail to be sent when the
        execution begins. The character 'e' causes a mail to be sent when
        the execution ends normally. The character 'a' causes a mail to be
        sent if the execution ends abnormally. The characters can be written
        in any order.
    :type email_alerts: str
    :param walltime: The maximal allowed time for the job, in the format
        'HH:MM:SS'.
    :type walltime: str
    :param queue: The Torque queue to schedule the job for
    :type queue: str
    :param cudapath: The path of the CUDA library on the cluster compute nodes
    :type cudapath: str
    :param spheredir: The path to the root directory of sphere on the cluster
    :type spheredir: str
    :param use_workdir: Use a different working directory than the sphere folder
    :type use_workdir: bool
    :param workdir: The working directory during the calculations, if
        `use_workdir=True`
    :type workdir: str

    :returns: The filename of the script
    :return type: str

    See also :func:`torqueScript()`
    '''

    filename = obj1.sid + '_' + obj2.sid + '_' + obj3.sid + '.sh'

    fh = None
    try:
        fh = open(filename, "w")

        fh.write('#!/bin/sh\n')
        fh.write('#PBS -N ' + obj1.sid + '_' + obj2.sid + '_' + obj3.sid + '\n')
        fh.write('#PBS -l nodes=1:ppn=1\n')
        fh.write('#PBS -l walltime=' + walltime + '\n')
        fh.write('#PBS -q ' + queue + '\n')
        fh.write('#PBS -M ' + email + '\n')
        fh.write('#PBS -m ' + email_alerts + '\n')
        fh.write('CUDAPATH=' + cudapath + '\n')
        fh.write('export PATH=$CUDAPATH/bin:$PATH\n')
        fh.write('export LD_LIBRARY_PATH=$CUDAPATH/lib64')
        fh.write(':$CUDAPATH/lib:$LD_LIBRARY_PATH\n')
        fh.write('echo "`whoami`@`hostname`"\n')
        fh.write('echo "Start at `date`"\n')
        if use_workdir:
            fh.write('ORIGDIR=' + spheredir + '\n')
            fh.write('WORKDIR=' + workdir + "/$PBS_JOBID\n")
            fh.write('cp -r $ORIGDIR/* $WORKDIR\n')
            fh.write('cd $WORKDIR\n')
        else:
            fh.write('cd ' + spheredir + '\n')
        fh.write('cmake . && make\n')
        fh.write('./sphere input/' + obj1.sid + '.bin > /dev/null &\n')
        fh.write('./sphere input/' + obj2.sid + '.bin > /dev/null &\n')
        fh.write('./sphere input/' + obj3.sid + '.bin > /dev/null &\n')
        fh.write('wait\n')
        if use_workdir:
            fh.write('cp $WORKDIR/output/* $ORIGDIR/output/\n')
        fh.write('echo "End at `date`"\n')
        return filename

    finally:
        if fh is not None:
            fh.close()

def status(project):
    '''
    Check the status.dat file for the target project, and return the last output
    file number.

    :param project: The simulation id of the target project
    :type project: str

    :returns: The last output file written in the simulation calculations
    :return type: int
    '''

    fh = None
    try:
        filepath = "../output/{0}.status.dat".format(project)
        fh = open(filepath)
        data = fh.read()
        return int(data.split()[2])  # Return last file number
    finally:
        if fh is not None:
            fh.close()

def cleanup(sb):
    '''
    Removes the input/output files and images belonging to the object simulation
    ID from the ``input/``, ``output/`` and ``img_out/`` folders.

    :param sb: A sphere.sim object
    :type sb: sim
    '''
    subprocess.call("rm -f ../input/" + sb.sid + ".bin", shell=True)
    subprocess.call("rm -f ../output/" + sb.sid + ".*.bin", shell=True)
    subprocess.call("rm -f ../img_out/" + sb.sid + ".*", shell=True)
    subprocess.call("rm -f ../output/" + sb.sid + ".status.dat", shell=True)
    subprocess.call("rm -f ../output/" + sb.sid + ".*.vtu", shell=True)
    subprocess.call("rm -f ../output/fluid-" + sb.sid + ".*.vti", shell=True)
    subprocess.call("rm -f ../output/" + sb.sid + "-conv.png", shell=True)
    subprocess.call("rm -f ../output/" + sb.sid + "-conv.log", shell=True)

def V_sphere(r):
    '''
    Calculates the volume of a sphere with radius r

    :returns: The sphere volume [m^3]
    :return type: float
    '''
    return 4.0/3.0 * math.pi * r**3.0
