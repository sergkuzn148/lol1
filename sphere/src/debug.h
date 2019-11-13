#ifndef DEBUG_H_
#define DEBUG_H_

// Enable profiling of kernel runtimes?
// 0: No (default)
// 1: Yes
#define PROFILING 1

// Output information about contacts to stdout?
// 0: No (default)
// 1: Yes
#define CONTACTINFO 0

// The number of fluid solver iterations to perform between checking the norm.
// residual value
const unsigned int nijacnorm = 10;

// Write max. residual during the latest solution loop to logfile
// 'max_res_norm.dat'
// 0: False, 1: True
const int write_res_log = 0;

// Report pressure (epsilon) values during Jacobi iterations to stdout
//#define REPORT_EPSILON
//#define REPORT_MORE_EPSILON

// Report the number of iterations it took before convergence to logfile
// 'output/<sid>-conv.dat'
// 0: False, 1: True
const int write_conv_log = 1;

// The interval between iteration number reporting in 'output/<sid>-conv.log'
const int conv_log_interval = 10;
//const int conv_log_interval = 4;
//const int conv_log_interval = 1;

// Enable drag force and particle fluid coupling
#define CFD_DEM_COUPLING

// Check if initial particle positions are finite values
#define CHECK_PARTICLES_FINITE

// Check for nan/inf values in fluid solver kernels
#define CHECK_FLUID_FINITE

// Enable reporting of velocity prediction components to stdout
//#define REPORT_V_P_COMPONENTS

// Enable reporting of velocity correction components to stdout
//#define REPORT_V_C_COMPONENTS

// Enable reporting of initial values of forcing function terms to stdout
//#define REPORT_FORCING_TERMS

// Enable reporting of forcing finction terms during Jacobian iterations to 
// stdout
//#define REPORT_FORCING_TERMS_JACOBIAN

// Choose solver model (see Zhou et al. 2010 "Discrete particle simulation of
// particle-fluid flow: model formulations and their applicability", table. 1.
// SET_1 corresponds exactly to Model B in Zhu et al. 2007 "Discrete particle
// simulation of particulate systems: Theoretical developments".
// SET_2 corresponds approximately to Model A in Zhu et al. 2007.
// Choose exactly one.
//#define SET_1
#define SET_2

#endif
