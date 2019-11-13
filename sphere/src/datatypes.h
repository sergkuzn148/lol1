// Avoiding multiple inclusions of header file
#ifndef DATATYPES_H_
#define DATATYPES_H_

#include <math.h>
#include "vector_functions.h"
#include "typedefs.h"
#include "constants.h"


////////////////////////////
// STRUCTURE DECLARATIONS //
////////////////////////////

// Structure containing kinematic particle values
struct Kinematics {
    Float4 *x;              // Positions + radii (w)
    Float4 *xyzsum;         // Horizontal distance traveled
    Float4 *vel;            // Translational velocities + fixvels (w)
    Float4 *acc;            // Translational accelerations
    Float4 *force;          // Sums of forces
    Float4 *angpos;         // Angular positions
    Float4 *angvel;         // Angular velocities
    Float4 *angacc;         // Angular accelerations
    Float4 *torque;         // Sums of torques
    unsigned int *contacts; // List of contacts per particle
    Float4 *distmod;        // Distance modifiers across periodic boundaries
    Float4 *delta_t;        // Accumulated shear distance of contacts
    uint2  *bonds;          // Particle bond pairs
    Float4 *bonds_delta;    // Particle bond displacement
    Float4 *bonds_omega;    // Particle bond rotation
    int    *color;          // Color index for visualization
};

// Structure containing individual particle energies
struct Energies {
    Float *es_dot; // Frictional dissipation rates
    Float *es;     // Frictional dissipations
    Float *ev_dot; // Viscous dissipation rates
    Float *ev;     // Viscous dissipations
    Float *p;      // Pressures
};

// Structure containing grid parameters
struct Grid {
    Float origo[ND];        // World coordinate system origo
    Float L[ND];            // World dimensions
    unsigned int num[ND];   // Neighbor-search cells along each axis
    int periodic;           // Behavior of boundaries at 1st and 2nd world edge
    int adaptive;           // Continuously rescale grid size to wall positions
};

struct Sorting {
    Float4 *x_sorted;                 // Positions + radii (w) (sorted)
    Float4 *vel_sorted;               // Velocities + fixvels (w) (sorted)
    Float4 *angvel_sorted;            // Angular velocities (sorted)
    unsigned int *gridParticleCellID; // Hash key (cell idx) in grid
    unsigned int *gridParticleIndex;  // Original indexes of particles
    unsigned int *cellStart;          // First index of sorted idx'es in cells
    unsigned int *cellEnd;            // Last index of sorted idx'es in cells
};


// Structure containing time parameters
struct Time {
    Float dt;                // Computational time step length
    double current;          // Current time
    double total;            // Total time (at the end of experiment)
    Float file_dt;           // Time between output files
    unsigned int step_count; // Number of output files written
};

// Structure containing constant, global physical parameters
struct Params {
    Float g[ND];          // Gravitational acceleration
    Float k_n;            // Normal stiffness
    Float k_t;            // Tangential stiffness
    Float k_r;            // Rotational stiffness
    Float E;              // Young's modulus
    Float gamma_n;        // Normal viscosity
    Float gamma_t;        // Tangential viscosity
    Float gamma_r;        // Rotational viscosity
    Float mu_s;           // Static friction coefficient
    Float mu_d;           // Dynamic friction coefficient
    Float mu_r;           // Rotational friction coefficient
    Float gamma_wn;       // Wall normal viscosity
    Float gamma_wt;       // Wall tangential viscosity
    Float mu_ws;          // Wall static friction coefficient
    Float mu_wd;          // Wall dynamic friction coefficient
    Float rho;            // Material density
    unsigned int contactmodel; // Inter-particle contact model
    Float kappa;          // Capillary bond prefactor
    Float db;             // Capillary bond debonding distance
    Float V_b;            // Volume of fluid in capillary bond
    Float lambda_bar;     // Radius multiplier to parallel-bond radii
    unsigned int nb0;     // Number of inter-particle bonds at t=0
    Float sigma_b;        // Bond tensile strength
    Float tau_b;          // Bond shear strength
    Float sigma0_A;       // Amplitude of modulations in normal stress
    Float sigma0_f;       // Frequency of modulations in normal stress
};

// Structure containing wall parameters
struct Walls {
    unsigned int nw;     // Number of walls (<= MAXWALLS)
    int wmode[MAXWALLS]; // Wall modes
    Float4* nx;          // Wall normal and position
    Float4* mvfd;        // Wall mass, velocity, force and normal stress
    Float*  tau_x;       // Wall shear stress
};

// Structures containing fluid parameters
struct NavierStokes {
    int     nx, ny, nz;  // Number of cells in each dim
    Float   dx, dy, dz;  // Cell length in each dim
    Float*  p;           // Cell hydraulic pressures
    Float3* v;           // Cell fluid velocity
    Float*  v_x;         // Fluid velocity in staggered grid
    Float*  v_y;         // Fluid velocity in staggered grid
    Float*  v_z;         // Fluid velocity in staggered grid
    //Float3* v_p;         // Predicted fluid velocity
    //Float*  v_p_x;       // Predicted fluid velocity in staggered grid
    //Float*  v_p_y;       // Predicted fluid velocity in staggered grid
    //Float*  v_p_z;       // Predicted fluid velocity in staggered grid
    Float*  phi;            // Cell porosity
    Float*  dphi;           // Cell porosity change
    Float*  norm;           // Normalized residual of epsilon updates
    Float*  epsilon;        // Iterative solution parameter
    Float*  epsilon_new;    // Updated value of iterative solution parameter
    Float   p_mod_A;        // Pressure modulation amplitude at top
    Float   p_mod_f;        // Pressure modulation frequency at top
    Float   p_mod_phi;      // Pressure modulation phase at top
    int     bc_bot;         // 0: Dirichlet, 1: Neumann, 3: Periodic, 4: Flux
    int     bc_top;         // 0: Dirichlet, 1: Neumann, 3: Periodic, 4: Flux
    int     free_slip_bot;  // 0: no, 1: yes
    int     free_slip_top;  // 0: no, 1: yes
    Float   bc_bot_flux;    // Flux normal to boundary
    Float   bc_top_flux;    // Flux normal to boundary
    int*    p_constant;     // Keep pressure in cell constant (0: False, 1:True)
    Float   gamma;          // Solver parameter: Smoothing
    Float   theta;          // Solver parameter: Under-relaxation
    Float   beta;           // Solver parameter: Solution method
    Float   tolerance;      // Solver parameter: Max residual tolerance
    unsigned int maxiter;   // Solver parameter: Max iterations to perform
    unsigned int ndem;      // Solver parameter: DEM time steps per CFD step
    Float   c_phi;          // Porosity scaling coefficient
    Float   c_v;            // Fluid velocity scaling coefficient
    Float   dt_dem_fac;     // DEM-CFD time scaling coefficient
    Float4* f_d;            // Drag force on particles
    Float4* f_p;            // Pressure force on particles
    Float4* f_v;            // Viscous force on particles
    Float4* f_sum;          // Viscous force on particles
    Float   mu;             // Fluid dynamic viscosity
    Float   rho_f;          // Fluid density
};

struct Darcy {
    int     nx, ny, nz;     // Number of cells in each dimension
    Float   dx, dy, dz;     // Cell length in each dim
    Float*  p;              // Cell hydraulic pressures
    Float3* v;              // Cell fluid velocity
    Float*  k;              // Cell hydraulic permeability
    Float*  phi;            // Cell porosity
    Float*  dphi;           // Cell porosity change
    Float*  norm;           // Normalized residual of epsilon updates
    Float   p_top_orig;     // Pressure at top boundary at t=0
    Float   p_mod_A;        // Pressure modulation amplitude at top
    Float   p_mod_f;        // Pressure modulation frequency at top
    Float   p_mod_phi;      // Pressure modulation phase at top
    int     bc_xn;          // 0: Dirichlet, 1: Neumann, 3: Periodic
    int     bc_xp;          // 0: Dirichlet, 1: Neumann, 3: Periodic
    int     bc_yn;          // 0: Dirichlet, 1: Neumann, 3: Periodic
    int     bc_yp;          // 0: Dirichlet, 1: Neumann, 3: Periodic
    int     bc_bot;         // 0: Dirichlet, 1: Neumann
    int     bc_top;         // 0: Dirichlet, 1: Neumann
    int     free_slip_bot;  // 0: no, 1: yes
    int     free_slip_top;  // 0: no, 1: yes
    Float   bc_bot_flux;    // Flux normal to boundary
    Float   bc_top_flux;    // Flux normal to boundary
    int*    p_constant;     // Keep pressure in cell constant (0: False, 1:True)
    Float   tolerance;      // Solver parameter: Max residual tolerance
    unsigned int maxiter;   // Solver parameter: Max iterations to perform
    unsigned int ndem;      // Solver parameter: DEM time steps per CFD step
    Float   c_phi;          // Porosity scaling coefficient
    Float4* f_p;            // Pressure force on particles
    Float   beta_f;         // Adiabatic fluid compressibility
    Float   k_c;            // Permeability prefactor in Kozeny-Carman eq.
    Float   mu;             // Fluid dynamic viscosity
    Float   rho_f;          // Fluid density
};

// Image structure
struct rgba {
    unsigned char r;  // Red
    unsigned char g;  // Green
    unsigned char b;  // Blue
    unsigned char a;  // Alpha
};

#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
