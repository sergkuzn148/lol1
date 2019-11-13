#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"
#include "utility.h"

// 1: Enable color output in array printing functions, 0: Disable
const int color_output = 0;

// Initialize memory
void DEM::initNSmem()
{
    // Number of cells
    ns.nx = grid.num[0];
    ns.ny = grid.num[1];
    ns.nz = grid.num[2];
    unsigned int ncells = NScells();
    unsigned int ncells_st = NScellsVelocity();

    ns.p     = new Float[ncells];     // hydraulic pressure
    ns.v     = new Float3[ncells];    // hydraulic velocity
    ns.v_x   = new Float[ncells_st];  // hydraulic velocity in staggered grid
    ns.v_y   = new Float[ncells_st];  // hydraulic velocity in staggered grid
    ns.v_z   = new Float[ncells_st];  // hydraulic velocity in staggered grid
    //ns.v_p   = new Float3[ncells];    // predicted hydraulic velocity
    //ns.v_p_x = new Float[ncells_st];  // pred. hydraulic velocity in st. grid
    //ns.v_p_y = new Float[ncells_st];  // pred. hydraulic velocity in st. grid
    //ns.v_p_z = new Float[ncells_st];  // pred. hydraulic velocity in st. grid
    ns.phi   = new Float[ncells];     // porosity
    ns.dphi  = new Float[ncells];     // porosity change
    ns.norm  = new Float[ncells];     // normalized residual of epsilon
    ns.epsilon = new Float[ncells];   // normalized residual of epsilon
    ns.epsilon_new = new Float[ncells]; // normalized residual of epsilon
    ns.f_d = new Float4[np]; // drag force on particles
    ns.f_p = new Float4[np]; // pressure force on particles
    ns.f_v = new Float4[np]; // viscous force on particles
    ns.f_sum = new Float4[np]; // sum of fluid forces on particles
    ns.p_constant = new int[ncells];  // unused
}

unsigned int DEM::NScells()
{
    //return ns.nx*ns.ny*ns.nz; // without ghost nodes
    return (ns.nx+2)*(ns.ny+2)*(ns.nz+2); // with ghost nodes
}

// Returns the number of velocity nodes in a congruent padded grid. There are
// velocity nodes between the boundary points and the pressure ghost nodes, but
// not on the outer side of the ghost nodes
unsigned int DEM::NScellsVelocity()
{
    // Congruent padding for velocity grids. See "Cohen and Molemaker 'A fast
    // double precision CFD code using CUDA'" for details
    //return (ns.nx+1)*(ns.ny+1)*(ns.nz+1); // without ghost nodes
    return (ns.nx+3)*(ns.ny+3)*(ns.nz+3); // with ghost nodes
}

// Free memory
void DEM::freeNSmem()
{
    delete[] ns.p;
    delete[] ns.v;
    delete[] ns.v_x;
    delete[] ns.v_y;
    delete[] ns.v_z;
    //delete[] ns.v_p;
    //delete[] ns.v_p_x;
    //delete[] ns.v_p_y;
    //delete[] ns.v_p_z;
    delete[] ns.phi;
    delete[] ns.dphi;
    delete[] ns.norm;
    delete[] ns.epsilon;
    delete[] ns.epsilon_new;
    delete[] ns.f_d;
    delete[] ns.f_p;
    delete[] ns.f_v;
    delete[] ns.f_sum;
    delete[] ns.p_constant;
}

// 3D index to 1D index
unsigned int DEM::idx(
        const int x,
        const int y,
        const int z)
{
    // without ghost nodes
    //return x + d.nx*y + d.nx*d.ny*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (ns.nx+2)*(y+1) + (ns.nx+2)*(ns.ny+2)*(z+1);
}

// 3D index to 1D index of cell-face velocity nodes. The cell-face velocities
// are placed at x = [0;nx], y = [0;ny], z = [0;nz].
// The coordinate x,y,z corresponds to the lowest corner of cell(x,y,z).
unsigned int DEM::vidx(
        const int x,
        const int y,
        const int z)
{
    //return x + (ns.nx+1)*y + (ns.nx+1)*(ns.ny+1)*z; // without ghost nodes
    return (x+1) + (ns.nx+3)*(y+1) + (ns.nx+3)*(ns.ny+3)*(z+1); // with ghost nodes
}

// Determine if the FTCS (forward time, central space) solution of the Navier
// Stokes equations is unstable
void DEM::checkNSstability()
{
    // Cell dimensions
    const Float dx = grid.L[0]/grid.num[0];
    const Float dy = grid.L[1]/grid.num[1];
    const Float dz = grid.L[2]/grid.num[2];

    // The smallest grid spacing
    const Float dmin = fmin(dx, fmin(dy, dz));

    // Check the diffusion term using von Neumann stability analysis
    if (ns.mu*time.dt/(dmin*dmin) > 0.5) {
        std::cerr << "Error: The time step is too large to ensure stability in "
            "the diffusive term of the fluid momentum equation.\n"
            "Decrease the viscosity, decrease the time step, and/or increase "
            "the fluid grid cell size." << std::endl;
        exit(1);
    }

    int x,y,z;
    Float3 v;
    for (x=0; x<ns.nx; ++x) {
        for (y=0; y<ns.ny; ++y) {
            for (z=0; z<ns.nz; ++z) {

                v = ns.v[idx(x,y,z)];

                // Check the advection term using the Courant-Friedrichs-Lewy
                // condition
                if (v.x*time.dt/dx + v.y*time.dt/dy + v.z*time.dt/dz > 1.0) {
                    std::cerr << "Error: The time step is too large to ensure "
                        "stability in the advective term of the fluid momentum "
                        "equation.\n"
                        "This is caused by too high fluid velocities. "
                        "You can try to decrease the time step, and/or "
                        "increase the fluid grid cell size.\n"
                        "v(" << x << ',' << y << ',' << z << ") = ["
                        << v.x << ',' << v.y << ',' << v.z << "] m/s"
                        << std::endl;
                    exit(1);
                }
            }
        }
    }



}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printNSarray(FILE* stream, Float* arr)
{
    int x, y, z;

    // show ghost nodes
    //for (z=-1; z<=ns.nz; z++) { // bottom to top
    //for (z = ns.nz-1; z >= -1; z--) { // top to bottom
    for (z = ns.nz; z >= -1; z--) { // top to bottom

        fprintf(stream, "z = %d\n", z);

        for (y=-1; y<=ns.ny; y++) {
            for (x=-1; x<=ns.nx; x++) {

    // hide ghost nodes
    /*for (z=0; z<ns.nz; z++) {
        for (y=0; y<ns.ny; y++) {
            for (x=0; x<ns.nx; x++) {*/

                if (x > -1 && x < ns.nx &&
                        y > -1 && y < ns.ny &&
                        z > -1 && z < ns.nz) {
                    fprintf(stream, "%f\t", arr[idx(x,y,z)]);
                } else { // ghost node
                    if (color_output) {
                        fprintf(stream, "\x1b[30;1m%f\x1b[0m\t",
                                arr[idx(x,y,z)]);
                    } else {
                        fprintf(stream, "%f\t", arr[idx(x,y,z)]);
                    }
                }
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printNSarray to add optional description
void DEM::printNSarray(FILE* stream, Float* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printNSarray(stream, arr);
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printNSarray(FILE* stream, Float3* arr)
{
    int x, y, z;
    for (z=0; z<ns.nz; z++) {
        for (y=0; y<ns.ny; y++) {
            for (x=0; x<ns.nx; x++) {
                fprintf(stream, "%f,%f,%f\t",
                        arr[idx(x,y,z)].x,
                        arr[idx(x,y,z)].y,
                        arr[idx(x,y,z)].z);
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printNSarray to add optional description
void DEM::printNSarray(FILE* stream, Float3* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printNSarray(stream, arr);
}

// Returns the mean particle radius
Float DEM::meanRadius()
{
    unsigned int i;
    Float r_sum;
    for (i=0; i<np; ++i)
        r_sum += k.x[i].w;
    return r_sum/((Float)np);
}

// Returns the average value of the normalized residuals
double DEM::avgNormResNS()
{
    double norm_res_sum, norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = static_cast<double>(ns.norm[idx(x,y,z)]);
                if (norm_res != norm_res) {
                    std::cerr << "\nError: normalized residual is NaN ("
                        << norm_res << ") in cell "
                        << x << "," << y << "," << z << std::endl;
                    std::cerr << "\tt = " << time.current << ", iter = "
                        << int(time.current/time.dt) << std::endl;
                    std::cerr << "This often happens if the system has become "
                        "unstable." << std::endl;
                    exit(1);
                }
                norm_res_sum += norm_res;
            }
        }
    }
    return norm_res_sum/(grid.num[0]*grid.num[1]*grid.num[2]);
}


// Returns the average value of the normalized residuals
double DEM::maxNormResNS()
{
    double max_norm_res = -1.0e9; // initialized to a small number
    double norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = fabs(static_cast<double>(ns.norm[idx(x,y,z)]));
                if (norm_res != norm_res) {
                    std::cerr << "\nError: normalized residual is NaN ("
                        << norm_res << ") in cell "
                        << x << "," << y << "," << z << std::endl;
                    std::cerr << "\tt = " << time.current << ", iter = "
                        << int(time.current/time.dt) << std::endl;
                    std::cerr << "This often happens if the system has become "
                        "unstable." << std::endl;
                    exit(1);
                }
                if (norm_res > max_norm_res)
                    max_norm_res = norm_res;
            }
        }
    }
    return max_norm_res;
}

// Initialize fluid parameters
void DEM::initNS()
{
    // Cell size 
    ns.dx = grid.L[0]/ns.nx;
    ns.dy = grid.L[1]/ns.ny;
    ns.dz = grid.L[2]/ns.nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << ns.nx << "*"
            << ns.ny << "*"
            << ns.nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << ns.dx << "*"
            << ns.dy << "*"
            << ns.dz << std::endl;
    }
}

// Write values in scalar field to file
void DEM::writeNSarray(Float* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printNSarray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}

// Write values in vector field to file
void DEM::writeNSarray(Float3* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printNSarray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}


// Print final heads and free memory
void DEM::endNS()
{
    // Write arrays to stdout/text files for debugging
    //writeNSarray(ns.phi, "ns_phi.txt");

    //printNSarray(stdout, ns.K, "ns.K");
    //printNSarray(stdout, ns.H, "ns.H");
    //printNSarray(stdout, ns.H_new, "ns.H_new");
    //printNSarray(stdout, ns.V, "ns.V");

    freeNSmem();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
