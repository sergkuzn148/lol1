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
void DEM::initDarcyMem()
{
    // Number of cells
    darcy.nx = grid.num[0];
    darcy.ny = grid.num[1];
    darcy.nz = grid.num[2];
    unsigned int ncells = darcyCells();
    //unsigned int ncells_st = darcyCellsVelocity();

    darcy.p     = new Float[ncells];     // hydraulic pressure
    darcy.v     = new Float3[ncells];    // hydraulic velocity
    darcy.phi   = new Float[ncells];     // porosity
    darcy.dphi  = new Float[ncells];     // porosity change
    darcy.norm  = new Float[ncells];     // normalized residual of epsilon
    darcy.f_p   = new Float4[np];        // pressure force on particles
    darcy.k     = new Float[ncells];     // hydraulic pressure
    darcy.p_constant = new int[ncells];  // constant pressure (0: no, 1: yes)
}

unsigned int DEM::darcyCells()
{
    //return darcy.nx*darcy.ny*darcy.nz; // without ghost nodes
    return (darcy.nx+2)*(darcy.ny+2)*(darcy.nz+2); // with ghost nodes
}

// Returns the number of velocity nodes in a congruent padded grid. There are
// velocity nodes between the boundary points and the pressure ghost nodes, but
// not on the outer side of the ghost nodes
unsigned int DEM::darcyCellsVelocity()
{
    // Congruent padding for velocity grids. See "Cohen and Molemaker 'A fast
    // double precision CFD code using CUDA'" for details
    //return (darcy.nx+1)*(darcy.ny+1)*(darcy.nz+1); // without ghost nodes
    return (darcy.nx+3)*(darcy.ny+3)*(darcy.nz+3); // with ghost nodes
}

// Free memory
void DEM::freeDarcyMem()
{
    delete[] darcy.p;
    delete[] darcy.v;
    delete[] darcy.phi;
    delete[] darcy.dphi;
    delete[] darcy.norm;
    delete[] darcy.f_p;
    delete[] darcy.k;
    delete[] darcy.p_constant;
}

// 3D index to 1D index
unsigned int DEM::d_idx(
        const int x,
        const int y,
        const int z)
{
    // without ghost nodes
    //return x + d.nx*y + d.nx*d.ny*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (darcy.nx+2)*(y+1) + (darcy.nx+2)*(darcy.ny+2)*(z+1);
}

// 3D index to 1D index of cell-face velocity nodes. The cell-face velocities
// are placed at x = [0;nx], y = [0;ny], z = [0;nz].
// The coordinate x,y,z corresponds to the lowest corner of cell(x,y,z).
unsigned int DEM::d_vidx(
        const int x,
        const int y,
        const int z)
{
    //return x + (darcy.nx+1)*y + (darcy.nx+1)*(darcy.ny+1)*z; // without ghost nodes
    return (x+1) + (darcy.nx+3)*(y+1) + (darcy.nx+3)*(darcy.ny+3)*(z+1); // with ghost nodes
}

Float DEM::largestDarcyPermeability()
{
    Float k_max = 0.0;
    for (unsigned int z=0; z<grid.num[2]; z++)
        for (unsigned int y=0; y<grid.num[1]; y++)
            for (unsigned int x=0; x<grid.num[0]; x++)
                if (darcy.k[d_idx(x,y,z)] > k_max)
                    k_max = darcy.k[d_idx(x,y,z)];
    return k_max;
}

Float DEM::smallestDarcyPorosity()
{
    Float phi_min = 10.0;
    for (unsigned int z=0; z<grid.num[2]; z++)
        for (unsigned int y=0; y<grid.num[1]; y++)
            for (unsigned int x=0; x<grid.num[0]; x++)
                if (darcy.phi[d_idx(x,y,z)] < phi_min)
                    phi_min = darcy.phi[d_idx(x,y,z)];
    return phi_min;
}

// Component-wise max absolute velocities
Float3 DEM::largestDarcyVelocities()
{
    Float3 v_max_abs = MAKE_FLOAT3(0.0, 0.0, 0.0);
    Float3 v;
    for (unsigned int z=0; z<grid.num[2]; z++)
        for (unsigned int y=0; y<grid.num[1]; y++)
            for (unsigned int x=0; x<grid.num[0]; x++) {
                v = darcy.v[d_idx(x,y,z)];
                if (v.x > v_max_abs.x)
                    v_max_abs.x = fabs(v.x);
                if (v.y > v_max_abs.y)
                    v_max_abs.y = fabs(v.y);
                if (v.z > v_max_abs.z)
                    v_max_abs.z = fabs(v.z);
            }
    return v_max_abs;
}

// Determine if the FTCS (forward time, central space) solution of the Navier
// Stokes equations is unstable
void DEM::checkDarcyStability()
{
    // Cell dimensions
    const Float dx = grid.L[0]/grid.num[0];
    const Float dy = grid.L[1]/grid.num[1];
    const Float dz = grid.L[2]/grid.num[2];

    /*const Float alpha_max = largestDarcyPermeability()
        /(darcy.beta_f*smallestDarcyPorosity()*darcy.mu);

    // von Neumann stability analysis
    if (time.dt >= 1.0/(2.0*alpha_max) *
            1.0/(1.0/(dx*dx) + 1.0/(dy*dy) + 1.0/(dz*dz))) {
        std::cerr
            << "\nError: The time step is too large to ensure stability in "
            "the diffusive term of the fluid momentum equation.\n"
            "Decrease the time step, increase the fluid viscosity, increase "
            "the fluid compressibility and/or increase "
            "the fluid grid cell size." << std::endl;
        //exit(1);
    }*/

    // Courant-Friedrichs-Lewy criteria
    Float3 v_max_abs = largestDarcyVelocities();
    if (v_max_abs.x*time.dt > dx ||
            v_max_abs.y*time.dt > dy ||
            v_max_abs.z*time.dt > dz) {
        std::cerr
            << "\nError: The time step is too large to ensure stability due to "
            "large fluid velocities.\n v_max_abs = "
            << v_max_abs.x << ", "
            << v_max_abs.y << ", "
            << v_max_abs.z <<
            " m/s.\nDecrease the time step "
            "and/or increase the fluid grid cell size." << std::endl;
    }
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printDarcyArray(FILE* stream, Float* arr)
{
    int x, y, z;

    // show ghost nodes
    for (z = darcy.nz; z >= -1; z--) { // top to bottom

        fprintf(stream, "z = %d\n", z);

        for (y=-1; y<=darcy.ny; y++) {
            for (x=-1; x<=darcy.nx; x++) {

                if (x > -1 && x < darcy.nx &&
                        y > -1 && y < darcy.ny &&
                        z > -1 && z < darcy.nz) {
                    fprintf(stream, "%f\t", arr[d_idx(x,y,z)]);
                } else { // ghost node
                    if (color_output) {
                        fprintf(stream, "\x1b[30;1m%f\x1b[0m\t",
                                arr[d_idx(x,y,z)]);
                    } else {
                        fprintf(stream, "%f\t", arr[d_idx(x,y,z)]);
                    }
                }
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printDarcyArray to add optional description
void DEM::printDarcyArray(FILE* stream, Float* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printDarcyArray(stream, arr);
}

// Print array values to file stream (stdout, stderr, other file)
void DEM::printDarcyArray(FILE* stream, Float3* arr)
{
    int x, y, z;
    for (z=0; z<darcy.nz; z++) {
        for (y=0; y<darcy.ny; y++) {
            for (x=0; x<darcy.nx; x++) {
                fprintf(stream, "%f,%f,%f\t",
                        arr[d_idx(x,y,z)].x,
                        arr[d_idx(x,y,z)].y,
                        arr[d_idx(x,y,z)].z);
            }
            fprintf(stream, "\n");
        }
        fprintf(stream, "\n");
    }
}

// Overload printDarcyArray to add optional description
void DEM::printDarcyArray(FILE* stream, Float3* arr, std::string desc)
{
    std::cout << "\n" << desc << ":\n";
    printDarcyArray(stream, arr);
}

// Returns the average value of the normalized residuals
double DEM::avgNormResDarcy()
{
    double norm_res_sum, norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = static_cast<double>(darcy.norm[d_idx(x,y,z)]);
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
double DEM::maxNormResDarcy()
{
    double max_norm_res = -1.0e9; // initialized to a small number
    double norm_res;

    // do not consider the values of the ghost nodes
    for (int z=0; z<grid.num[2]; ++z) {
        for (int y=0; y<grid.num[1]; ++y) {
            for (int x=0; x<grid.num[0]; ++x) {
                norm_res = fabs(static_cast<double>(darcy.norm[d_idx(x,y,z)]));
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
void DEM::initDarcy()
{
    // Cell size 
    darcy.dx = grid.L[0]/darcy.nx;
    darcy.dy = grid.L[1]/darcy.ny;
    darcy.dz = grid.L[2]/darcy.nz;

    if (verbose == 1) {
        std::cout << "  - Fluid grid dimensions: "
            << darcy.nx << "*"
            << darcy.ny << "*"
            << darcy.nz << std::endl;
        std::cout << "  - Fluid grid cell size: "
            << darcy.dx << "*"
            << darcy.dy << "*"
            << darcy.dz << std::endl;
    }
}

// Write values in scalar field to file
void DEM::writeDarcyArray(Float* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printDarcyArray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}

// Write values in vector field to file
void DEM::writeDarcyArray(Float3* array, const char* filename)
{
    FILE* file;
    if ((file = fopen(filename,"w"))) {
        printDarcyArray(file, array);
        fclose(file);
    } else {
        fprintf(stderr, "Error, could not open %s.\n", filename);
    }
}


// Print final heads and free memory
void DEM::endDarcy()
{
    // Write arrays to stdout/text files for debugging
    //writeDarcyArray(darcy.phi, "ns_phi.txt");

    freeDarcyMem();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
