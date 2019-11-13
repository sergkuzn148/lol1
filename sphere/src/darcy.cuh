// darcy.cuh
// CUDA implementation of Darcy porous flow

#include <iostream>
#include <cuda.h>
//#include <cutil_math.h>
#include <helper_math.h>

#include "vector_arithmetic.h"  // for arbitrary precision vectors
#include "sphere.h"
#include "datatypes.h"
#include "utility.h"
#include "constants.cuh"
#include "debug.h"

// Initialize memory
void DEM::initDarcyMemDev(void)
{
    // size of scalar field
    unsigned int memSizeF = sizeof(Float)*darcyCells();

    cudaMalloc((void**)&dev_darcy_dp_expl, memSizeF); // Expl. pressure change
    cudaMalloc((void**)&dev_darcy_p_old, memSizeF); // old pressure
    cudaMalloc((void**)&dev_darcy_p, memSizeF);     // hydraulic pressure
    cudaMalloc((void**)&dev_darcy_p_new, memSizeF); // updated pressure
    cudaMalloc((void**)&dev_darcy_v, memSizeF*3);   // cell hydraulic velocity
    cudaMalloc((void**)&dev_darcy_vp_avg, memSizeF*3); // avg. particle velocity
    cudaMalloc((void**)&dev_darcy_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_darcy_dphi, memSizeF);  // cell porosity change
    cudaMalloc((void**)&dev_darcy_norm, memSizeF);  // normalized residual
    cudaMalloc((void**)&dev_darcy_f_p, sizeof(Float4)*np); // pressure force
    cudaMalloc((void**)&dev_darcy_k, memSizeF);        // hydraulic permeability
    cudaMalloc((void**)&dev_darcy_grad_k, memSizeF*3); // grad(permeability)
    cudaMalloc((void**)&dev_darcy_div_v_p, memSizeF);  // divergence(v_p)
    cudaMalloc((void**)&dev_darcy_grad_p, memSizeF*3); // grad(pressure)
    cudaMalloc((void**)&dev_darcy_p_constant,
            sizeof(int)*darcyCells()); // grad(pressure)

    checkForCudaErrors("End of initDarcyMemDev");
}

// Free memory
void DEM::freeDarcyMemDev()
{
    cudaFree(dev_darcy_dp_expl);
    cudaFree(dev_darcy_p_old);
    cudaFree(dev_darcy_p);
    cudaFree(dev_darcy_p_new);
    cudaFree(dev_darcy_v);
    cudaFree(dev_darcy_vp_avg);
    cudaFree(dev_darcy_phi);
    cudaFree(dev_darcy_dphi);
    cudaFree(dev_darcy_norm);
    cudaFree(dev_darcy_f_p);
    cudaFree(dev_darcy_k);
    cudaFree(dev_darcy_grad_k);
    cudaFree(dev_darcy_div_v_p);
    cudaFree(dev_darcy_grad_p);
    cudaFree(dev_darcy_p_constant);
}

// Transfer to device
void DEM::transferDarcyToGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
            "transferDarcyToGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "  Transfering fluid data to the device:           ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*darcyCells();

    //writeNSarray(ns.p, "ns.p.txt");

    cudaMemcpy(dev_darcy_p, darcy.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferDarcytoGlobalDeviceMemory after first "
            "cudaMemcpy");
    cudaMemcpy(dev_darcy_v, darcy.v, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_phi, darcy.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_dphi, darcy.dphi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_f_p, darcy.f_p, sizeof(Float4)*np,
            cudaMemcpyHostToDevice);
    cudaMemcpy(dev_darcy_p_constant, darcy.p_constant, sizeof(int)*darcyCells(),
            cudaMemcpyHostToDevice);

    checkForCudaErrors("End of transferDarcyToGlobalDeviceMemory");
    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "Done" << std::endl;
}

// Transfer from device
void DEM::transferDarcyFromGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering fluid data from the device:         ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*darcyCells();

    cudaMemcpy(darcy.p, dev_darcy_p, memSizeF, cudaMemcpyDeviceToHost);
    checkForCudaErrors("In transferDarcyFromGlobalDeviceMemory, dev_darcy_p", 0);
    cudaMemcpy(darcy.v, dev_darcy_v, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.phi, dev_darcy_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.dphi, dev_darcy_dphi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.f_p, dev_darcy_f_p, sizeof(Float4)*np,
            cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.k, dev_darcy_k, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(darcy.p_constant, dev_darcy_p_constant, sizeof(int)*darcyCells(),
            cudaMemcpyDeviceToHost);

    checkForCudaErrors("End of transferDarcyFromGlobalDeviceMemory", 0);
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
}

// Transfer the normalized residuals from device to host
void DEM::transferDarcyNormFromGlobalDeviceMemory()
{
    cudaMemcpy(darcy.norm, dev_darcy_norm, sizeof(Float)*darcyCells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferDarcyNormFromGlobalDeviceMemory");
}

// Transfer the pressures from device to host
void DEM::transferDarcyPressuresFromGlobalDeviceMemory()
{
    cudaMemcpy(darcy.p, dev_darcy_p, sizeof(Float)*darcyCells(),
            cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferDarcyNormFromGlobalDeviceMemory");
}

// Get linear index from 3D grid position
__inline__ __device__ unsigned int d_idx(
        const int x, const int y, const int z)
{
    // without ghost nodes
    //return x + dev_grid.num[0]*y + dev_grid.num[0]*dev_grid.num[1]*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH
    return (x+1) + (devC_grid.num[0]+2)*(y+1) +
        (devC_grid.num[0]+2)*(devC_grid.num[1]+2)*(z+1);
}

// Get linear index of velocity node from 3D grid position in staggered grid
__inline__ __device__ unsigned int d_vidx(
        const int x, const int y, const int z)
{
    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH+1
    return (x+1) + (devC_grid.num[0]+3)*(y+1)
        + (devC_grid.num[0]+3)*(devC_grid.num[1]+3)*(z+1);
}

// The normalized residuals are given an initial value of 0, since the values at
// the Dirichlet boundaries aren't written during the iterations.
__global__ void setDarcyNormZero(
        Float* __restrict__ dev_darcy_norm)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        dev_darcy_norm[d_idx(x,y,z)] = 0.0;
    }
}

// Set an array of scalars to 0.0 inside devC_grid
    template<typename T>
__global__ void setDarcyZeros(T* __restrict__ dev_scalarfield)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        dev_scalarfield[d_idx(x,y,z)] = 0.0;
    }
}


// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid using
// setDarcyGhostNodes<datatype><<<.. , ..>>>( .. );
    template<typename T>
__global__ void setDarcyGhostNodes(
        T* __restrict__ dev_scalarfield,
        const int bc_xn,
        const int bc_xp,
        const int bc_yn,
        const int bc_yp,
        const int bc_bot,
        const int bc_top)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const T val = dev_scalarfield[d_idx(x,y,z)];

        // x
        if (x == 0 && bc_xn == 0)
            dev_scalarfield[idx(-1,y,z)] = val;     // Dirichlet
        if (x == 1 && bc_xn == 1)
            dev_scalarfield[idx(-1,y,z)] = val;     // Neumann
        if (x == 0 && bc_xn == 2)
            dev_scalarfield[idx(nx,y,z)] = val;     // Periodic -y

        if (x == nx-1 && bc_xp == 0)
            dev_scalarfield[idx(nx,y,z)] = val;     // Dirichlet
        if (x == nx-2 && bc_xp == 1)
            dev_scalarfield[idx(nx,y,z)] = val;     // Neumann
        if (x == nx-1 && bc_xp == 2)
            dev_scalarfield[idx(-1,y,z)] = val;     // Periodic +x

        // y
        if (y == 0 && bc_yn == 0)
            dev_scalarfield[idx(x,-1,z)] = val;     // Dirichlet
        if (y == 1 && bc_yn == 1)
            dev_scalarfield[idx(x,-1,z)] = val;     // Neumann
        if (y == 0 && bc_yn == 2)
            dev_scalarfield[idx(x,ny,z)] = val;     // Periodic -y

        if (y == ny-1 && bc_yp == 0)
            dev_scalarfield[idx(x,ny,z)] = val;     // Dirichlet
        if (y == ny-2 && bc_yp == 1)
            dev_scalarfield[idx(x,ny,z)] = val;     // Neumann
        if (y == ny-1 && bc_yp == 2)
            dev_scalarfield[idx(x,-1,z)] = val;     // Periodic +y

        // z
        if (z == 0 && bc_bot == 0)
            dev_scalarfield[idx(x,y,-1)] = val;     // Dirichlet
        if (z == 1 && bc_bot == 1)
            dev_scalarfield[idx(x,y,-1)] = val;     // Neumann
        if (z == 0 && bc_bot == 2)
            dev_scalarfield[idx(x,y,nz)] = val;     // Periodic -z

        if (z == nz-1 && bc_top == 0)
            dev_scalarfield[idx(x,y,nz)] = val;     // Dirichlet
        if (z == nz-2 && bc_top == 1)
            dev_scalarfield[idx(x,y,nz)] = val;     // Neumann
        if (z == nz-1 && bc_top == 2)
            dev_scalarfield[idx(x,y,-1)] = val;     // Periodic +z
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid using
// setDarcyGhostNodes<datatype><<<.. , ..>>>( .. );
    template<typename T>
__global__ void setDarcyGhostNodesFlux(
        T* __restrict__ dev_scalarfield, // out
        const int bc_bot, // in
        const int bc_top, // in
        const Float bc_bot_flux, // in
        const Float bc_top_flux, // in
        const Float* __restrict__ dev_darcy_k, // in
        const Float mu) // in
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz && (bc_bot == 4 || bc_top == 4)) {

        const T p = dev_scalarfield[d_idx(x,y,z)];
        const Float k = dev_darcy_k[d_idx(x,y,z)];
        const Float dz = devC_grid.L[2]/nz;

        Float q_z = 0.;
        if (z == 0)
            q_z = bc_bot_flux;
        else if (z == nz-1)
            q_z = bc_top_flux;

        const Float p_ghost = -mu/k*q_z * dz + p;

        // z
        if (z == 0 && bc_bot == 4)
            dev_scalarfield[idx(x,y,-1)] = p_ghost;

        if (z == nz-1 && bc_top == 4)
            dev_scalarfield[idx(x,y,nz)] = p_ghost;
    }
}

// Return the volume of a sphere with radius r
__forceinline__ __device__ Float sphereVolume(const Float r)
{
    return 4.0/3.0*M_PI*pow(r, 3);
}

__device__ Float3 abs(const Float3 v)
{
    return MAKE_FLOAT3(abs(v.x), abs(v.y), abs(v.z));
}
            

// Returns a weighting factor based on particle position and fluid center
// position
__device__ Float weight(
        const Float3 x_p, // in: Particle center position
        const Float3 x_f, // in: Fluid pressure node position
        const Float  dx,  // in: Cell spacing, x
        const Float  dy,  // in: Cell spacing, y
        const Float  dz)  // in: Cell spacing, z
{
    const Float3 dist = abs(x_p - x_f);
    if (dist.x < dx && dist.y < dy && dist.z < dz)
        return (1.0 - dist.x/dx)*(1.0 - dist.y/dy)*(1.0 - dist.z/dz);
    else
        return 0.0;
}

// Returns a weighting factor based on particle position and fluid center
// position
__device__ Float weightDist(
        const Float3 x,   // in: Vector between cell and particle center
        const Float  dx,  // in: Cell spacing, x
        const Float  dy,  // in: Cell spacing, y
        const Float  dz)  // in: Cell spacing, z
{
    const Float3 dist = abs(x);
    if (dist.x < dx && dist.y < dy && dist.z < dz)
        return (1.0 - dist.x/dx)*(1.0 - dist.y/dy)*(1.0 - dist.z/dz);
    else
        return 0.0;
}

// Find the porosity in each cell on the base of a bilinear weighing scheme, 
// centered at the cell center. 
__global__ void findDarcyPorositiesLinear(
        const unsigned int* __restrict__ dev_cellStart,   // in
        const unsigned int* __restrict__ dev_cellEnd,     // in
        const Float4* __restrict__ dev_x_sorted,          // in
        const Float4* __restrict__ dev_vel_sorted,        // in
        const unsigned int iteration,                     // in
        const unsigned int ndem,                          // in
        const unsigned int np,                            // in
        const Float c_phi,                                // in
        Float*  __restrict__ dev_darcy_phi,               // in + out
        Float*  __restrict__ dev_darcy_dphi,              // in + out
        Float*  __restrict__ dev_darcy_div_v_p,           // out
        Float3* __restrict__ dev_darcy_vp_avg)            // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell dimensions
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    Float solid_volume = 0.0;
    Float solid_volume_new = 0.0;
    Float4 xr;  // particle pos. and radius

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (np > 0) {

            // Cell center position
            const Float3 X = MAKE_FLOAT3(
                    x*dx + 0.5*dx,
                    y*dy + 0.5*dy,
                    z*dz + 0.5*dz);

            //Float d, r;
            Float s, vol_p;
            Float phi = 1.00;
            Float4 v;
            Float3 vp_avg_num   = MAKE_FLOAT3(0.0, 0.0, 0.0);
            Float  vp_avg_denum = 0.0;

            // Read old porosity
            __syncthreads();
            Float phi_0 = dev_darcy_phi[d_idx(x,y,z)];

            // The cell 3d index
            const int3 gridPos = make_int3((int)x,(int)y,(int)z);

            // The neighbor cell 3d index
            int3 targetCell;

            // The distance modifier for particles across periodic boundaries
            Float3 dist, distmod;

            unsigned int cellID, startIdx, endIdx, i;

            // Iterate over 27 neighbor cells, R = cell width
            for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                    for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis

                        // Index of neighbor cell this iteration is looking at
                        targetCell = gridPos + make_int3(x_dim, y_dim, z_dim);

                        // Do not artifically enhance porosity at lower boundary
                        if (targetCell.z == -1)
                            targetCell.z = 1;

                        // Mirror particle grid at frictionless boundaries
                        if (devC_grid.periodic == 2) {
                            if (targetCell.y == -1) {
                                targetCell.y = 1;
                            }
                            if (targetCell.y == devC_grid.num[1]) {
                                targetCell.y = devC_grid.num[1] - 2;
                            }
                        }

                        // Get distance modifier for interparticle
                        // vector, if it crosses a periodic boundary
                        distmod = MAKE_FLOAT3(0.0, 0.0, 0.0);
                        if (findDistMod(&targetCell, &distmod) != -1) {

                            // Calculate linear cell ID
                            cellID = targetCell.x
                                + targetCell.y * devC_grid.num[0]
                                + (devC_grid.num[0] * devC_grid.num[1])
                                * targetCell.z;

                            // Lowest particle index in cell
                            __syncthreads();
                            startIdx = dev_cellStart[cellID];

                            // Make sure cell is not empty
                            if (startIdx != 0xffffffff) {

                                // Highest particle index in cell
                                __syncthreads();
                                endIdx = dev_cellEnd[cellID];

                                // Iterate over cell particles
                                for (i=startIdx; i<endIdx; ++i) {

                                    // Read particle position and radius
                                    __syncthreads();
                                    xr = dev_x_sorted[i];
                                    v  = dev_vel_sorted[i];
                                    //x3 = MAKE_FLOAT3(xr.x, xr.y, xr.z);
                                    //v3 = MAKE_FLOAT3(v.x, v.y, v.z);

                                    // Find center distance
                                    dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z);
                                    dist += distmod;
                                    s = weightDist(dist, dx, dy, dz);
                                    vol_p = sphereVolume(xr.w);

                                    solid_volume += s*vol_p;

                                    // Summation procedure for average velocity
                                    vp_avg_num +=
                                        s*vol_p*MAKE_FLOAT3(v.x, v.y, v.z);
                                    vp_avg_denum += s*vol_p;

                                    // Find projected new void volume
                                    // Eulerian update of positions
                                    xr += v*devC_dt;
                                    dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z) + distmod;
                                    solid_volume_new +=
                                        weightDist(dist, dx, dy, dz)*vol_p;
                                }
                            }
                        }
                    }
                }
            }

            Float cell_volume = dx*dy*dz;
            if (z == nz - 1)
                cell_volume *= 0.875;

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(0.9, fmax(0.1, 1.0 - solid_volume/cell_volume));
            Float phi_new = fmin(0.9, fmax(0.1,
                        1.0 - solid_volume_new/cell_volume));

            Float dphi;
            Float3 vp_avg;
            if (iteration == 0) {
                dphi = 0.0;
                vp_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);
            } else {
                dphi = 0.5*(phi_new - phi_0);
                vp_avg = vp_avg_num/fmax(1.0e-16, vp_avg_denum);
            }

            // Save porosity and porosity change
            __syncthreads();
            const unsigned int cellidx = d_idx(x,y,z);
            dev_darcy_phi[cellidx]     = phi*c_phi;
            dev_darcy_dphi[cellidx]    = dphi*c_phi;
            dev_darcy_vp_avg[cellidx] = vp_avg;

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
#endif
        } else {

            __syncthreads();
            const unsigned int cellidx = d_idx(x,y,z);

            dev_darcy_phi[cellidx]  = 0.9;
            dev_darcy_dphi[cellidx] = 0.0;
        }
    }
}


// Copy the porosity, porosity change, div_v_p and vp_avg values to the grid 
// edges from the grid interior at the frictionless Y boundaries (grid.periodic 
// == 2).
__global__ void copyDarcyPorositiesToEdges(
        Float*  __restrict__ dev_darcy_phi,               // in + out
        Float*  __restrict__ dev_darcy_dphi,              // in + out
        Float*  __restrict__ dev_darcy_div_v_p,           // in + out
        Float3* __restrict__ dev_darcy_vp_avg)            // in + out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // check that we are not outside the fluid grid
    if (devC_grid.periodic == 2 &&
            x < nx && (y == 0 || y == ny - 1) && z < nz) {

            // Read porosities from this cell
            int y_read;

            // read values from inside cells
            if (y == 0)
                y_read = 1;
            if (y == ny - 1)
                y_read = ny - 2;

            const unsigned int readidx = d_idx(x, y_read, z);
            const unsigned int writeidx = d_idx(x, y, z);

            __syncthreads();
            dev_darcy_phi[writeidx] = dev_darcy_phi[readidx];
            dev_darcy_dphi[writeidx] = dev_darcy_dphi[readidx];
            dev_darcy_div_v_p[writeidx] = dev_darcy_div_v_p[readidx];
            dev_darcy_vp_avg[writeidx] = dev_darcy_vp_avg[readidx];
    }
}


// Copy the porosity, porosity change, div_v_p and vp_avg values to the grid 
// bottom from the grid interior at the frictionless Y boundaries (grid.periodic 
// == 2).
__global__ void copyDarcyPorositiesToBottom(
        Float*  __restrict__ dev_darcy_phi,               // in + out
        Float*  __restrict__ dev_darcy_dphi,              // in + out
        Float*  __restrict__ dev_darcy_div_v_p,           // in + out
        Float3* __restrict__ dev_darcy_vp_avg)            // in + out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];

    // check that we are not outside the fluid grid
    if (devC_grid.periodic == 2 &&
            x < nx && y < ny && z == 0) {

            const unsigned int readidx = d_idx(x, y, 1);
            const unsigned int writeidx = d_idx(x, y, z);

            __syncthreads();
            dev_darcy_phi[writeidx] = dev_darcy_phi[readidx];
            dev_darcy_dphi[writeidx] = dev_darcy_dphi[readidx];
            dev_darcy_div_v_p[writeidx] = dev_darcy_div_v_p[readidx];
            dev_darcy_vp_avg[writeidx] = dev_darcy_vp_avg[readidx];
    }
}


// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
__global__ void findDarcyPorosities(
        const unsigned int* __restrict__ dev_cellStart,   // in
        const unsigned int* __restrict__ dev_cellEnd,     // in
        const Float4* __restrict__ dev_x_sorted,          // in
        const Float4* __restrict__ dev_vel_sorted,        // in
        const unsigned int iteration,                     // in
        const unsigned int ndem,                          // in
        const unsigned int np,                            // in
        const Float c_phi,                                // in
        Float*  __restrict__ dev_darcy_phi,               // in + out
        Float*  __restrict__ dev_darcy_dphi)              // in + out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell dimensions
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Cell sphere radius
    const Float R = fmin(dx, fmin(dy,dz)) * 0.5; // diameter = cell width
    const Float cell_volume = sphereVolume(R);

    Float void_volume = cell_volume;     // current void volume
    Float void_volume_new = cell_volume; // projected new void volume
    Float4 xr;  // particle pos. and radius

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (np > 0) {

            // Cell sphere center position
            const Float3 X = MAKE_FLOAT3(
                    x*dx + 0.5*dx,
                    y*dy + 0.5*dy,
                    z*dz + 0.5*dz);

            Float d, r;
            Float phi = 1.00;
            Float4 v;
            //unsigned int n = 0;

            // Read old porosity
            __syncthreads();
            Float phi_0 = dev_darcy_phi[d_idx(x,y,z)];

            // The cell 3d index
            const int3 gridPos = make_int3((int)x,(int)y,(int)z);

            // The neighbor cell 3d index
            int3 targetCell;

            // The distance modifier for particles across periodic boundaries
            Float3 dist, distmod;

            unsigned int cellID, startIdx, endIdx, i;

            // Iterate over 27 neighbor cells, R = cell width
            for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                    for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis

                        // Index of neighbor cell this iteration is looking at
                        targetCell = gridPos + make_int3(x_dim, y_dim, z_dim);

                        // Get distance modifier for interparticle
                        // vector, if it crosses a periodic boundary
                        distmod = MAKE_FLOAT3(0.0, 0.0, 0.0);
                        if (findDistMod(&targetCell, &distmod) != -1) {

                            // Calculate linear cell ID
                            cellID = targetCell.x
                                + targetCell.y * devC_grid.num[0]
                                + (devC_grid.num[0] * devC_grid.num[1])
                                * targetCell.z;

                            // Lowest particle index in cell
                            __syncthreads();
                            startIdx = dev_cellStart[cellID];

                            // Make sure cell is not empty
                            if (startIdx != 0xffffffff) {

                                // Highest particle index in cell
                                __syncthreads();
                                endIdx = dev_cellEnd[cellID];

                                // Iterate over cell particles
                                for (i=startIdx; i<endIdx; ++i) {

                                    // Read particle position and radius
                                    __syncthreads();
                                    xr = dev_x_sorted[i];
                                    v  = dev_vel_sorted[i];
                                    r = xr.w;

                                    // Find center distance
                                    dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z);
                                    dist += distmod;
                                    d = length(dist);

                                    // Lens shaped intersection
                                    if ((R - r) < d && d < (R + r))
                                        void_volume -=
                                            1.0/(12.0*d) * (
                                                    M_PI*(R + r - d)*(R + r - d)
                                                    *(d*d + 2.0*d*r - 3.0*r*r
                                                        + 2.0*d*R + 6.0*r*R
                                                        - 3.0*R*R) );

                                    // Particle fully contained in cell sphere
                                    if (d <= R - r)
                                        void_volume -= sphereVolume(r);

                                    //// Find projected new void volume
                                    // Eulerian update of positions
                                    xr += v*devC_dt;
                                    
                                    // Find center distance
                                    dist = MAKE_FLOAT3(
                                            X.x - xr.x, 
                                            X.y - xr.y,
                                            X.z - xr.z);
                                    dist += distmod;
                                    d = length(dist);

                                    // Lens shaped intersection
                                    if ((R - r) < d && d < (R + r))
                                        void_volume_new -=
                                            1.0/(12.0*d) * (
                                                    M_PI*(R + r - d)*(R + r - d)
                                                    *(d*d + 2.0*d*r - 3.0*r*r
                                                        + 2.0*d*R + 6.0*r*R
                                                        - 3.0*R*R) );

                                    // Particle fully contained in cell sphere
                                    if (d <= R - r)
                                        void_volume_new -= sphereVolume(r);
                                }
                            }
                        }
                    }
                }
            }

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(0.9, fmax(0.1, void_volume/cell_volume));
            Float phi_new = fmin(0.9, fmax(0.1, void_volume_new/cell_volume));

            // Central difference after first iteration
            Float dphi;
            if (iteration == 0)
                dphi = phi_new - phi;
            else
                dphi = 0.5*(phi_new - phi_0);

            // Save porosity and porosity change
            __syncthreads();
            //phi = 0.5; dphi = 0.0; // disable porosity effects
            const unsigned int cellidx = d_idx(x,y,z);
            dev_darcy_phi[cellidx]  = phi*c_phi;
            dev_darcy_dphi[cellidx] = dphi*c_phi;

            /*printf("\n%d,%d,%d: findDarcyPorosities\n"
                    "\tphi     = %f\n"
                    "\tdphi    = %e\n"
                    "\tdphi_eps= %e\n"
                    "\tX       = %e, %e, %e\n"
                    "\txr      = %e, %e, %e\n"
                    "\tq       = %e\n"
                    "\tdiv_v_p = %e\n"
                    , x,y,z,
                    phi, dphi,
                    dot_epsilon_kk*(1.0 - phi)*devC_dt,
                    X.x, X.y, X.z,
                    xr.x, xr.y, xr.z,
                    q,
                    dot_epsilon_kk);*/

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
#endif
        } else {

            __syncthreads();
            const unsigned int cellidx = d_idx(x,y,z);

            dev_darcy_phi[cellidx]  = 0.999;
            dev_darcy_dphi[cellidx] = 0.0;
        }
    }
}

// Find the particle velocity divergence at the cell center from the average
// particle velocities on the cell faces
__global__ void findDarcyParticleVelocityDivergence(
        const Float* __restrict__ dev_darcy_v_p_x,  // in
        const Float* __restrict__ dev_darcy_v_p_y,  // in
        const Float* __restrict__ dev_darcy_v_p_z,  // in
        Float* __restrict__ dev_darcy_div_v_p)      // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    if (x < nx && y < ny && z < nz) {

        // read values
        __syncthreads();
        Float v_p_xn = dev_darcy_v_p_x[d_vidx(x,y,z)];
        Float v_p_xp = dev_darcy_v_p_x[d_vidx(x+1,y,z)];
        Float v_p_yn = dev_darcy_v_p_y[d_vidx(x,y,z)];
        Float v_p_yp = dev_darcy_v_p_y[d_vidx(x,y+1,z)];
        Float v_p_zn = dev_darcy_v_p_z[d_vidx(x,y,z)];
        Float v_p_zp = dev_darcy_v_p_z[d_vidx(x,y,z+1)];

        // cell dimensions
        const Float dx = devC_grid.L[0]/nx;
        const Float dy = devC_grid.L[1]/ny;
        const Float dz = devC_grid.L[2]/nz;

        // calculate the divergence using first order central finite differences
        const Float div_v_p =
            (v_p_xp - v_p_xn)/dx +
            (v_p_yp - v_p_yn)/dy +
            (v_p_zp - v_p_zn)/dz;

        __syncthreads();
        dev_darcy_div_v_p[d_idx(x,y,z)] = div_v_p;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("div_v_p", x, y, z, div_v_p);
#endif
    }
}

// Find the cell-centered pressure gradient using central differences
__global__ void findDarcyPressureGradient(
        const Float* __restrict__ dev_darcy_p,  // in
        Float3* __restrict__ dev_darcy_grad_p)  // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    if (x < nx && y < ny && z < nz) {

        // read values
        __syncthreads();
        Float p_xn = dev_darcy_p[d_idx(x-1,y,z)];
        Float p_xp = dev_darcy_p[d_idx(x+1,y,z)];
        Float p_yn = dev_darcy_p[d_idx(x,y-1,z)];
        Float p_yp = dev_darcy_p[d_idx(x,y+1,z)];
        Float p_zn = dev_darcy_p[d_idx(x,y,z-1)];
        Float p_zp = dev_darcy_p[d_idx(x,y,z+1)];

        // cell dimensions
        const Float dx = devC_grid.L[0]/nx;
        const Float dy = devC_grid.L[1]/ny;
        const Float dz = devC_grid.L[2]/nz;

        // calculate the divergence using first order central finite differences
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        __syncthreads();
        dev_darcy_grad_p[d_idx(x,y,z)] = grad_p;

        //if (grad_p.x != 0.0 || grad_p.y != 0.0 || grad_p.z != 0.0)
        /*printf("%d,%d,%d findDarcyPressureGradient:\n"
                "\tp_x    = %.2e, %.2e\n"
                "\tp_y    = %.2e, %.2e\n"
                "\tp_z    = %.2e, %.2e\n"
                "\tgrad_p = %.2e, %.2e, %.2e\n",
                x, y, z,
                p_xn, p_xp,
                p_yn, p_yp,
                p_zn, p_zp,
                grad_p.x, grad_p.y, grad_p.z); // */ 
#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("grad_p", x, y, z, grad_p);
#endif
    }
}

// Find particle-fluid interaction force as outlined by Goren et al. 2011, and
// originally by Gidaspow 1992. All terms other than the pressure force are
// neglected. The buoyancy force is included.
__global__ void findDarcyPressureForceLinear(
    const Float4* __restrict__ dev_x,            // in
    const Float3* __restrict__ dev_darcy_grad_p, // in
    const Float*  __restrict__ dev_darcy_phi,    // in
    const unsigned int wall0_iz,                 // in
    const Float rho_f,                           // in
    const int bc_top,                            // in
    Float4* __restrict__ dev_force,              // out
    Float4* __restrict__ dev_darcy_f_p)          // out
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x; // Particle index

    if (i < devC_np) {

        // read particle information
        __syncthreads();
        const Float4 x = dev_x[i];
        const Float3 x3 = MAKE_FLOAT3(x.x, x.y, x.z);

        // determine fluid cell containing the particle
        const unsigned int i_x =
            floor((x.x - devC_grid.origo[0])/(devC_grid.L[0]/devC_grid.num[0]));
        const unsigned int i_y =
            floor((x.y - devC_grid.origo[1])/(devC_grid.L[1]/devC_grid.num[1]));
        const unsigned int i_z =
            floor((x.z - devC_grid.origo[2])/(devC_grid.L[2]/devC_grid.num[2]));
        const unsigned int cellidx = d_idx(i_x, i_y, i_z);

        // determine cell dimensions
        const unsigned int nx = devC_grid.num[0];
        const unsigned int ny = devC_grid.num[1];
        const unsigned int nz = devC_grid.num[2];
        const Float dx = devC_grid.L[0]/nx;
        const Float dy = devC_grid.L[1]/ny;
        const Float dz = devC_grid.L[2]/nz;

        // read fluid information
        __syncthreads();
        //const Float phi = dev_darcy_phi[cellidx];

        // Cell center position
        const Float3 X = MAKE_FLOAT3(
                i_x*dx + 0.5*dx,
                i_y*dy + 0.5*dy,
                i_z*dz + 0.5*dz);

        Float3 grad_p = MAKE_FLOAT3(0., 0., 0.);
        Float3 grad_p_iter, n;
        int ix_n, iy_n, iz_n; // neighbor indexes

        // Loop over 27 closest cells to find all pressure gradient
        // contributions
        for (int d_iz = -1; d_iz<2; d_iz++) {
            for (int d_iy = -1; d_iy<2; d_iy++) {
                for (int d_ix = -1; d_ix<2; d_ix++) {

                    ix_n = i_x + d_ix;
                    iy_n = i_y + d_iy;
                    iz_n = i_z + d_iz;

                    __syncthreads();
                    grad_p_iter = dev_darcy_grad_p[d_idx(ix_n, iy_n, iz_n)];

                    // make sure edge and corner ghost nodes aren't read
                    if (    // edges passing through (0,0,0)
                            (ix_n == -1 && iy_n == -1) ||
                            (ix_n == -1 && iz_n == -1) ||
                            (iy_n == -1 && iz_n == -1) ||

                            // edges passing through (nx,ny,nz)
                            (ix_n == nx && iy_n == ny) ||
                            (ix_n == nx && iz_n == nz) ||
                            (iy_n == ny && iz_n == nz) ||

                            (ix_n == nx && iy_n == -1) ||
                            (ix_n == nx && iz_n == -1) ||

                            (iy_n == ny && ix_n == -1) ||
                            (iy_n == ny && iz_n == -1) ||

                            (iz_n == nz && ix_n == -1) ||
                            (iz_n == nz && iy_n == -1))
                        grad_p_iter = MAKE_FLOAT3(0., 0., 0.);

                    // Add Neumann BC at top wall
                    if (bc_top == 0 && i_z + d_iz >= wall0_iz - 1)
                        grad_p_iter.z = 0.0;

                    n = MAKE_FLOAT3(dx*d_ix, dy*d_iy, dz*d_iz);

                    grad_p += weight(x3, X + n, dx, dy, dz)*grad_p_iter;

                    //*
                    Float s = weight(x3, X + n, dx, dy, dz);
                    /*if (s > 1.0e-12)
                    printf("[%d+%d, %d+%d, %d+%d] findPF nb\n"
                            "\tn      = %f, %f, %f\n"
                            "\tgrad_pi= %.2e, %.2e, %.2e\n"
                            "\ts      = %f\n"
                            "\tgrad_p = %.2e, %.2e, %.2e\n",
                            i_x, d_ix,
                            i_y, d_iy,
                            i_z, d_iz,
                            n.x, n.y, n.z,
                            grad_p_iter.x, grad_p_iter.y, grad_p_iter.z, 
                            s,
                            s*grad_p_iter.x,
                            s*grad_p_iter.y,
                            s*grad_p_iter.z); // */
                }
            }
        }

        // find particle volume (radius in x.w)
        const Float v = sphereVolume(x.w);

        // find pressure gradient force plus buoyancy force.
        // buoyancy force = weight of displaced fluid
        Float3 f_p = -1.0*grad_p*v//(1.0 - phi)
            - rho_f*v*MAKE_FLOAT3(
                    devC_params.g[0],
                    devC_params.g[1],
                    devC_params.g[2]);

        // Add Neumann BC at top wall
        //if (i_z >= wall0_iz - 1)
        if (bc_top == 0 && i_z >= wall0_iz)
            f_p.z = 0.0;

        //if (length(f_p) > 1.0e-12)
        /*printf("%d,%d,%d findPressureForceLinear:\n"
                "\tphi    = %f\n"
                "\tx      = %f, %f, %f\n"
                "\tX      = %f, %f, %f\n"
                "\tgrad_p = %.2e, %.2e, %.2e\n"
                //"\tp_x    = %.2e, %.2e\n"
                //"\tp_y    = %.2e, %.2e\n"
                //"\tp_z    = %.2e, %.2e\n"
                "\tf_p    = %.2e, %.2e, %.2e\n",
                i_x, i_y, i_z,
                phi,
                x3.x, x3.y, x3.z,
                X.x, X.y, X.z,
                grad_p.x, grad_p.y, grad_p.z,
                f_p.x, f_p.y, f_p.z); // */

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("f_p", i_x, i_y, i_z, f_p);
#endif
        // save force
        __syncthreads();
        dev_force[i]    += MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
        dev_darcy_f_p[i] = MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
    }
}

// Find particle-fluid interaction force as outlined by Zhou et al. 2010, and
// originally by Gidaspow 1992. All terms other than the pressure force are
// neglected. The buoyancy force is included.
__global__ void findDarcyPressureForce(
    const Float4* __restrict__ dev_x,           // in
    const Float*  __restrict__ dev_darcy_p,     // in
    //const Float*  __restrict__ dev_darcy_phi,   // in
    const unsigned int wall0_iz,                // in
    const Float rho_f,                          // in
    Float4* __restrict__ dev_force,             // out
    Float4* __restrict__ dev_darcy_f_p)         // out
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x; // Particle index

    if (i < devC_np) {

        // read particle information
        __syncthreads();
        const Float4 x = dev_x[i];

        // determine fluid cell containing the particle
        const unsigned int i_x =
            floor((x.x - devC_grid.origo[0])/(devC_grid.L[0]/devC_grid.num[0]));
        const unsigned int i_y =
            floor((x.y - devC_grid.origo[1])/(devC_grid.L[1]/devC_grid.num[1]));
        const unsigned int i_z =
            floor((x.z - devC_grid.origo[2])/(devC_grid.L[2]/devC_grid.num[2]));
        const unsigned int cellidx = d_idx(i_x, i_y, i_z);

        // determine cell dimensions
        const Float dx = devC_grid.L[0]/devC_grid.num[0];
        const Float dy = devC_grid.L[1]/devC_grid.num[1];
        const Float dz = devC_grid.L[2]/devC_grid.num[2];

        // read fluid information
        __syncthreads();
        //const Float phi = dev_darcy_phi[cellidx];
        const Float p_xn = dev_darcy_p[d_idx(i_x-1,i_y,i_z)];
        const Float p    = dev_darcy_p[cellidx];
        const Float p_xp = dev_darcy_p[d_idx(i_x+1,i_y,i_z)];
        const Float p_yn = dev_darcy_p[d_idx(i_x,i_y-1,i_z)];
        const Float p_yp = dev_darcy_p[d_idx(i_x,i_y+1,i_z)];
        const Float p_zn = dev_darcy_p[d_idx(i_x,i_y,i_z-1)];
        Float p_zp = dev_darcy_p[d_idx(i_x,i_y,i_z+1)];

        // Add Neumann BC at top wall
        if (i_z >= wall0_iz - 1)
            p_zp = p;

        // find particle volume (radius in x.w)
        const Float V = sphereVolume(x.w);

        // determine pressure gradient from first order central difference
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // find pressure gradient force plus buoyancy force.
        // buoyancy force = weight of displaced fluid
        Float3 f_p = -1.0*grad_p*V
            - rho_f*V*MAKE_FLOAT3(
                    devC_params.g[0],
                    devC_params.g[1],
                    devC_params.g[2]);

        // Add Neumann BC at top wall
        if (i_z >= wall0_iz)
            f_p.z = 0.0;

        /*printf("%d,%d,%d findPF:\n"
                "\tphi    = %f\n"
                "\tp      = %f\n"
                "\tgrad_p = % f, % f, % f\n"
                "\tf_p    = % f, % f, % f\n",
                i_x, i_y, i_z,
                phi, p,
                grad_p.x, grad_p.y, grad_p.z,
                f_p.x, f_p.y, f_p.z);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("f_p", i_x, i_y, i_z, f_p);
#endif
        // save force
        __syncthreads();
        dev_force[i]    += MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
        dev_darcy_f_p[i] = MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
    }
}

// Set the pressure at the top boundary to new_pressure
__global__ void setDarcyTopPressure(
    const Float new_pressure,
    Float* __restrict__ dev_darcy_p,
    const unsigned int wall0_iz)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary or at the top wall
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == devC_grid.num[2]-1 || z == wall0_iz) {

        const unsigned int cellidx = idx(x,y,z);

        // Write the new pressure the top boundary cells
        __syncthreads();
        dev_darcy_p[cellidx] = new_pressure;
    }
}

// Set the pressure at the top wall to new_pressure
__global__ void setDarcyTopWallPressure(
    const Float new_pressure,
    const unsigned int wall0_iz,
    Float* __restrict__ dev_darcy_p)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == wall0_iz) {

        const unsigned int cellidx = idx(x,y,z);

        // Write the new pressure the top boundary cells
        __syncthreads();
        dev_darcy_p[cellidx] = new_pressure;
    }
}

// Enforce fixed-flow BC at top wall
__global__ void setDarcyTopWallFixedFlow(
    const unsigned int wall0_iz,
    Float* __restrict__ dev_darcy_p)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == wall0_iz+1) {

        // Write the new pressure the top boundary cells
        __syncthreads();
        const Float new_pressure = dev_darcy_p[idx(x,y,z-2)];
        dev_darcy_p[idx(x,y,z)] = new_pressure;
    }
}


// Find the cell permeabilities from the Kozeny-Carman equation
__global__ void findDarcyPermeabilities(
        const Float k_c,                            // in
        const Float* __restrict__ dev_darcy_phi,    // in
        Float* __restrict__ dev_darcy_k)            // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        __syncthreads();
        Float phi = dev_darcy_phi[cellidx];

        // avoid division by zero
        if (phi > 0.9999)
            phi = 0.9999;

        Float k = k_c*pow(phi,3)/pow(1.0 - phi, 2);

        /*printf("%d,%d,%d findK:\n"
                "\tphi    = %f\n"
                "\tk      = %e\n",
                x, y, z,
                phi, k);*/

        // limit permeability [m*m]
        // K_gravel = 3.0e-2 m/s => k_gravel = 2.7e-9 m*m
        //k = fmin(2.7e-9, k);
        k = fmin(2.7e-10, k);

        __syncthreads();
        dev_darcy_k[cellidx] = k;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("k", x, y, z, k);
#endif
    }
}

// Find the spatial gradients of the permeability.
__global__ void findDarcyPermeabilityGradients(
        const Float*  __restrict__ dev_darcy_k,   // in
        Float3* __restrict__ dev_darcy_grad_k)    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float k_xn = dev_darcy_k[d_idx(x-1,y,z)];
        const Float k_xp = dev_darcy_k[d_idx(x+1,y,z)];
        const Float k_yn = dev_darcy_k[d_idx(x,y-1,z)];
        const Float k_yp = dev_darcy_k[d_idx(x,y+1,z)];
        const Float k_zn = dev_darcy_k[d_idx(x,y,z-1)];
        const Float k_zp = dev_darcy_k[d_idx(x,y,z+1)];

        // gradient approximated by first-order central difference
        const Float3 grad_k = MAKE_FLOAT3(
                (k_xp - k_xn)/(dx+dx),
                (k_yp - k_yn)/(dy+dy),
                (k_zp - k_zn)/(dz+dz));

        // write result
        __syncthreads();
        dev_darcy_grad_k[cellidx] = grad_k;

        /*printf("%d,%d,%d findK:\n"
                "\tk_x     = %e, %e\n"
                "\tk_y     = %e, %e\n"
                "\tk_z     = %e, %e\n"
                "\tgrad(k) = %e, %e, %e\n",
                x, y, z,
                k_xn, k_xp,
                k_yn, k_yp,
                k_zn, k_zp,
                grad_k.x, grad_k.y, grad_k.z);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("grad_k", x, y, z, grad_k);
#endif
    }
}

// Find the temporal gradient in pressure using the backwards Euler method
__global__ void findDarcyPressureChange(
        const Float* __restrict__ dev_darcy_p_old,    // in
        const Float* __restrict__ dev_darcy_p,        // in
        const unsigned int iter,                      // in
        const unsigned int ndem,                      // in
        Float* __restrict__ dev_darcy_dpdt)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float p_old = dev_darcy_p_old[cellidx];
        const Float p     = dev_darcy_p[cellidx];

        Float dpdt = (p - p_old)/(ndem*devC_dt);

        // Ignore the large initial pressure gradients caused by solver "warm
        // up" towards hydrostatic pressure distribution
        if (iter < 2)
            dpdt = 0.0;

        // write result
        __syncthreads();
        dev_darcy_dpdt[cellidx] = dpdt;

        /*printf("%d,%d,%d\n"
                "\tp_old = %e\n"
                "\tp     = %e\n"
                "\tdt    = %e\n"
                "\tdpdt  = %e\n",
                x,y,z,
                p_old, p,
                devC_dt, dpdt);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dpdt", x, y, z, dpdt);
#endif
    }
}

__global__ void firstDarcySolution(
        const Float*  __restrict__ dev_darcy_p,       // in
        const Float*  __restrict__ dev_darcy_k,       // in
        const Float*  __restrict__ dev_darcy_phi,     // in
        const Float*  __restrict__ dev_darcy_dphi,    // in
        const Float*  __restrict__ dev_darcy_div_v_p, // in
        const Float3* __restrict__ dev_darcy_vp_avg,  // in
        const Float3* __restrict__ dev_darcy_grad_k,  // in
        const Float beta_f,                           // in
        const Float mu,                               // in
        const int bc_xn,                              // in
        const int bc_xp,                              // in
        const int bc_yn,                              // in
        const int bc_yp,                              // in
        const int bc_bot,                             // in
        const int bc_top,                             // in
        const unsigned int ndem,                      // in
        const unsigned int wall0_iz,                  // in
        const int* __restrict__ dev_darcy_p_constant, // in
        Float* __restrict__ dev_darcy_dp_expl)        // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float  phi_xn = dev_darcy_phi[d_idx(x-1,y,z)];
        const Float  phi    = dev_darcy_phi[cellidx];
        const Float  phi_xp = dev_darcy_phi[d_idx(x+1,y,z)];
        const Float  phi_yn = dev_darcy_phi[d_idx(x,y-1,z)];
        const Float  phi_yp = dev_darcy_phi[d_idx(x,y+1,z)];
        const Float  phi_zn = dev_darcy_phi[d_idx(x,y,z-1)];
        const Float  phi_zp = dev_darcy_phi[d_idx(x,y,z+1)];
        const Float  dphi   = dev_darcy_dphi[cellidx];
        const Float3 vp_avg = dev_darcy_vp_avg[cellidx];
        const int p_constant = dev_darcy_p_constant[cellidx];

        Float p_xn    = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p = dev_darcy_p[cellidx];
        Float p_xp    = dev_darcy_p[d_idx(x+1,y,z)];
        Float p_yn    = dev_darcy_p[d_idx(x,y-1,z)];
        Float p_yp    = dev_darcy_p[d_idx(x,y+1,z)];
        Float p_zn    = dev_darcy_p[d_idx(x,y,z-1)];
        Float p_zp    = dev_darcy_p[d_idx(x,y,z+1)];

        const Float k_xn    = dev_darcy_k[d_idx(x-1,y,z)];
        const Float k       = dev_darcy_k[cellidx];
        const Float k_xp    = dev_darcy_k[d_idx(x+1,y,z)];
        const Float k_yn    = dev_darcy_k[d_idx(x,y-1,z)];
        const Float k_yp    = dev_darcy_k[d_idx(x,y+1,z)];
        const Float k_zn    = dev_darcy_k[d_idx(x,y,z-1)];
        const Float k_zp    = dev_darcy_k[d_idx(x,y,z+1)];

        // Neumann BCs
        if (x == 0 && bc_xn == 1)
            p_xn = p;
        if (x == nx-1 && bc_xp == 1)
            p_xp = p;
        if (y == 0 && bc_yn == 1)
            p_yn = p;
        if (y == ny-1 && bc_yp == 1)
            p_yp = p;
        if (z == 0 && bc_bot == 1)
            p_zn = p;
        if (z == nz-1 && bc_top == 1)
            p_zp = p;

        const Float3 grad_phi_central = MAKE_FLOAT3(
                (phi_xp - phi_xn)/(dx + dx),
                (phi_yp - phi_yn)/(dy + dy),
                (phi_zp - phi_zn)/(dz + dz));

        // Solve div(k*grad(p)) as a single term, using harmonic mean for 
        // permeability. k_HM*grad(p) is found between the pressure nodes.
        const Float div_k_grad_p =
                (2.*k_xp*k/(k_xp + k) *
                 (p_xp - p)/dx
                 -
                 2.*k_xn*k/(k_xn + k) *
                 (p - p_xn)/dx)/dx
            +
                (2.*k_yp*k/(k_yp + k) *
                 (p_yp - p)/dy
                 -
                 2.*k_yn*k/(k_yn + k) *
                 (p - p_yn)/dy)/dy
            +
                (2.*k_zp*k/(k_zp + k) *
                 (p_zp - p)/dz
                 -
                 2.*k_zn*k/(k_zn + k) *
                 (p - p_zn)/dz)/dz;

        Float dp_expl =
            ndem*devC_dt/(beta_f*phi*mu)*div_k_grad_p
            -(ndem*devC_dt/(beta_f*phi*(1.0 - phi)))
            *(dphi/(ndem*devC_dt) + dot(vp_avg, grad_phi_central));

        // Dirichlet BC at fixed-pressure boundaries and at the dynamic top 
        // wall.  wall0_iz will be larger than the grid if the wall isn't 
        // dynamic
        if ((bc_bot == 0 && z == 0) || (bc_top == 0 && z == nz-1)
                || (z >= wall0_iz && bc_top == 0)
                || (bc_xn == 0 && x == 0) || (bc_xp == 0 && x == nx-1)
                || (bc_yn == 0 && y == 0) || (bc_yp == 0 && y == nx-1)
                || p_constant == 1)
            dp_expl = 0.0;

#ifdef REPORT_FORCING_TERMS
            const Float dp_diff = 
                ndem*devC_dt/(beta_f*phi*mu)
                *div_k_grad_p;
            const Float dp_forc =
                -(ndem*devC_dt/(beta_f*phi*(1.0 - phi)))
                *(dphi/(ndem*devC_dt) + dot(vp_avg, grad_phi));
                
        printf("\n%d,%d,%d firstDarcySolution\n"
                "p           = %e\n"
                "p_x         = %e, %e\n"
                "p_y         = %e, %e\n"
                "p_z         = %e, %e\n"
                "dp_expl     = %e\n"
                "div_k_grad_p= %e\n"
                "dp_diff     = %e\n"
                "dp_forc     = %e\n"
                "phi         = %e\n"
                "dphi        = %e\n"
                "dphi/dt     = %e\n"
                "vp_avg      = %e, %e, %e\n"
                "grad_phi    = %e, %e, %e\n"
                ,
                x,y,z,
                p,
                p_xn, p_xp,
                p_yn, p_yp,
                p_zn, p_zp,
                dp_expl,
                div_k_grad_p,
                dp_diff,
                dp_forc,
                phi,
                dphi,
                dphi/(ndem*devC_dt),
                vp_avg.x, vp_avg.y, vp_avg.z,
                grad_phi_central.x, grad_phi_central.y, grad_phi_central.z
                );
#endif

        // save explicit integrated pressure change
        __syncthreads();
        dev_darcy_dp_expl[cellidx] = dp_expl;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dp_expl", x, y, z, dp_expl);
#endif
    }
}
// A single jacobi iteration where the pressure values are updated to
// dev_darcy_p_new.
// bc = 0: Dirichlet, 1: Neumann
__global__ void updateDarcySolution(
        const Float*  __restrict__ dev_darcy_p_old,   // in
        const Float*  __restrict__ dev_darcy_dp_expl, // in
        const Float*  __restrict__ dev_darcy_p,       // in
        const Float*  __restrict__ dev_darcy_k,       // in
        const Float*  __restrict__ dev_darcy_phi,     // in
        const Float*  __restrict__ dev_darcy_dphi,    // in
        const Float*  __restrict__ dev_darcy_div_v_p, // in
        const Float3* __restrict__ dev_darcy_vp_avg,  // in
        const Float3* __restrict__ dev_darcy_grad_k,  // in
        const Float beta_f,                           // in
        const Float mu,                               // in
        const int bc_xn,                              // in
        const int bc_xp,                              // in
        const int bc_yn,                              // in
        const int bc_yp,                              // in
        const int bc_bot,                             // in
        const int bc_top,                             // in
        const unsigned int ndem,                      // in
        const unsigned int wall0_iz,                  // in
        const int* __restrict__ dev_darcy_p_constant, // in
        Float* __restrict__ dev_darcy_p_new,          // out
        Float* __restrict__ dev_darcy_norm)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    if (x < nx && y < ny && z < nz) {

        // 1D thread index
        const unsigned int cellidx = d_idx(x,y,z);

        // read values
        __syncthreads();
        const Float  phi_xn = dev_darcy_phi[d_idx(x-1,y,z)];
        const Float  phi    = dev_darcy_phi[cellidx];
        const Float  phi_xp = dev_darcy_phi[d_idx(x+1,y,z)];
        const Float  phi_yn = dev_darcy_phi[d_idx(x,y-1,z)];
        const Float  phi_yp = dev_darcy_phi[d_idx(x,y+1,z)];
        const Float  phi_zn = dev_darcy_phi[d_idx(x,y,z-1)];
        const Float  phi_zp = dev_darcy_phi[d_idx(x,y,z+1)];
        const Float  dphi   = dev_darcy_dphi[cellidx];
        const Float3 vp_avg = dev_darcy_vp_avg[cellidx];
        const int p_constant = dev_darcy_p_constant[cellidx];

        const Float p_old   = dev_darcy_p_old[cellidx];
        const Float dp_expl = dev_darcy_dp_expl[cellidx];

        Float p_xn    = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p = dev_darcy_p[cellidx];
        Float p_xp    = dev_darcy_p[d_idx(x+1,y,z)];
        Float p_yn    = dev_darcy_p[d_idx(x,y-1,z)];
        Float p_yp    = dev_darcy_p[d_idx(x,y+1,z)];
        Float p_zn    = dev_darcy_p[d_idx(x,y,z-1)];
        Float p_zp    = dev_darcy_p[d_idx(x,y,z+1)];

        const Float k_xn    = dev_darcy_k[d_idx(x-1,y,z)];
        const Float k       = dev_darcy_k[cellidx];
        const Float k_xp    = dev_darcy_k[d_idx(x+1,y,z)];
        const Float k_yn    = dev_darcy_k[d_idx(x,y-1,z)];
        const Float k_yp    = dev_darcy_k[d_idx(x,y+1,z)];
        const Float k_zn    = dev_darcy_k[d_idx(x,y,z-1)];
        const Float k_zp    = dev_darcy_k[d_idx(x,y,z+1)];

        // Neumann BCs
        if (x == 0 && bc_xn == 1)
            p_xn = p;
        if (x == nx-1 && bc_xp == 1)
            p_xp = p;
        if (y == 0 && bc_yn == 1)
            p_yn = p;
        if (y == ny-1 && bc_yp == 1)
            p_yp = p;
        if (z == 0 && bc_bot == 1)
            p_zn = p;
        if (z == nz-1 && bc_top == 1)
            p_zp = p;

        const Float3 grad_phi_central = MAKE_FLOAT3(
                (phi_xp - phi_xn)/(dx + dx),
                (phi_yp - phi_yn)/(dy + dy),
                (phi_zp - phi_zn)/(dz + dz));

        // Solve div(k*grad(p)) as a single term, using harmonic mean for 
        // permeability. k_HM*grad(p) is found between the pressure nodes.
        const Float div_k_grad_p =
                (2.*k_xp*k/(k_xp + k) *
                 (p_xp - p)/dx
                 -
                 2.*k_xn*k/(k_xn + k) *
                 (p - p_xn)/dx)/dx
            +
                (2.*k_yp*k/(k_yp + k) *
                 (p_yp - p)/dy
                 -
                 2.*k_yn*k/(k_yn + k) *
                 (p - p_yn)/dy)/dy
            +
                (2.*k_zp*k/(k_zp + k) *
                 (p_zp - p)/dz
                 -
                 2.*k_zn*k/(k_zn + k) *
                 (p - p_zn)/dz)/dz;

        //Float p_new = p_old
        Float dp_impl =
            ndem*devC_dt/(beta_f*phi*mu)*div_k_grad_p
            -(ndem*devC_dt/(beta_f*phi*(1.0 - phi)))
            *(dphi/(ndem*devC_dt) + dot(vp_avg, grad_phi_central));

        // Dirichlet BC at fixed-pressure boundaries and at the dynamic top 
        // wall.  wall0_iz will be larger than the grid if the wall isn't 
        // dynamic
        if ((bc_bot == 0 && z == 0) || (bc_top == 0 && z == nz-1)
                || (z >= wall0_iz && bc_top == 0)
                || (bc_xn == 0 && x == 0) || (bc_xp == 0 && x == nx-1)
                || (bc_yn == 0 && y == 0) || (bc_yp == 0 && y == nx-1)
                || p_constant == 1)
            dp_impl = 0.0;

        // choose integration method, parameter in [0.0; 1.0]
        //    epsilon = 0.0: explicit
        //    epsilon = 0.5: Crank-Nicolson
        //    epsilon = 1.0: implicit
        const Float epsilon = 0.5;
        Float p_new = p_old + (1.0 - epsilon)*dp_expl + epsilon*dp_impl;

        // add underrelaxation
        const Float theta = 0.05;
        p_new = p*(1.0 - theta) + p_new*theta;

        // normalized residual, avoid division by zero
        //const Float res_norm = (p_new - p)*(p_new - p)/(p_new*p_new + 1.0e-16);
        const Float res_norm = (p_new - p)/(p + 1.0e-16);

#ifdef REPORT_FORCING_TERMS_JACOBIAN
        const Float dp_diff = (ndem*devC_dt)/(beta_f*phi*mu)
            *(k*laplace_p + dot(grad_k, grad_p));
        const Float dp_forc =
            -(ndem*devC_dt/(beta_f*phi*(1.0 - phi)))
            *(dphi/(ndem*devC_dt) + dot(vp_avg, grad_phi));
        printf("\n%d,%d,%d updateDarcySolution\n"
                "p_new       = %e\n"
                "p           = %e\n"
                "p_x         = %e, %e\n"
                "p_y         = %e, %e\n"
                "p_z         = %e, %e\n"
                "dp_expl     = %e\n"
                "p_old       = %e\n"
                "div_k_grad_p= %e\n"
                "dp_diff     = %e\n"
                "dp_forc     = %e\n"
                "div_v_p     = %e\n"
                "res_norm    = %e\n"
                ,
                x,y,z,
                p_new, p,
                p_xn, p_xp,
                p_yn, p_yp,
                p_zn, p_zp,
                dp_expl,
                p_old,
                div_k_grad_p,
                dp_diff,
                dp_forc,
                dphi, dphi/(ndem*devC_dt),
                res_norm); //
#endif

        // save new pressure and the residual
        __syncthreads();
        dev_darcy_p_new[cellidx] = p_new;
        dev_darcy_norm[cellidx]  = res_norm;

        /*printf("%d,%d,%d\tp = % f\tp_new = % f\tres_norm = % f\n",
                x,y,z,
                p,
                p_new,
                res_norm);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("p_new", x, y, z, p_new);
        checkFiniteFloat("res_norm", x, y, z, res_norm);
#endif
    }
}

__global__ void findNewPressure(
        const Float* __restrict__ dev_darcy_dp,     // in
        Float* __restrict__ dev_darcy_p)            // in+out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const unsigned int cellidx = d_idx(x,y,z);

        const Float dp = dev_darcy_dp[cellidx];

        // save new pressure
        __syncthreads();
        dev_darcy_p[cellidx] += dp;

        /*printf("%d,%d,%d\tp = % f\tp_new = % f\tres_norm = % f\n",
                x,y,z,
                p,
                p_new,
                res_norm);*/

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("dp", x, y, z, dp);
#endif
    }
}

// Find cell velocities
__global__ void findDarcyVelocities(
        const Float* __restrict__ dev_darcy_p,      // in
        const Float* __restrict__ dev_darcy_phi,    // in
        const Float* __restrict__ dev_darcy_k,      // in
        const Float mu,                             // in
        Float3* __restrict__ dev_darcy_v)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell size
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const unsigned int cellidx = d_idx(x,y,z);

        __syncthreads();
        const Float p_xn = dev_darcy_p[d_idx(x-1,y,z)];
        const Float p_xp = dev_darcy_p[d_idx(x+1,y,z)];
        const Float p_yn = dev_darcy_p[d_idx(x,y-1,z)];
        const Float p_yp = dev_darcy_p[d_idx(x,y+1,z)];
        const Float p_zn = dev_darcy_p[d_idx(x,y,z-1)];
        const Float p_zp = dev_darcy_p[d_idx(x,y,z+1)];

        const Float k   = dev_darcy_k[cellidx];
        const Float phi = dev_darcy_phi[cellidx];

        // approximate pressure gradient with first order central differences
        const Float3 grad_p = MAKE_FLOAT3(
                (p_xp - p_xn)/(dx + dx),
                (p_yp - p_yn)/(dy + dy),
                (p_zp - p_zn)/(dz + dz));

        // Flux [m/s]: q = -k/nu * dH
        // Pore velocity [m/s]: v = q/n

        // calculate flux
        //const Float3 q = -k/mu*grad_p;

        // calculate velocity
        //const Float3 v = q/phi;
        const Float3 v = (-k/mu * grad_p)/phi;

        // Save velocity
        __syncthreads();
        dev_darcy_v[cellidx] = v;
    }
}

// Print final heads and free memory
void DEM::endDarcyDev()
{
    freeDarcyMemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
