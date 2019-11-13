// navierstokes.cuh
// CUDA implementation of porous flow

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

// Arithmetic mean of two numbers
__inline__ __device__ Float amean(Float a, Float b) {
    return (a+b)*0.5;
}

// Harmonic mean of two numbers
__inline__ __device__ Float hmean(Float a, Float b) {
    return (2.0*a*b)/(a+b);
}

// Helper functions for checking whether a value is NaN or Inf
__device__ int checkFiniteFloat(
    const char* desc,
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const Float s)
{
    __syncthreads();
    if (!isfinite(s)) {
        printf("\n[%d,%d,%d]: Error: %s = %f\n", x, y, z, desc, s);
        return 1;
    }
    return 0;
}

__device__ int checkFiniteFloat3(
    const char* desc,
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const Float3 v)
{
    __syncthreads();
    if (!isfinite(v.x) || !isfinite(v.y)  || !isfinite(v.z)) {
        printf("\n[%d,%d,%d]: Error: %s = %f, %f, %f\n",
               x, y, z, desc, v.x, v.y, v.z);
        return 1;
    }
    return 0;
}

// Initialize memory
void DEM::initNSmemDev(void)
{
    // size of scalar field
    unsigned int memSizeF = sizeof(Float)*NScells();

    // size of cell-face arrays in staggered grid discretization
    unsigned int memSizeFface = sizeof(Float)*NScellsVelocity();

    cudaMalloc((void**)&dev_ns_p, memSizeF);     // hydraulic pressure
    cudaMalloc((void**)&dev_ns_v, memSizeF*3);   // cell hydraulic velocity
    cudaMalloc((void**)&dev_ns_v_x, memSizeFface);// velocity in stag. grid
    cudaMalloc((void**)&dev_ns_v_y, memSizeFface);// velocity in stag. grid
    cudaMalloc((void**)&dev_ns_v_z, memSizeFface);// velocity in stag. grid
    cudaMalloc((void**)&dev_ns_v_p, memSizeF*3); // predicted cell velocity
    cudaMalloc((void**)&dev_ns_v_p_x, memSizeFface); // pred. vel. in stag. grid
    cudaMalloc((void**)&dev_ns_v_p_y, memSizeFface); // pred. vel. in stag. grid
    cudaMalloc((void**)&dev_ns_v_p_z, memSizeFface); // pred. vel. in stag. grid
    cudaMalloc((void**)&dev_ns_vp_avg, memSizeF*3); // avg. particle velocity
    cudaMalloc((void**)&dev_ns_d_avg, memSizeF); // avg. particle diameter
    cudaMalloc((void**)&dev_ns_F_pf, memSizeF*3);  // interaction force
    //cudaMalloc((void**)&dev_ns_F_pf_x, memSizeFface);  // interaction force
    //cudaMalloc((void**)&dev_ns_F_pf_y, memSizeFface);  // interaction force
    //cudaMalloc((void**)&dev_ns_F_pf_z, memSizeFface);  // interaction force
    cudaMalloc((void**)&dev_ns_phi, memSizeF);   // cell porosity
    cudaMalloc((void**)&dev_ns_dphi, memSizeF);  // cell porosity change
    //cudaMalloc((void**)&dev_ns_div_phi_v_v, memSizeF*3); // div(phi v v)
    cudaMalloc((void**)&dev_ns_epsilon, memSizeF); // pressure difference
    cudaMalloc((void**)&dev_ns_epsilon_new, memSizeF); // new pressure diff.
    cudaMalloc((void**)&dev_ns_epsilon_old, memSizeF); // old pressure diff.
    cudaMalloc((void**)&dev_ns_norm, memSizeF);  // normalized residual
    cudaMalloc((void**)&dev_ns_f, memSizeF);     // forcing function value
    cudaMalloc((void**)&dev_ns_f1, memSizeF);    // constant addition in forcing
    cudaMalloc((void**)&dev_ns_f2, memSizeF*3);  // constant slope in forcing
    //cudaMalloc((void**)&dev_ns_tau, memSizeF*6); // stress tensor
    //cudaMalloc((void**)&dev_ns_div_tau, memSizeF*3); // div(tau), cell center
    cudaMalloc((void**)&dev_ns_div_tau_x, memSizeFface); // div(tau), cell face
    cudaMalloc((void**)&dev_ns_div_tau_y, memSizeFface); // div(tau), cell face
    cudaMalloc((void**)&dev_ns_div_tau_z, memSizeFface); // div(tau), cell face
    cudaMalloc((void**)&dev_ns_div_phi_vi_v, memSizeF*3); // div(phi*vi*v)
    //cudaMalloc((void**)&dev_ns_div_phi_tau, memSizeF*3);  // div(phi*tau)
    cudaMalloc((void**)&dev_ns_f_pf, sizeof(Float3)*np); // particle fluid force
    cudaMalloc((void**)&dev_ns_f_d, sizeof(Float4)*np); // drag force
    cudaMalloc((void**)&dev_ns_f_p, sizeof(Float4)*np); // pressure force
    cudaMalloc((void**)&dev_ns_f_v, sizeof(Float4)*np); // viscous force
    cudaMalloc((void**)&dev_ns_f_sum, sizeof(Float4)*np); // sum of int. forces

    checkForCudaErrors("End of initNSmemDev");
}

// Free memory
void DEM::freeNSmemDev()
{
    cudaFree(dev_ns_p);
    cudaFree(dev_ns_v);
    cudaFree(dev_ns_v_x);
    cudaFree(dev_ns_v_y);
    cudaFree(dev_ns_v_z);
    cudaFree(dev_ns_v_p);
    cudaFree(dev_ns_v_p_x);
    cudaFree(dev_ns_v_p_y);
    cudaFree(dev_ns_v_p_z);
    cudaFree(dev_ns_vp_avg);
    cudaFree(dev_ns_d_avg);
    cudaFree(dev_ns_F_pf);
    //cudaFree(dev_ns_F_pf_x);
    //cudaFree(dev_ns_F_pf_y);
    //cudaFree(dev_ns_F_pf_z);
    cudaFree(dev_ns_phi);
    cudaFree(dev_ns_dphi);
    //cudaFree(dev_ns_div_phi_v_v);
    cudaFree(dev_ns_epsilon);
    cudaFree(dev_ns_epsilon_new);
    cudaFree(dev_ns_epsilon_old);
    cudaFree(dev_ns_norm);
    cudaFree(dev_ns_f);
    cudaFree(dev_ns_f1);
    cudaFree(dev_ns_f2);
    //cudaFree(dev_ns_tau);
    cudaFree(dev_ns_div_phi_vi_v);
    //cudaFree(dev_ns_div_phi_tau);
    //cudaFree(dev_ns_div_tau);
    cudaFree(dev_ns_div_tau_x);
    cudaFree(dev_ns_div_tau_y);
    cudaFree(dev_ns_div_tau_z);
    cudaFree(dev_ns_f_pf);
    cudaFree(dev_ns_f_d);
    cudaFree(dev_ns_f_p);
    cudaFree(dev_ns_f_v);
    cudaFree(dev_ns_f_sum);
}

// Transfer to device
void DEM::transferNStoGlobalDeviceMemory(int statusmsg)
{
    checkForCudaErrors("Before attempting cudaMemcpy in "
                       "transferNStoGlobalDeviceMemory");

    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "  Transfering fluid data to the device:           ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    //writeNSarray(ns.p, "ns.p.txt");

    cudaMemcpy(dev_ns_p, ns.p, memSizeF, cudaMemcpyHostToDevice);
    checkForCudaErrors("transferNStoGlobalDeviceMemory after first cudaMemcpy");
    cudaMemcpy(dev_ns_v, ns.v, memSizeF*3, cudaMemcpyHostToDevice);
    //cudaMemcpy(dev_ns_v_p, ns.v_p, memSizeF*3, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_phi, ns.phi, memSizeF, cudaMemcpyHostToDevice);
    cudaMemcpy(dev_ns_dphi, ns.dphi, memSizeF, cudaMemcpyHostToDevice);

    checkForCudaErrors("End of transferNStoGlobalDeviceMemory");
    //if (verbose == 1 && statusmsg == 1)
    //std::cout << "Done" << std::endl;
}

// Transfer from device
void DEM::transferNSfromGlobalDeviceMemory(int statusmsg)
{
    if (verbose == 1 && statusmsg == 1)
        std::cout << "  Transfering fluid data from the device:         ";

    // memory size for a scalar field
    unsigned int memSizeF  = sizeof(Float)*NScells();

    cudaMemcpy(ns.p, dev_ns_p, memSizeF, cudaMemcpyDeviceToHost);
    checkForCudaErrors("In transferNSfromGlobalDeviceMemory, dev_ns_p", 0);
    cudaMemcpy(ns.v, dev_ns_v, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.v_x, dev_ns_v_x, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.v_y, dev_ns_v_y, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.v_z, dev_ns_v_z, memSizeF, cudaMemcpyDeviceToHost);
    //cudaMemcpy(ns.v_p, dev_ns_v_p, memSizeF*3, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.phi, dev_ns_phi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.dphi, dev_ns_dphi, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.norm, dev_ns_norm, memSizeF, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.f_d, dev_ns_f_d, sizeof(Float4)*np, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.f_p, dev_ns_f_p, sizeof(Float4)*np, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.f_v, dev_ns_f_v, sizeof(Float4)*np, cudaMemcpyDeviceToHost);
    cudaMemcpy(ns.f_sum, dev_ns_f_sum, sizeof(Float4)*np,
            cudaMemcpyDeviceToHost);

    checkForCudaErrors("End of transferNSfromGlobalDeviceMemory", 0);
    if (verbose == 1 && statusmsg == 1)
        std::cout << "Done" << std::endl;
}

// Transfer the normalized residuals from device to host
void DEM::transferNSnormFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.norm, dev_ns_norm, sizeof(Float)*NScells(),
               cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSnormFromGlobalDeviceMemory");
}

// Transfer the pressure change from device to host
void DEM::transferNSepsilonFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.epsilon, dev_ns_epsilon, sizeof(Float)*NScells(),
               cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSepsilonFromGlobalDeviceMemory");
}

// Transfer the pressure change from device to host
void DEM::transferNSepsilonNewFromGlobalDeviceMemory()
{
    cudaMemcpy(ns.epsilon_new, dev_ns_epsilon_new, sizeof(Float)*NScells(),
               cudaMemcpyDeviceToHost);
    checkForCudaErrors("End of transferNSepsilonFromGlobalDeviceMemory");
}

// Get linear index from 3D grid position
__inline__ __device__ unsigned int idx(
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
__inline__ __device__ unsigned int vidx(
    const int x, const int y, const int z)
{
    // without ghost nodes
    //return x + (devC_grid.num[0]+1)*y
    //+ (devC_grid.num[0]+1)*(devC_grid.num[1]+1)*z;

    // with ghost nodes
    // the ghost nodes are placed at x,y,z = -1 and WIDTH+1
    return (x+1) + (devC_grid.num[0]+3)*(y+1)
        + (devC_grid.num[0]+3)*(devC_grid.num[1]+3)*(z+1);
}

// Find averaged cell velocities from cell-face velocities. This function works
// for both normal and predicted velocities. Launch for every cell in the
// dev_ns_v or dev_ns_v_p array. This function does not set the averaged
// velocity values in the ghost node cells.
__global__ void findNSavgVel(
    Float3* __restrict__ dev_ns_v,    // out
    const Float* __restrict__ dev_ns_v_x,  // in
    const Float* __restrict__ dev_ns_v_y,  // in
    const Float* __restrict__ dev_ns_v_z)  // in
{

    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x<devC_grid.num[0] && y<devC_grid.num[1] && z<devC_grid.num[2]-1) {
        const unsigned int cellidx = idx(x,y,z);

        // Read cell-face velocities
        __syncthreads();
        const Float v_xn = dev_ns_v_x[vidx(x,y,z)];
        const Float v_xp = dev_ns_v_x[vidx(x+1,y,z)];
        const Float v_yn = dev_ns_v_y[vidx(x,y,z)];
        const Float v_yp = dev_ns_v_y[vidx(x,y+1,z)];
        const Float v_zn = dev_ns_v_z[vidx(x,y,z)];
        const Float v_zp = dev_ns_v_z[vidx(x,y,z+1)];

        // Find average velocity using arithmetic means
        const Float3 v_bar = MAKE_FLOAT3(
            amean(v_xn, v_xp),
            amean(v_yn, v_yp),
            amean(v_zn, v_zp));

        // Save value
        __syncthreads();
        dev_ns_v[idx(x,y,z)] = v_bar;
    }
}

// Find cell-face velocities from averaged velocities. This function works for
// both normal and predicted velocities. Launch for every cell in the dev_ns_v
// or dev_ns_v_p array. Make sure that the averaged velocity ghost nodes are set
// beforehand.
__global__ void findNScellFaceVel(
    const Float3* __restrict__ dev_ns_v,    // in
    Float* __restrict__ dev_ns_v_x,  // out
    Float* __restrict__ dev_ns_v_y,  // out
    Float* __restrict__ dev_ns_v_z)  // out
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
    if (x < nx && y < ny && x < nz) {
        const unsigned int cellidx = idx(x,y,z);

        // Read the averaged velocity from this cell as well as the required
        // components from the neighbor cells
        __syncthreads();
        const Float3 v = dev_ns_v[idx(x,y,z)];
        const Float v_xn = dev_ns_v[idx(x-1,y,z)].x;
        const Float v_xp = dev_ns_v[idx(x+1,y,z)].x;
        const Float v_yn = dev_ns_v[idx(x,y-1,z)].y;
        const Float v_yp = dev_ns_v[idx(x,y+1,z)].y;
        const Float v_zn = dev_ns_v[idx(x,y,z-1)].z;
        const Float v_zp = dev_ns_v[idx(x,y,z+1)].z;

        // Find cell-face velocities and save them right away
        __syncthreads();

        // Values at the faces closest to the coordinate system origo
        dev_ns_v_x[vidx(x,y,z)] = amean(v_xn, v.x);
        dev_ns_v_y[vidx(x,y,z)] = amean(v_yn, v.y);
        dev_ns_v_z[vidx(x,y,z)] = amean(v_zn, v.z);

        // Values at the cell faces furthest from the coordinate system origo.
        // These values should only be written at the corresponding boundaries
        // in order to avoid write conflicts.
        if (x == nx-1)
            dev_ns_v_x[vidx(x+1,y,z)] = amean(v.x, v_xp);
        if (y == ny-1)
            dev_ns_v_x[vidx(x+1,y,z)] = amean(v.y, v_yp);
        if (z == nz-1)
            dev_ns_v_x[vidx(x+1,y,z)] = amean(v.z, v_zp);
    }
}


// Set the initial guess of the values of epsilon.
__global__ void setNSepsilonInterior(
    Float* __restrict__ dev_ns_epsilon,
    const Float value)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] &&
        z > 0 && z < devC_grid.num[2]-1) {
        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_epsilon[cellidx] = value;
    }
}

// The normalized residuals are given an initial value of 0, since the values at
// the Dirichlet boundaries aren't written during the iterations.
__global__ void setNSnormZero(Float* __restrict__ dev_ns_norm)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_norm[idx(x,y,z)]    = 0.0;
    }
}


// Set the constant values of epsilon at the lower boundary.  Since the
// Dirichlet boundary values aren't transfered during array swapping, the values
// also need to be written to the new array of epsilons.  A value of 0 equals
// the Dirichlet boundary condition: the new value should be identical to the
// old value, i.e. the temporal gradient is 0
__global__ void setNSepsilonBottom(
    Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_epsilon_new,
    const Float value)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid, and at the z boundaries
    //if (x < devC_grid.num[0] && y < devC_grid.num[1] &&
    //        (z == devC_grid.num[2]-1 || z == 0)) {
    // check that we are not outside the fluid grid, and at the lower z boundary
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z == 0) {

        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_epsilon[cellidx]     = value;
        dev_ns_epsilon_new[cellidx] = value;
    }
}

// Set the constant values of epsilon at the upper boundary.  Since the
// Dirichlet boundary values aren't transfered during array swapping, the values
// also need to be written to the new array of epsilons.
__global__ void setNSepsilonTop(
    Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_epsilon_new,
    const Float value)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid, and at the upper z boundary
    if (x < devC_grid.num[0] && y < devC_grid.num[1] &&
        z == devC_grid.num[2]-1) {

        __syncthreads();
        const unsigned int cellidx = idx(x,y,z);
        dev_ns_epsilon[cellidx]     = value;
        dev_ns_epsilon_new[cellidx] = value;
    }
}

// Set the constant values of epsilon and grad_z(epsilon) at the upper wall, if
// it is dynamic (Dirichlet+Neumann). Since the Dirichlet boundary values aren't
// transfered during array swapping, the values also need to be written to the
// new array of epsilons.
__global__ void setNSepsilonAtTopWall(
    Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_epsilon_new,
    const unsigned int iz,
    const Float value,
    const Float dp_dz)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    const unsigned int cellidx = idx(x,y,z);

    // cells containing the wall (Dirichlet BC)
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2] &&
            z == iz) {
        __syncthreads();
        dev_ns_epsilon[cellidx]     = value;
        dev_ns_epsilon_new[cellidx] = value;
    }

    // cells above the wall (Neumann BC)
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2] &&
            z == iz+1) {

        // Pressure value in order to obtain hydrostatic pressure distribution
        // for Neumann BC. The pressure should equal the value at the top wall
        // minus the contribution due to the fluid weight.
        // p_iz+1 = p_iz - rho_f*g*dz
        const Float p = value - dp_dz;

        __syncthreads();
        dev_ns_epsilon[cellidx]     = p;
        dev_ns_epsilon_new[cellidx] = p;
    }
}

__device__ void copyNSvalsDev(
    const unsigned int read,
    const unsigned int write,
    Float*  __restrict__ dev_ns_p,
    Float3* __restrict__ dev_ns_v,
    Float3* __restrict__ dev_ns_v_p,
    Float*  __restrict__ dev_ns_phi,
    Float*  __restrict__ dev_ns_dphi,
    Float*  __restrict__ dev_ns_epsilon)
{
    // Coalesced read
    const Float  p       = dev_ns_p[read];
    const Float3 v       = dev_ns_v[read];
    const Float3 v_p     = dev_ns_v_p[read];
    const Float  phi     = dev_ns_phi[read];
    const Float  dphi    = dev_ns_dphi[read];
    const Float  epsilon = dev_ns_epsilon[read];

    // Coalesced write
    __syncthreads();
    dev_ns_p[write]       = p;
    dev_ns_v[write]       = v;
    dev_ns_v_p[write]     = v_p;
    dev_ns_phi[write]     = phi;
    dev_ns_dphi[write]    = dphi;
    dev_ns_epsilon[write] = epsilon;
}


// Update ghost nodes from their parent cell values. The edge (diagonal) cells
// are not written since they are not read. Launch this kernel for all cells in
// the grid
__global__ void setNSghostNodesDev(
    Float*  __restrict__ dev_ns_p,
    Float3* __restrict__ dev_ns_v,
    Float3* __restrict__ dev_ns_v_p,
    Float*  __restrict__ dev_ns_phi,
    Float*  __restrict__ dev_ns_dphi,
    Float*  __restrict__ dev_ns_epsilon)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // 1D position of ghost node
    unsigned int writeidx;

    // check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        if (x == 0) {
            writeidx = idx(nx,y,z);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }
        if (x == nx-1) {
            writeidx = idx(-1,y,z);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }

        if (y == 0) {
            writeidx = idx(x,ny,z);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }
        if (y == ny-1) {
            writeidx = idx(x,-1,z);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }

        // Z boundaries fixed
        if (z == 0) {
            writeidx = idx(x,y,-1);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }
        if (z == nz-1) {
            writeidx = idx(x,y,nz);
            copyNSvalsDev(cellidx, writeidx,
                          dev_ns_p,
                          dev_ns_v, dev_ns_v_p,
                          dev_ns_phi, dev_ns_dphi,
                          dev_ns_epsilon);
        }

        // Z boundaries periodic
        /*if (z == 0) {
          writeidx = idx(x,y,nz);
          copyNSvalsDev(cellidx, writeidx,
          dev_ns_p,
          dev_ns_v, dev_ns_v_p,
          dev_ns_phi, dev_ns_dphi,
          dev_ns_epsilon);
          }
          if (z == nz-1) {
          writeidx = idx(x,y,-1);
          copyNSvalsDev(cellidx, writeidx,
          dev_ns_p,
          dev_ns_v, dev_ns_v_p,
          dev_ns_phi, dev_ns_dphi,
          dev_ns_epsilon);
          }*/
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read. Launch this kernel
// for all cells in the grid usind setNSghostNodes<datatype><<<.. , ..>>>( .. );
template<typename T>
__global__ void setNSghostNodes(T* __restrict__ dev_scalarfield)
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

        const T val = dev_scalarfield[idx(x,y,z)];

        if (x == 0)
            dev_scalarfield[idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[idx(-1,y,z)] = val;

        if (y == 0)
            dev_scalarfield[idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[idx(x,-1,z)] = val;

        if (z == 0)
            dev_scalarfield[idx(x,y,-1)] = val;     // Dirichlet
        //dev_scalarfield[idx(x,y,nz)] = val;    // Periodic -z
        if (z == nz-1)
            dev_scalarfield[idx(x,y,nz)] = val;     // Dirichlet
        //dev_scalarfield[idx(x,y,-1)] = val;    // Periodic +z
    }
}

// Update a field in the ghost nodes from their parent cell values. The edge
// (diagonal) cells are not written since they are not read.
template<typename T>
__global__ void setNSghostNodes(
    T* __restrict__ dev_scalarfield,
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

        const T val = dev_scalarfield[idx(x,y,z)];

        // x
        if (x == 0)
            dev_scalarfield[idx(nx,y,z)] = val;
        if (x == nx-1)
            dev_scalarfield[idx(-1,y,z)] = val;

        // y
        if (y == 0)
            dev_scalarfield[idx(x,ny,z)] = val;
        if (y == ny-1)
            dev_scalarfield[idx(x,-1,z)] = val;

        // z
        if (z == 0 && bc_bot == 0)
            dev_scalarfield[idx(x,y,-1)] = val;     // Dirichlet
        //if (z == 1 && bc_bot == 1)
        if (z == 0 && bc_bot == 1)
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
// (diagonal) cells are not written since they are not read.
// Launch per face.
// According to Griebel et al. 1998 "Numerical Simulation in Fluid Dynamics"
template<typename T>
__global__ void setNSghostNodesFace(
    T* __restrict__ dev_scalarfield_x,
    T* __restrict__ dev_scalarfield_y,
    T* __restrict__ dev_scalarfield_z,
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
    //if (x <= nx && y <= ny && z <= nz) {
    if (x < nx && y < ny && z < nz) {

        const T val_x = dev_scalarfield_x[vidx(x,y,z)];
        const T val_y = dev_scalarfield_y[vidx(x,y,z)];
        const T val_z = dev_scalarfield_z[vidx(x,y,z)];

        // x (periodic)
        if (x == 0) {
            dev_scalarfield_x[vidx(nx,y,z)] = val_x;
            dev_scalarfield_y[vidx(nx,y,z)] = val_y;
            dev_scalarfield_z[vidx(nx,y,z)] = val_z;
        }
        if (x == 1) {
            dev_scalarfield_x[vidx(nx+1,y,z)] = val_x;
        }
        if (x == nx-1) {
            dev_scalarfield_x[vidx(-1,y,z)] = val_x;
            dev_scalarfield_y[vidx(-1,y,z)] = val_y;
            dev_scalarfield_z[vidx(-1,y,z)] = val_z;
        }

        // z ghost nodes at x = -1 and z = nz,
        // equal to the ghost node at x = nx-1 and z = nz
        if (z == nz-1 && x == nx-1 && bc_top == 0) // Dirichlet +z
            dev_scalarfield_z[vidx(-1,y,nz)] = val_z;

        if (z == nz-1 && x == nx-1 && (bc_top == 1 || bc_top == 2)) //Neumann +z
            dev_scalarfield_z[vidx(-1,y,nz)] = 0.0;

        if (z == 0 && x == nx-1 && bc_top == 3) // Periodic +z
            dev_scalarfield_z[vidx(-1,y,nz)] = val_z;

        // z ghost nodes at y = -1 and z = nz,
        // equal to the ghost node at y = ny-1 and z = nz
        if (z == nz-1 && y == ny-1 && bc_top == 0) // Dirichlet +z
            dev_scalarfield_z[vidx(x,-1,nz)] = val_z;

        if (z == nz-1 && y == ny-1 && (bc_top == 1 || bc_top == 2)) //Neumann +z
            dev_scalarfield_z[vidx(x,-1,nz)] = 0.0;

        if (z == 0 && y == ny-1 && bc_top == 3) // Periodic +z
            dev_scalarfield_z[vidx(x,-1,nz)] = val_z;


        // x ghost nodes at x = nx and z = -1,
        // equal to the ghost nodes at x = 0 and z = -1
        // Dirichlet, Neumann free slip or periodic -z
        if (z == 0 && x == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_x[vidx(nx,y,-1)] = val_x;

        if (z == 0 && x == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_x[vidx(nx,y,-1)] = -val_x;

        // y ghost nodes at y = ny and z = -1,
        // equal to the ghost node at x = 0 and z = -1
        // Dirichlet, Neumann free slip or periodic -z
        if (z == 0 && y == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_y[vidx(x,ny,-1)] = val_y;

        if (z == 0 && y == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_y[vidx(x,ny,-1)] = -val_y;


        // z ghost nodes at x = nx and z = nz
        // equal to the ghost node at x = 0 and z = nz
        if (z == nz-1 && x == 0 && (bc_top == 0 || bc_top == 3)) // D. or p. +z
            dev_scalarfield_z[vidx(nx,y,nz)] = val_z;

        if (z == nz-1 && x == 0 && (bc_top == 1 || bc_top == 2)) // N. +z
            dev_scalarfield_z[vidx(nx,y,nz)] = 0.0;

        // z ghost nodes at y = ny and z = nz
        // equal to the ghost node at y = 0 and z = nz
        if (z == nz-1 && y == 0 && (bc_top == 0 || bc_top == 3)) // D. or p. +z
            dev_scalarfield_z[vidx(x,ny,nz)] = val_z;

        if (z == nz-1 && y == 0 && (bc_top == 1 || bc_top == 2)) // N. +z
            dev_scalarfield_z[vidx(x,ny,nz)] = 0.0;


        // x ghost nodes at x = nx and z = nz,
        // equal to the ghost nodes at x = 0 and z = nz
        // Dirichlet, Neumann free slip or periodic +z
        if (z == nz-1 && x == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_x[vidx(nx,y,nz)] = val_x;

        if (z == nz-1 && x == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_x[vidx(nx,y,nz)] = -val_x;

        // y ghost nodes at y = ny and z = nz,
        // equal to the ghost nodes at y = 0 and z = nz
        // Dirichlet, Neumann free slip or periodic +z
        if (z == nz-1 && y == 0 && (bc_bot == 0 || bc_bot == 1 || bc_bot == 3))
            dev_scalarfield_y[vidx(x,ny,nz)] = val_y;

        if (z == nz-1 && y == 0 && bc_bot == 2) // Neumann no slip -z
            dev_scalarfield_y[vidx(x,ny,nz)] = -val_y;


        // y (periodic)
        if (y == 0) {
            dev_scalarfield_x[vidx(x,ny,z)] = val_x;
            dev_scalarfield_y[vidx(x,ny,z)] = val_y;
            dev_scalarfield_z[vidx(x,ny,z)] = val_z;
        }
        if (y == 1) {
            dev_scalarfield_y[vidx(x,ny+1,z)] = val_y;
        }
        if (y == ny-1) {
            dev_scalarfield_x[vidx(x,-1,z)] = val_x;
            dev_scalarfield_y[vidx(x,-1,z)] = val_y;
            dev_scalarfield_z[vidx(x,-1,z)] = val_z;
        }

        // z
        if (z == 0 && bc_bot == 0) {
            dev_scalarfield_x[vidx(x,y,-1)] = val_y;     // Dirichlet -z
            dev_scalarfield_y[vidx(x,y,-1)] = val_x;     // Dirichlet -z
            dev_scalarfield_z[vidx(x,y,-1)] = val_z;     // Dirichlet -z
        }
        if (z == 0 && bc_bot == 1) {
            //dev_scalarfield_x[vidx(x,y,-1)] = val_x;   // Neumann free slip -z
            //dev_scalarfield_y[vidx(x,y,-1)] = val_y;   // Neumann free slip -z
            //dev_scalarfield_z[vidx(x,y,-1)] = val_z;   // Neumann free slip -z
            dev_scalarfield_x[vidx(x,y,-1)] = val_x;     // Neumann free slip -z
            dev_scalarfield_y[vidx(x,y,-1)] = val_y;     // Neumann free slip -z
            dev_scalarfield_z[vidx(x,y,-1)] = 0.0;       // Neumann free slip -z
        }
        if (z == 0 && bc_bot == 2) {
            //dev_scalarfield_x[vidx(x,y,-1)] = val_x;     // Neumann no slip -z
            //dev_scalarfield_y[vidx(x,y,-1)] = val_y;     // Neumann no slip -z
            //dev_scalarfield_z[vidx(x,y,-1)] = val_z;     // Neumann no slip -z
            dev_scalarfield_x[vidx(x,y,-1)] = -val_x;    // Neumann no slip -z
            dev_scalarfield_y[vidx(x,y,-1)] = -val_y;    // Neumann no slip -z
            dev_scalarfield_z[vidx(x,y,-1)] = 0.0;       // Neumann no slip -z
        }
        if (z == 0 && bc_bot == 3) {
            dev_scalarfield_x[vidx(x,y,nz)] = val_x;     // Periodic -z
            dev_scalarfield_y[vidx(x,y,nz)] = val_y;     // Periodic -z
            dev_scalarfield_z[vidx(x,y,nz)] = val_z;     // Periodic -z
        }
        if (z == 1 && bc_bot == 3) {
            dev_scalarfield_z[vidx(x,y,nz+1)] = val_z;   // Periodic -z
        }

        if (z == nz-1 && bc_top == 0) {
            dev_scalarfield_z[vidx(x,y,nz)] = val_z;     // Dirichlet +z
        }
        if (z == nz-1 && bc_top == 1) {
            //dev_scalarfield_x[vidx(x,y,nz)] = val_x;   // Neumann free slip +z
            //dev_scalarfield_y[vidx(x,y,nz)] = val_y;   // Neumann free slip +z
            //dev_scalarfield_z[vidx(x,y,nz)] = val_z;   // Neumann free slip +z
            //dev_scalarfield_z[vidx(x,y,nz+1)] = val_z; // Neumann free slip +z
            dev_scalarfield_x[vidx(x,y,nz)] = val_x;     // Neumann free slip +z
            dev_scalarfield_y[vidx(x,y,nz)] = val_y;     // Neumann free slip +z
            dev_scalarfield_z[vidx(x,y,nz)] = 0.0;     // Neumann free slip +z
            dev_scalarfield_z[vidx(x,y,nz+1)] = 0.0;   // Neumann free slip +z
        }
        if (z == nz-1 && bc_top == 2) {
            //dev_scalarfield_x[vidx(x,y,nz)] = val_x;     // Neumann no slip +z
            //dev_scalarfield_y[vidx(x,y,nz)] = val_y;     // Neumann no slip +z
            //dev_scalarfield_z[vidx(x,y,nz)] = val_z;     // Neumann no slip +z
            //dev_scalarfield_z[vidx(x,y,nz+1)] = val_z;   // Neumann no slip +z
            dev_scalarfield_x[vidx(x,y,nz)] = -val_x;    // Neumann no slip +z
            dev_scalarfield_y[vidx(x,y,nz)] = -val_y;    // Neumann no slip +z
            dev_scalarfield_z[vidx(x,y,nz)] = 0.0;       // Neumann no slip +z
            dev_scalarfield_z[vidx(x,y,nz+1)] = 0.0;     // Neumann no slip +z
        }
        if (z == nz-1 && bc_top == 3) {
            dev_scalarfield_x[vidx(x,y,-1)] = val_x;     // Periodic +z
            dev_scalarfield_y[vidx(x,y,-1)] = val_y;     // Periodic +z
            dev_scalarfield_z[vidx(x,y,-1)] = val_z;     // Periodic +z
        }
    }
}

// Update the tensor field for the ghost nodes from their parent cell values.
// The edge (diagonal) cells are not written since they are not read. Launch
// this kernel for all cells in the grid.
__global__ void setNSghostNodes_tau(
    Float* __restrict__ dev_ns_tau,
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

        // Linear index of length-6 vector field entry
        unsigned int cellidx6 = idx(x,y,z)*6;

        // Read parent values
        __syncthreads();
        const Float tau_xx = dev_ns_tau[cellidx6];
        const Float tau_xy = dev_ns_tau[cellidx6+1];
        const Float tau_xz = dev_ns_tau[cellidx6+2];
        const Float tau_yy = dev_ns_tau[cellidx6+3];
        const Float tau_yz = dev_ns_tau[cellidx6+4];
        const Float tau_zz = dev_ns_tau[cellidx6+5];

        // x
        if (x == 0) {
            cellidx6 = idx(nx,y,z)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (x == nx-1) {
            cellidx6 = idx(-1,y,z)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }

        // y
        if (y == 0) {
            cellidx6 = idx(x,ny,z)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (y == ny-1) {
            cellidx6 = idx(x,-1,z)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }

        // z
        if (z == 0 && bc_bot == 0) {  // Dirichlet
            cellidx6 = idx(x,y,-1)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (z == 1 && bc_bot == 1) {  // Neumann
            cellidx6 = idx(x,y,-1)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (z == 0 && bc_bot == 2) {  // Periodic
            cellidx6 = idx(x,y,nz)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }

        if (z == nz-1 && bc_top == 0) {  // Dirichlet
            cellidx6 = idx(x,y,nz)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (z == nz-2 && bc_top == 1) {  // Neumann
            cellidx6 = idx(x,y,nz)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
        if (z == nz-1 && bc_top == 2) {  // Periodic
            cellidx6 = idx(x,y,-1)*6;
            dev_ns_tau[cellidx6]   = tau_xx;
            dev_ns_tau[cellidx6+1] = tau_xy;
            dev_ns_tau[cellidx6+2] = tau_xz;
            dev_ns_tau[cellidx6+3] = tau_yy;
            dev_ns_tau[cellidx6+4] = tau_yz;
            dev_ns_tau[cellidx6+5] = tau_zz;
        }
    }
}

// Update a the forcing values in the ghost nodes from their parent cell values.
// The edge (diagonal) cells are not written since they are not read. Launch
// this kernel for all cells in the grid.
/*
  __global__ void setNSghostNodesForcing(
  Float*  dev_ns_f1,
  Float3* dev_ns_f2,
  Float*  dev_ns_f,
  unsigned int nijac)

  {
  // 3D thread index
  const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
  const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
  const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

  // Grid dimensions
  const unsigned int nx = devC_grid.num[0];
  const unsigned int ny = devC_grid.num[1];
  const unsigned int nz = devC_grid.num[2];

  // 1D thread index
  unsigned int cellidx = idx(x,y,z);

  // check that we are not outside the fluid grid
  if (x < nx && y < ny && z < nz) {

  __syncthreads();
  const Float f  = dev_ns_f[cellidx];
  Float  f1;
  Float3 f2;

  if (nijac == 0) {
  __syncthreads();
  f1 = dev_ns_f1[cellidx];
  f2 = dev_ns_f2[cellidx];
  }

  if (x == 0) {
  cellidx = idx(nx,y,z);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }
  if (x == nx-1) {
  cellidx = idx(-1,y,z);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }

  if (y == 0) {
  cellidx = idx(x,ny,z);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }
  if (y == ny-1) {
  cellidx = idx(x,-1,z);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }

  if (z == 0) {
  cellidx = idx(x,y,nz);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }
  if (z == nz-1) {
  cellidx = idx(x,y,-1);
  dev_ns_f[cellidx] = f;
  if (nijac == 0) {
  dev_ns_f1[cellidx] = f1;
  dev_ns_f2[cellidx] = f2;
  }
  }
  }
  }
*/

// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
__global__ void findPorositiesVelocitiesDiametersSpherical(
    const unsigned int* __restrict__ dev_cellStart,
    const unsigned int* __restrict__ dev_cellEnd,
    const Float4* __restrict__ dev_x_sorted,
    const Float4* __restrict__ dev_vel_sorted,
    Float*  __restrict__ dev_ns_phi,
    Float*  __restrict__ dev_ns_dphi,
    Float3* __restrict__ dev_ns_vp_avg,
    Float*  __restrict__ dev_ns_d_avg,
    const unsigned int iteration,
    const unsigned int np,
    const Float c_phi)
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
    //const Float R = fmin(dx, fmin(dy,dz)) * 0.5; // diameter = cell width
    const Float R = fmin(dx, fmin(dy,dz));       // diameter = 2*cell width
    const Float cell_volume = 4.0/3.0*M_PI*R*R*R;

    Float void_volume = cell_volume;
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
            unsigned int n = 0;

            Float3 v_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);
            Float  d_avg = 0.0;

            // Read old porosity
            __syncthreads();
            Float phi_0 = dev_ns_phi[idx(x,y,z)];

            // The cell 3d index
            const int3 gridPos = make_int3((int)x,(int)y,(int)z);

            // The neighbor cell 3d index
            int3 targetCell;

            // The distance modifier for particles across periodic boundaries
            Float3 dist, distmod;

            unsigned int cellID, startIdx, endIdx, i;

            // Iterate over 27 neighbor cells, R = cell width
            /*for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
              for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
              for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis*/

            // Iterate over 27 neighbor cells, R = 2*cell width
            for (int z_dim=-2; z_dim<3; ++z_dim) { // z-axis
                //for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-2; y_dim<3; ++y_dim) { // y-axis
                    for (int x_dim=-2; x_dim<3; ++x_dim) { // x-axis

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
                                    if ((R - r) < d && d < (R + r)) {
                                        void_volume -=
                                            1.0/(12.0*d) * (
                                                M_PI*(R + r - d)*(R + r - d)
                                                *(d*d + 2.0*d*r - 3.0*r*r
                                                  + 2.0*d*R + 6.0*r*R
                                                  - 3.0*R*R) );
                                        v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        d_avg += 2.0*r;
                                        n++;
                                    }

                                    // Particle fully contained in cell sphere
                                    if (d <= R - r) {
                                        void_volume -= 4.0/3.0*M_PI*r*r*r;
                                        v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                        d_avg += 2.0*r;
                                        n++;
                                    }
                                }
                            }
                        }
                    }
                }
            }

            if (phi < 0.999) {
                v_avg /= n;
                d_avg /= n;
            }

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(1.00, fmax(0.00, void_volume/cell_volume));
            //phi = void_volume/cell_volume;

            Float dphi = phi - phi_0;
            if (iteration == 0)
                dphi = 0.0;

            // report values to stdout for debugging
            //printf("%d,%d,%d\tphi = %f dphi = %f v_avg = %f,%f,%f d_avg = %f\n",
            //       x,y,z, phi, dphi, v_avg.x, v_avg.y, v_avg.z, d_avg);

            // Save porosity, porosity change, average velocity and average diameter
            __syncthreads();
            //phi = 0.5; dphi = 0.0; // disable porosity effects
            const unsigned int cellidx = idx(x,y,z);
            dev_ns_phi[cellidx]  = phi*c_phi;
            dev_ns_dphi[cellidx] = dphi*c_phi;
            dev_ns_vp_avg[cellidx] = v_avg;
            dev_ns_d_avg[cellidx]  = d_avg;

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
            (void)checkFiniteFloat3("v_avg", x, y, z, v_avg);
            (void)checkFiniteFloat("d_avg", x, y, z, d_avg);
#endif
        } else {

            __syncthreads();
            const unsigned int cellidx = idx(x,y,z);

            //Float phi = 0.5;
            //Float dphi = 0.0;
            //if (iteration == 20 && x == nx/2 && y == ny/2 && z == nz/2) {
            //phi = 0.4;
            //dphi = 0.1;
            //}
            //dev_ns_phi[cellidx]  = phi;
            //dev_ns_dphi[cellidx] = dphi;
            dev_ns_phi[cellidx]  = 1.0;
            dev_ns_dphi[cellidx] = 0.0;

            dev_ns_vp_avg[cellidx] = MAKE_FLOAT3(0.0, 0.0, 0.0);
            dev_ns_d_avg[cellidx]  = 0.0;
        }
    }
}

// Find the porosity in each cell on the base of a sphere, centered at the cell
// center. 
__global__ void findPorositiesVelocitiesDiametersSphericalGradient(
    const unsigned int* __restrict__ dev_cellStart,
    const unsigned int* __restrict__ dev_cellEnd,
    const Float4* __restrict__ dev_x_sorted,
    const Float4* __restrict__ dev_vel_sorted,
    Float*  __restrict__ dev_ns_phi,
    Float*  __restrict__ dev_ns_dphi,
    Float3* __restrict__ dev_ns_vp_avg,
    Float*  __restrict__ dev_ns_d_avg,
    const unsigned int iteration,
    const unsigned int ndem,
    const unsigned int np)
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
    const Float R = fmin(dx, fmin(dy,dz));  // diameter = 2*cell width

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
            unsigned int n = 0;

            Float3 v_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);
            Float  d_avg = 0.0;

            // Read old porosity
            __syncthreads();
            Float phi_0 = dev_ns_phi[idx(x,y,z)];

            // The cell 3d index
            const int3 gridPos = make_int3((int)x,(int)y,(int)z);

            // The neighbor cell 3d index
            int3 targetCell;

            // The distance modifier for particles across periodic boundaries
            Float3 distmod;

            unsigned int cellID, startIdx, endIdx, i;

            // Diagonal strain rate tensor components
            Float3 dot_epsilon_ii = MAKE_FLOAT3(0.0, 0.0, 0.0);

            // Vector pointing from cell center to particle center
            Float3 x_p;

            // Normal vector pointing from cell center towards particle center
            Float3 n_p;

            // Normalized sphere-particle distance
            Float q;

            // Kernel function derivative value
            Float dw_q;

            // Iterate over 27 neighbor cells, R = 2*cell width
            for (int z_dim=-2; z_dim<3; ++z_dim) { // z-axis
                for (int y_dim=-2; y_dim<3; ++y_dim) { // y-axis
                    for (int x_dim=-2; x_dim<3; ++x_dim) { // x-axis

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
                            startIdx = dev_cellStart[cellID];

                            // Make sure cell is not empty
                            if (startIdx != 0xffffffff) {

                                // Highest particle index in cell
                                endIdx = dev_cellEnd[cellID];

                                // Iterate over cell particles
                                for (i=startIdx; i<endIdx; ++i) {

                                    // Read particle position and radius
                                    __syncthreads();
                                    xr = dev_x_sorted[i];
                                    v  = dev_vel_sorted[i];
                                    r = xr.w;

                                    // Find center distance and normal vector
                                    x_p = MAKE_FLOAT3(
                                        xr.x - X.x,
                                        xr.y - X.y,
                                        xr.z - X.z);
                                    d = length(x_p);
                                    n_p = x_p/d;
                                    q = d/R;


                                    dw_q = 0.0;
                                    if (0.0 < q && q < 1.0) {
                                        // kernel for 2d disc approximation
                                        //dw_q = -1.0;

                                        // kernel for 3d sphere approximation
                                        dw_q = -1.5*pow(-q + 1.0, 0.5)
                                            *pow(q + 1.0, 0.5)
                                            + 0.5*pow(-q + 1.0, 1.5)
                                            *pow(q + 1.0, -0.5);
                                    }

                                    v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                                    d_avg += 2.0*r;
                                    dot_epsilon_ii +=
                                        dw_q*MAKE_FLOAT3(v.x, v.y, v.z)*n_p;
                                    n++;

                                }
                            }
                        }
                    }
                }
            }

            dot_epsilon_ii /= R;
            const Float dot_epsilon_kk =
                dot_epsilon_ii.x + dot_epsilon_ii.y + dot_epsilon_ii.z;

            const Float dphi =
                (1.0 - fmin(phi_0,0.99))*dot_epsilon_kk*ndem*devC_dt;
            phi = phi_0 + dphi/(ndem*devC_dt);

            //if (dot_epsilon_kk != 0.0)
            //printf("%d,%d,%d\tdot_epsilon_kk = %f\tdphi = %f\tphi = %f\n",
            //x,y,z, dot_epsilon_kk, dphi, phi);

            // Make sure that the porosity is in the interval [0.0;1.0]
            phi = fmin(1.00, fmax(0.00, phi));

            if (phi < 0.999) {
                v_avg /= n;
                d_avg /= n;
            }

            // report values to stdout for debugging
            //printf("%d,%d,%d\tphi = %f dphi = %f v_avg = %f,%f,%f d_avg = %f\n",
            //       x,y,z, phi, dphi, v_avg.x, v_avg.y, v_avg.z, d_avg);

            // Save porosity, porosity change, average velocity and average diameter
            __syncthreads();
            const unsigned int cellidx = idx(x,y,z);
            //phi = 0.5; dphi = 0.0; // disable porosity effects const unsigned int cellidx = idx(x,y,z);
            dev_ns_phi[cellidx]  = phi;
            dev_ns_dphi[cellidx] = dphi;
            dev_ns_vp_avg[cellidx] = v_avg;
            dev_ns_d_avg[cellidx]  = d_avg;

#ifdef CHECK_FLUID_FINITE
            (void)checkFiniteFloat("phi", x, y, z, phi);
            (void)checkFiniteFloat("dphi", x, y, z, dphi);
            (void)checkFiniteFloat3("v_avg", x, y, z, v_avg);
            (void)checkFiniteFloat("d_avg", x, y, z, d_avg);
#endif
        } else {
            // np=0: there are no particles

            __syncthreads();
            const unsigned int cellidx = idx(x,y,z);

            dev_ns_dphi[cellidx] = 0.0;

            dev_ns_vp_avg[cellidx] = MAKE_FLOAT3(0.0, 0.0, 0.0);
            dev_ns_d_avg[cellidx]  = 0.0;
        }
    }
}

// Modulate the hydraulic pressure at the upper boundary
__global__ void setUpperPressureNS(
    Float* __restrict__ dev_ns_p,
    Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_epsilon_new,
    const Float  beta,
    const Float new_pressure)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;
    
    // check that the thread is located at the top boundary
    if (x < devC_grid.num[0] &&
        y < devC_grid.num[1] &&
        z == devC_grid.num[2]-1) {

        const unsigned int cellidx = idx(x,y,z);

        // Read the current pressure
        const Float pressure = dev_ns_p[cellidx];

        // Determine the new epsilon boundary condition
        const Float epsilon = new_pressure - beta*pressure;

        // Write the new pressure and epsilon values to the top boundary cells
        __syncthreads();
        dev_ns_epsilon[cellidx] = epsilon;
        dev_ns_epsilon_new[cellidx] = epsilon;
        dev_ns_p[cellidx] = new_pressure;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("epsilon", x, y, z, epsilon);
        (void)checkFiniteFloat("new_pressure", x, y, z, new_pressure);
#endif
    }
}

// Find the gradient in a cell in a homogeneous, cubic 3D scalar field using
// finite central differences
__device__ Float3 gradient(
    const Float* __restrict__ dev_scalarfield,
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const Float dx,
    const Float dy,
    const Float dz)
{
    // Read 6 neighbor cells
    __syncthreads();
    //const Float p  = dev_scalarfield[idx(x,y,z)];
    const Float xn = dev_scalarfield[idx(x-1,y,z)];
    const Float xp = dev_scalarfield[idx(x+1,y,z)];
    const Float yn = dev_scalarfield[idx(x,y-1,z)];
    const Float yp = dev_scalarfield[idx(x,y+1,z)];
    const Float zn = dev_scalarfield[idx(x,y,z-1)];
    const Float zp = dev_scalarfield[idx(x,y,z+1)];

    //__syncthreads();
    //if (p != 0.0)
    //printf("p[%d,%d,%d] =\t%f\n", x,y,z, p);

    // Calculate central-difference gradients
    return MAKE_FLOAT3(
        (xp - xn)/(2.0*dx),
        (yp - yn)/(2.0*dy),
        (zp - zn)/(2.0*dz));
}

// Find the divergence in a cell in a homogeneous, cubic, 3D vector field
__device__ Float divergence(
    const Float* __restrict__ dev_vectorfield_x,
    const Float* __restrict__ dev_vectorfield_y,
    const Float* __restrict__ dev_vectorfield_z,
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const Float dx,
    const Float dy,
    const Float dz)
{
    // Read 6 cell-face values
    __syncthreads();
    const Float xn = dev_vectorfield_x[vidx(x,y,z)];
    const Float xp = dev_vectorfield_x[vidx(x+1,y,z)];
    const Float yn = dev_vectorfield_y[vidx(x,y,z)];
    const Float yp = dev_vectorfield_y[vidx(x,y+1,z)];
    const Float zn = dev_vectorfield_z[vidx(x,y,z)];
    const Float zp = dev_vectorfield_z[vidx(x,y,z+1)];

    // Calculate the central difference gradrients and the divergence
    return
        (xp - xn)/dx +
        (yp - yn)/dy +
        (zp - zn)/dz;
}

// Find the divergence of a tensor field
__device__ Float3 divergence_tensor(
    const Float* __restrict__ dev_tensorfield,
    const unsigned int x,
    const unsigned int y,
    const unsigned int z,
    const Float dx,
    const Float dy,
    const Float dz)
{
    __syncthreads();

    // Read the tensor in the 6 neighbor cells
    const Float t_xx_xp = dev_tensorfield[idx(x+1,y,z)*6];
    const Float t_xy_xp = dev_tensorfield[idx(x+1,y,z)*6+1];
    const Float t_xz_xp = dev_tensorfield[idx(x+1,y,z)*6+2];
    const Float t_yy_xp = dev_tensorfield[idx(x+1,y,z)*6+3];
    const Float t_yz_xp = dev_tensorfield[idx(x+1,y,z)*6+4];
    const Float t_zz_xp = dev_tensorfield[idx(x+1,y,z)*6+5];

    const Float t_xx_xn = dev_tensorfield[idx(x-1,y,z)*6];
    const Float t_xy_xn = dev_tensorfield[idx(x-1,y,z)*6+1];
    const Float t_xz_xn = dev_tensorfield[idx(x-1,y,z)*6+2];
    const Float t_yy_xn = dev_tensorfield[idx(x-1,y,z)*6+3];
    const Float t_yz_xn = dev_tensorfield[idx(x-1,y,z)*6+4];
    const Float t_zz_xn = dev_tensorfield[idx(x-1,y,z)*6+5];

    const Float t_xx_yp = dev_tensorfield[idx(x,y+1,z)*6];
    const Float t_xy_yp = dev_tensorfield[idx(x,y+1,z)*6+1];
    const Float t_xz_yp = dev_tensorfield[idx(x,y+1,z)*6+2];
    const Float t_yy_yp = dev_tensorfield[idx(x,y+1,z)*6+3];
    const Float t_yz_yp = dev_tensorfield[idx(x,y+1,z)*6+4];
    const Float t_zz_yp = dev_tensorfield[idx(x,y+1,z)*6+5];

    const Float t_xx_yn = dev_tensorfield[idx(x,y-1,z)*6];
    const Float t_xy_yn = dev_tensorfield[idx(x,y-1,z)*6+1];
    const Float t_xz_yn = dev_tensorfield[idx(x,y-1,z)*6+2];
    const Float t_yy_yn = dev_tensorfield[idx(x,y-1,z)*6+3];
    const Float t_yz_yn = dev_tensorfield[idx(x,y-1,z)*6+4];
    const Float t_zz_yn = dev_tensorfield[idx(x,y-1,z)*6+5];

    const Float t_xx_zp = dev_tensorfield[idx(x,y,z+1)*6];
    const Float t_xy_zp = dev_tensorfield[idx(x,y,z+1)*6+1];
    const Float t_xz_zp = dev_tensorfield[idx(x,y,z+1)*6+2];
    const Float t_yy_zp = dev_tensorfield[idx(x,y,z+1)*6+3];
    const Float t_yz_zp = dev_tensorfield[idx(x,y,z+1)*6+4];
    const Float t_zz_zp = dev_tensorfield[idx(x,y,z+1)*6+5];

    const Float t_xx_zn = dev_tensorfield[idx(x,y,z-1)*6];
    const Float t_xy_zn = dev_tensorfield[idx(x,y,z-1)*6+1];
    const Float t_xz_zn = dev_tensorfield[idx(x,y,z-1)*6+2];
    const Float t_yy_zn = dev_tensorfield[idx(x,y,z-1)*6+3];
    const Float t_yz_zn = dev_tensorfield[idx(x,y,z-1)*6+4];
    const Float t_zz_zn = dev_tensorfield[idx(x,y,z-1)*6+5];

    // Calculate div(phi*tau)
    const Float3 div_tensor = MAKE_FLOAT3(
        // x
        (t_xx_xp - t_xx_xn)/dx +
        (t_xy_yp - t_xy_yn)/dy +
        (t_xz_zp - t_xz_zn)/dz,
        // y
        (t_xy_xp - t_xy_xn)/dx +
        (t_yy_yp - t_yy_yn)/dy +
        (t_yz_zp - t_yz_zn)/dz,
        // z
        (t_xz_xp - t_xz_xn)/dx +
        (t_yz_yp - t_yz_yn)/dy +
        (t_zz_zp - t_zz_zn)/dz);

#ifdef CHECK_FLUID_FINITE
    (void)checkFiniteFloat3("div_tensor", x, y, z, div_tensor);
#endif
    return div_tensor;
}


// Find the spatial gradient in e.g. pressures per cell
// using first order central differences
__global__ void findNSgradientsDev(
    const Float* __restrict__ dev_scalarfield,     // in
    Float3* __restrict__ dev_vectorfield)    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Grid sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const Float3 grad = gradient(dev_scalarfield, x, y, z, dx, dy, dz);

        // Write gradient
        __syncthreads();
        dev_vectorfield[cellidx] = grad;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("grad", x, y, z, grad);
#endif
    }
}

// Find the outer product of v v
__global__ void findvvOuterProdNS(
    const Float3* __restrict__ dev_ns_v,       // in
    Float*  __restrict__ dev_ns_v_prod)  // out
{
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // 1D thread index
    const unsigned int cellidx6 = idx(x,y,z)*6;

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        __syncthreads();
        const Float3 v = dev_ns_v[idx(x,y,z)];

        // The outer product (v v) looks like:
        // [[ v_x^2    v_x*v_y  v_x*v_z ]
        //  [ v_y*v_x  v_y^2    v_y*v_z ]
        //  [ v_z*v_x  v_z*v_y  v_z^2   ]]

        // The tensor is symmetrical: value i,j = j,i.
        // Only the upper triangle is saved, with the cells given a linear index
        // enumerated as:
        // [[ 0 1 2 ]
        //  [   3 4 ]
        //  [     5 ]]

        __syncthreads();
        dev_ns_v_prod[cellidx6]   = v.x*v.x;
        dev_ns_v_prod[cellidx6+1] = v.x*v.y;
        dev_ns_v_prod[cellidx6+2] = v.x*v.z;
        dev_ns_v_prod[cellidx6+3] = v.y*v.y;
        dev_ns_v_prod[cellidx6+4] = v.y*v.z;
        dev_ns_v_prod[cellidx6+5] = v.z*v.z;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("v_prod[0]", x, y, z, v.x*v.x);
        (void)checkFiniteFloat("v_prod[1]", x, y, z, v.x*v.y);
        (void)checkFiniteFloat("v_prod[2]", x, y, z, v.x*v.z);
        (void)checkFiniteFloat("v_prod[3]", x, y, z, v.y*v.y);
        (void)checkFiniteFloat("v_prod[4]", x, y, z, v.y*v.z);
        (void)checkFiniteFloat("v_prod[5]", x, y, z, v.z*v.z);
#endif
    }
}


// Find the fluid stress tensor. It is symmetrical, and can thus be saved in 6
// values in 3D.
__global__ void findNSstressTensor(
    const Float3* __restrict__ dev_ns_v,       // in
    const Float mu,                            // in
    Float* __restrict__ dev_ns_tau)     // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx6 = idx(x,y,z)*6;

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // The fluid stress tensor (tau) looks like
        // [[ tau_xx  tau_xy  tau_xz ]
        //  [ tau_yx  tau_xy  tau_yz ]
        //  [ tau_zx  tau_zy  tau_zz ]]

        // The tensor is symmetrical: value i,j = j,i.
        // Only the upper triangle is saved, with the cells given a linear index
        // enumerated as:
        // [[ 0 1 2 ]
        //  [   3 4 ]
        //  [     5 ]]

        // Read neighbor values for central differences
        __syncthreads();
        const Float3 xp = dev_ns_v[idx(x+1,y,z)];
        const Float3 xn = dev_ns_v[idx(x-1,y,z)];
        const Float3 yp = dev_ns_v[idx(x,y+1,z)];
        const Float3 yn = dev_ns_v[idx(x,y-1,z)];
        const Float3 zp = dev_ns_v[idx(x,y,z+1)];
        const Float3 zn = dev_ns_v[idx(x,y,z-1)];

        // The diagonal stress tensor components
        const Float tau_xx = 2.0*mu*(xp.x - xn.x)/(2.0*dx);
        const Float tau_yy = 2.0*mu*(yp.y - yn.y)/(2.0*dy);
        const Float tau_zz = 2.0*mu*(zp.z - zn.z)/(2.0*dz);

        // The off-diagonal stress tensor components
        const Float tau_xy =
            mu*((yp.x - yn.x)/(2.0*dy) + (xp.y - xn.y)/(2.0*dx));
        const Float tau_xz =
            mu*((zp.x - zn.x)/(2.0*dz) + (xp.z - xn.z)/(2.0*dx));
        const Float tau_yz =
            mu*((zp.y - zn.y)/(2.0*dz) + (yp.z - yn.z)/(2.0*dy));

        /*
          if (x == 0 && y == 0 && z == 0)
          printf("mu = %f\n", mu);
          if (tau_xz > 1.0e-6)
          printf("%d,%d,%d\ttau_xx = %f\n", x,y,z, tau_xx);
          if (tau_yz > 1.0e-6)
          printf("%d,%d,%d\ttau_yy = %f\n", x,y,z, tau_yy);
          if (tau_zz > 1.0e-6)
          printf("%d,%d,%d\ttau_zz = %f\n", x,y,z, tau_zz);
        */

        // Store values in global memory
        __syncthreads();
        dev_ns_tau[cellidx6]   = tau_xx;
        dev_ns_tau[cellidx6+1] = tau_xy;
        dev_ns_tau[cellidx6+2] = tau_xz;
        dev_ns_tau[cellidx6+3] = tau_yy;
        dev_ns_tau[cellidx6+4] = tau_yz;
        dev_ns_tau[cellidx6+5] = tau_zz;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("tau_xx", x, y, z, tau_xx);
        (void)checkFiniteFloat("tau_xy", x, y, z, tau_xy);
        (void)checkFiniteFloat("tau_xz", x, y, z, tau_xz);
        (void)checkFiniteFloat("tau_yy", x, y, z, tau_yy);
        (void)checkFiniteFloat("tau_yz", x, y, z, tau_yz);
        (void)checkFiniteFloat("tau_zz", x, y, z, tau_zz);
#endif
    }
}


// Find the divergence of phi*v*v
__global__ void findNSdivphiviv(
    const Float*  __restrict__ dev_ns_phi,          // in
    const Float3* __restrict__ dev_ns_v,            // in
    Float3* __restrict__ dev_ns_div_phi_vi_v) // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read porosity and velocity in the 6 neighbor cells
        __syncthreads();
        const Float  phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float  phi    = dev_ns_phi[idx(x,y,z)];
        const Float  phi_xp = dev_ns_phi[idx(x+1,y,z)];
        const Float  phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float  phi_yp = dev_ns_phi[idx(x,y+1,z)];
        const Float  phi_zn = dev_ns_phi[idx(x,y,z-1)];
        const Float  phi_zp = dev_ns_phi[idx(x,y,z+1)];

        const Float3 v_xn = dev_ns_v[idx(x-1,y,z)];
        const Float3 v    = dev_ns_v[idx(x,y,z)];
        const Float3 v_xp = dev_ns_v[idx(x+1,y,z)];
        const Float3 v_yn = dev_ns_v[idx(x,y-1,z)];
        const Float3 v_yp = dev_ns_v[idx(x,y+1,z)];
        const Float3 v_zn = dev_ns_v[idx(x,y,z-1)];
        const Float3 v_zp = dev_ns_v[idx(x,y,z+1)];

        // Calculate upwind coefficients
        //*
        const Float3 a = MAKE_FLOAT3(
            copysign(1.0, v.x),
            copysign(1.0, v.y),
            copysign(1.0, v.z));

        // Calculate the divergence based on the upwind differences (Griebel et
        // al. 1998, eq. 3.9)
        const Float3 div_uw = MAKE_FLOAT3(
            // x
            ((1.0 + a.x)*(phi*v.x*v.x - phi_xn*v_xn.x*v_xn.x) +
             (1.0 - a.x)*(phi_xp*v_xp.x*v_xp.x - phi*v.x*v.x))/(2.0*dx) +

            ((1.0 + a.y)*(phi*v.x*v.y - phi_yn*v_yn.x*v_yn.y) +
             (1.0 - a.y)*(phi_yp*v_yp.x*v_yp.y - phi*v.x*v.y))/(2.0*dy) +

            ((1.0 + a.z)*(phi*v.x*v.z - phi_zn*v_zn.x*v_zn.z) +
             (1.0 - a.z)*(phi_zp*v_zp.x*v_zp.z - phi*v.x*v.z))/(2.0*dz),

            // y
            ((1.0 + a.x)*(phi*v.y*v.x - phi_xn*v_xn.y*v_xn.x) +
             (1.0 - a.x)*(phi_xp*v_xp.y*v_xp.x - phi*v.y*v.x))/(2.0*dx) +

            ((1.0 + a.y)*(phi*v.y*v.y - phi_yn*v_yn.y*v_yn.y) +
             (1.0 - a.y)*(phi_yp*v_yp.y*v_yp.y - phi*v.y*v.y))/(2.0*dy) +

            ((1.0 + a.z)*(phi*v.y*v.z - phi_zn*v_zn.y*v_zn.z) +
             (1.0 - a.z)*(phi_zp*v_zp.y*v_zp.z - phi*v.y*v.z))/(2.0*dz),

            // z
            ((1.0 + a.x)*(phi*v.z*v.x - phi_xn*v_xn.z*v_xn.x) +
             (1.0 - a.x)*(phi_xp*v_xp.z*v_xp.x - phi*v.z*v.x))/(2.0*dx) +

            ((1.0 + a.y)*(phi*v.z*v.y - phi_yn*v_yn.z*v_yn.y) +
             (1.0 - a.y)*(phi_yp*v_yp.z*v_yp.y - phi*v.z*v.y))/(2.0*dy) +

            ((1.0 + a.z)*(phi*v.z*v.z - phi_zn*v_zn.z*v_zn.z) +
             (1.0 - a.z)*(phi_zp*v_zp.z*v_zp.z - phi*v.z*v.z))/(2.0*dz));


        // Calculate the divergence based on the central-difference gradients
        const Float3 div_cd = MAKE_FLOAT3(
            // x
            (phi_xp*v_xp.x*v_xp.x - phi_xn*v_xn.x*v_xn.x)/(2.0*dx) +
            (phi_yp*v_yp.x*v_yp.y - phi_yn*v_yn.x*v_yn.y)/(2.0*dy) +
            (phi_zp*v_zp.x*v_zp.z - phi_zn*v_zn.x*v_zn.z)/(2.0*dz),
            // y
            (phi_xp*v_xp.y*v_xp.x - phi_xn*v_xn.y*v_xn.x)/(2.0*dx) +
            (phi_yp*v_yp.y*v_yp.y - phi_yn*v_yn.y*v_yn.y)/(2.0*dy) +
            (phi_zp*v_zp.y*v_zp.z - phi_zn*v_zn.y*v_zn.z)/(2.0*dz),
            // z
            (phi_xp*v_xp.z*v_xp.x - phi_xn*v_xn.z*v_xn.x)/(2.0*dx) +
            (phi_yp*v_yp.z*v_yp.y - phi_yn*v_yn.z*v_yn.y)/(2.0*dy) +
            (phi_zp*v_zp.z*v_zp.z - phi_zn*v_zn.z*v_zn.z)/(2.0*dz));

        // Weighting parameter
        const Float tau = 0.5;

        // Determine the weighted average of both discretizations
        const Float3 div_phi_vi_v = tau*div_uw + (1.0 - tau)*div_cd;
        //*/

        /*
        // Calculate the divergence: div(phi*v_i*v)
        const Float3 div_phi_vi_v = MAKE_FLOAT3(
        // x
        (phi_xp*v_xp.x*v_xp.x - phi_xn*v_xn.x*v_xn.x)/(2.0*dx) +
        (phi_yp*v_yp.x*v_yp.y - phi_yn*v_yn.x*v_yn.y)/(2.0*dy) +
        (phi_zp*v_zp.x*v_zp.z - phi_zn*v_zn.x*v_zn.z)/(2.0*dz),
        // y
        (phi_xp*v_xp.y*v_xp.x - phi_xn*v_xn.y*v_xn.x)/(2.0*dx) +
        (phi_yp*v_yp.y*v_yp.y - phi_yn*v_yn.y*v_yn.y)/(2.0*dy) +
        (phi_zp*v_zp.y*v_zp.z - phi_zn*v_zn.y*v_zn.z)/(2.0*dz),
        // z
        (phi_xp*v_xp.z*v_xp.x - phi_xn*v_xn.z*v_xn.x)/(2.0*dx) +
        (phi_yp*v_yp.z*v_yp.y - phi_yn*v_yn.z*v_yn.y)/(2.0*dy) +
        (phi_zp*v_zp.z*v_zp.z - phi_zn*v_zn.z*v_zn.z)/(2.0*dz));
        // */

        // Write divergence
        __syncthreads();
        dev_ns_div_phi_vi_v[cellidx] = div_phi_vi_v;

        //printf("div(phi*v*v) [%d,%d,%d] = %f, %f, %f\n", x,y,z,
        //div_phi_vi_v.x, div_phi_vi_v.y, div_phi_vi_v.z);

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("div_phi_vi_v", x, y, z, div_phi_vi_v);
#endif
    }
}

__global__ void findNSdivtau(
    const Float* __restrict__ dev_ns_tau,      // in
    Float3* __restrict__ dev_ns_div_tau)  // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {
        
        __syncthreads();
        const Float3 div_tau =
            divergence_tensor(dev_ns_tau, x, y, z, dx, dy, dz);

        __syncthreads();
        dev_ns_div_tau[cellidx] = div_tau;
    }
}


// Find the divergence of phi*tau
__global__ void findNSdivphitau(
    const Float* __restrict__ dev_ns_phi,          // in
    const Float* __restrict__ dev_ns_tau,          // in
    Float3* __restrict__ dev_ns_div_phi_tau)  // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read the porosity in the 6 neighbor cells
        __syncthreads();
        const Float phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float phi_xp = dev_ns_phi[idx(x+1,y,z)];
        const Float phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float phi_yp = dev_ns_phi[idx(x,y+1,z)];
        const Float phi_zn = dev_ns_phi[idx(x,y,z-1)];
        const Float phi_zp = dev_ns_phi[idx(x,y,z+1)];

        // Read the stress tensor in the 6 neighbor cells
        const Float tau_xx_xp = dev_ns_tau[idx(x+1,y,z)*6];
        const Float tau_xy_xp = dev_ns_tau[idx(x+1,y,z)*6+1];
        const Float tau_xz_xp = dev_ns_tau[idx(x+1,y,z)*6+2];
        const Float tau_yy_xp = dev_ns_tau[idx(x+1,y,z)*6+3];
        const Float tau_yz_xp = dev_ns_tau[idx(x+1,y,z)*6+4];
        const Float tau_zz_xp = dev_ns_tau[idx(x+1,y,z)*6+5];

        const Float tau_xx_xn = dev_ns_tau[idx(x-1,y,z)*6];
        const Float tau_xy_xn = dev_ns_tau[idx(x-1,y,z)*6+1];
        const Float tau_xz_xn = dev_ns_tau[idx(x-1,y,z)*6+2];
        const Float tau_yy_xn = dev_ns_tau[idx(x-1,y,z)*6+3];
        const Float tau_yz_xn = dev_ns_tau[idx(x-1,y,z)*6+4];
        const Float tau_zz_xn = dev_ns_tau[idx(x-1,y,z)*6+5];

        const Float tau_xx_yp = dev_ns_tau[idx(x,y+1,z)*6];
        const Float tau_xy_yp = dev_ns_tau[idx(x,y+1,z)*6+1];
        const Float tau_xz_yp = dev_ns_tau[idx(x,y+1,z)*6+2];
        const Float tau_yy_yp = dev_ns_tau[idx(x,y+1,z)*6+3];
        const Float tau_yz_yp = dev_ns_tau[idx(x,y+1,z)*6+4];
        const Float tau_zz_yp = dev_ns_tau[idx(x,y+1,z)*6+5];

        const Float tau_xx_yn = dev_ns_tau[idx(x,y-1,z)*6];
        const Float tau_xy_yn = dev_ns_tau[idx(x,y-1,z)*6+1];
        const Float tau_xz_yn = dev_ns_tau[idx(x,y-1,z)*6+2];
        const Float tau_yy_yn = dev_ns_tau[idx(x,y-1,z)*6+3];
        const Float tau_yz_yn = dev_ns_tau[idx(x,y-1,z)*6+4];
        const Float tau_zz_yn = dev_ns_tau[idx(x,y-1,z)*6+5];

        const Float tau_xx_zp = dev_ns_tau[idx(x,y,z+1)*6];
        const Float tau_xy_zp = dev_ns_tau[idx(x,y,z+1)*6+1];
        const Float tau_xz_zp = dev_ns_tau[idx(x,y,z+1)*6+2];
        const Float tau_yy_zp = dev_ns_tau[idx(x,y,z+1)*6+3];
        const Float tau_yz_zp = dev_ns_tau[idx(x,y,z+1)*6+4];
        const Float tau_zz_zp = dev_ns_tau[idx(x,y,z+1)*6+5];

        const Float tau_xx_zn = dev_ns_tau[idx(x,y,z-1)*6];
        const Float tau_xy_zn = dev_ns_tau[idx(x,y,z-1)*6+1];
        const Float tau_xz_zn = dev_ns_tau[idx(x,y,z-1)*6+2];
        const Float tau_yy_zn = dev_ns_tau[idx(x,y,z-1)*6+3];
        const Float tau_yz_zn = dev_ns_tau[idx(x,y,z-1)*6+4];
        const Float tau_zz_zn = dev_ns_tau[idx(x,y,z-1)*6+5];

        // Calculate div(phi*tau)
        const Float3 div_phi_tau = MAKE_FLOAT3(
            // x
            (phi_xp*tau_xx_xp - phi_xn*tau_xx_xn)/dx +
            (phi_yp*tau_xy_yp - phi_yn*tau_xy_yn)/dy +
            (phi_zp*tau_xz_zp - phi_zn*tau_xz_zn)/dz,
            // y
            (phi_xp*tau_xy_xp - phi_xn*tau_xy_xn)/dx +
            (phi_yp*tau_yy_yp - phi_yn*tau_yy_yn)/dy +
            (phi_zp*tau_yz_zp - phi_zn*tau_yz_zn)/dz,
            // z
            (phi_xp*tau_xz_xp - phi_xn*tau_xz_xn)/dx +
            (phi_yp*tau_yz_yp - phi_yn*tau_yz_yn)/dy +
            (phi_zp*tau_zz_zp - phi_zn*tau_zz_zn)/dz);

        // Write divergence
        __syncthreads();
        dev_ns_div_phi_tau[cellidx] = div_phi_tau;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("div_phi_tau", x, y, z, div_phi_tau);
#endif
    }
}

// Find the divergence of phi v v
// Unused
__global__ void findNSdivphivv(
    const Float* __restrict__ dev_ns_v_prod, // in
    const Float* __restrict__ dev_ns_phi,    // in
    Float3* __restrict__ dev_ns_div_phi_v_v) // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Read cell and 6 neighbor cells
        __syncthreads();
        //const Float  phi    = dev_ns_phi[cellidx];
        const Float  phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float  phi_xp = dev_ns_phi[idx(x+1,y,z)];
        const Float  phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float  phi_yp = dev_ns_phi[idx(x,y+1,z)];
        const Float  phi_zn = dev_ns_phi[idx(x,y,z-1)];
        const Float  phi_zp = dev_ns_phi[idx(x,y,z+1)];

        // The tensor is symmetrical: value i,j = j,i.
        // Only the upper triangle is saved, with the cells given a linear index
        // enumerated as:
        // [[ 0 1 2 ]
        //  [   3 4 ]
        //  [     5 ]]

        // div(T) = 
        //  [ de_xx/dx + de_xy/dy + de_xz/dz ,
        //    de_yx/dx + de_yy/dy + de_yz/dz ,
        //    de_zx/dx + de_zy/dy + de_zz/dz ]

        // This function finds the divergence of (phi v v), which is a vector

        // Calculate the divergence. See
        // https://en.wikipedia.org/wiki/Divergence#Application_in_Cartesian_coordinates
        // The symmetry described in findvvOuterProdNS is used
        __syncthreads();
        const Float3 div = MAKE_FLOAT3(
            ((dev_ns_v_prod[idx(x+1,y,z)*6]*phi_xp
              - dev_ns_v_prod[idx(x-1,y,z)*6]*phi_xn)/(2.0*dx) +
             (dev_ns_v_prod[idx(x,y+1,z)*6+1]*phi_yp
              - dev_ns_v_prod[idx(x,y-1,z)*6+1]*phi_yn)/(2.0*dy) +
             (dev_ns_v_prod[idx(x,y,z+1)*6+2]*phi_zp
              - dev_ns_v_prod[idx(x,y,z-1)*6+2]*phi_zn)/(2.0*dz)),
            ((dev_ns_v_prod[idx(x+1,y,z)*6+1]*phi_xp
              - dev_ns_v_prod[idx(x-1,y,z)*6+1]*phi_xn)/(2.0*dx) +
             (dev_ns_v_prod[idx(x,y+1,z)*6+3]*phi_yp
              - dev_ns_v_prod[idx(x,y-1,z)*6+3]*phi_yn)/(2.0*dy) +
             (dev_ns_v_prod[idx(x,y,z+1)*6+4]*phi_zp
              - dev_ns_v_prod[idx(x,y,z-1)*6+4]*phi_zn)/(2.0*dz)),
            ((dev_ns_v_prod[idx(x+1,y,z)*6+2]*phi_xp
              - dev_ns_v_prod[idx(x-1,y,z)*6+2]*phi_xn)/(2.0*dx) +
             (dev_ns_v_prod[idx(x,y+1,z)*6+4]*phi_yp
              - dev_ns_v_prod[idx(x,y-1,z)*6+4]*phi_yn)/(2.0*dy) +
             (dev_ns_v_prod[idx(x,y,z+1)*6+5]*phi_zp
              - dev_ns_v_prod[idx(x,y,z-1)*6+5]*phi_zn)/(2.0*dz)) );

        //printf("div[%d,%d,%d] = %f\t%f\t%f\n", x, y, z, div.x, div.y, div.z);

        // Write divergence
        __syncthreads();
        dev_ns_div_phi_v_v[cellidx] = div;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("div_phi_v_v", x, y, z, div);
#endif
    }
}


// Find predicted fluid velocity
// Launch per face.
__global__ void findPredNSvelocities(
    const Float*  __restrict__ dev_ns_p,               // in
    const Float*  __restrict__ dev_ns_v_x,             // in
    const Float*  __restrict__ dev_ns_v_y,             // in
    const Float*  __restrict__ dev_ns_v_z,             // in
    const Float*  __restrict__ dev_ns_phi,             // in
    const Float*  __restrict__ dev_ns_dphi,            // in
    const Float*  __restrict__ dev_ns_div_tau_x,       // in
    const Float*  __restrict__ dev_ns_div_tau_y,       // in
    const Float*  __restrict__ dev_ns_div_tau_z,       // in
    const Float3* __restrict__ dev_ns_div_phi_vi_v,    // in
    const int     bc_bot,                              // in
    const int     bc_top,                              // in
    const Float   beta,                                // in
    const Float3* __restrict__ dev_ns_F_pf,            // in
    const unsigned int ndem,                           // in
    const unsigned int wall0_iz,                       // in
    const Float   c_v,                                 // in
    const Float   mu,                                  // in
    const Float   rho_f,                               // in
    Float* __restrict__ dev_ns_v_p_x,           // out
    Float* __restrict__ dev_ns_v_p_y,           // out
    Float* __restrict__ dev_ns_v_p_z)           // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int fidx = vidx(x,y,z);
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    //if (x <= nx && y <= ny && z <= nz) {
    if (x < nx && y < ny && z < nz) {

        // Values that are needed for calculating the predicted velocity
        __syncthreads();
        const Float3 v = MAKE_FLOAT3(
            dev_ns_v_x[fidx],
            dev_ns_v_y[fidx],
            dev_ns_v_z[fidx]);
        //printf("v in v* [%d,%d,%d] = %f, %f, %f\n", x,y,z, v.x, v.y, v.z);

        Float3 div_tau = MAKE_FLOAT3(0.0, 0.0, 0.0);
        if (mu > 0.0) {
            div_tau = MAKE_FLOAT3(
                dev_ns_div_tau_x[fidx],
                dev_ns_div_tau_y[fidx],
                dev_ns_div_tau_z[fidx]);
        }

        // cell center values
        const Float phi_xn    = dev_ns_phi[idx(x-1,y,z)];
        const Float phi_c     = dev_ns_phi[cellidx];
        const Float phi_yn    = dev_ns_phi[idx(x,y-1,z)];
        const Float phi_zn    = dev_ns_phi[idx(x,y,z-1)];

        const Float dphi_xn   = dev_ns_dphi[idx(x-1,y,z)];
        const Float dphi_c    = dev_ns_dphi[cellidx];
        const Float dphi_yn   = dev_ns_dphi[idx(x,y-1,z)];
        const Float dphi_zn   = dev_ns_dphi[idx(x,y,z-1)];

        const Float3 div_phi_vi_v_xn = dev_ns_div_phi_vi_v[idx(x-1,y,z)];
        const Float3 div_phi_vi_v_c  = dev_ns_div_phi_vi_v[cellidx];
        const Float3 div_phi_vi_v_yn = dev_ns_div_phi_vi_v[idx(x,y-1,z)];
        const Float3 div_phi_vi_v_zn = dev_ns_div_phi_vi_v[idx(x,y,z-1)];

        // component-wise average values
        const Float3 phi = MAKE_FLOAT3(
            amean(phi_c, phi_xn),
            amean(phi_c, phi_yn),
            amean(phi_c, phi_zn));
        const Float3 dphi = MAKE_FLOAT3(
            amean(dphi_c, dphi_xn),
            amean(dphi_c, dphi_yn),
            amean(dphi_c, dphi_zn));

        // The particle-fluid interaction force should only be incoorporated if
        // there is a fluid viscosity
        Float3 f_i_c, f_i_xn, f_i_yn, f_i_zn;
        if (mu > 0.0 && devC_np > 0) {
            f_i_c  = dev_ns_F_pf[cellidx];
            f_i_xn = dev_ns_F_pf[idx(x-1,y,z)];
            f_i_yn = dev_ns_F_pf[idx(x,y-1,z)];
            f_i_zn = dev_ns_F_pf[idx(x,y,z-1)];
        } else {
            f_i_c  = MAKE_FLOAT3(0.0, 0.0, 0.0);
            f_i_xn = MAKE_FLOAT3(0.0, 0.0, 0.0);
            f_i_yn = MAKE_FLOAT3(0.0, 0.0, 0.0);
            f_i_zn = MAKE_FLOAT3(0.0, 0.0, 0.0);
        }
        const Float3 f_i = MAKE_FLOAT3(
            amean(f_i_c.x, f_i_xn.x),
            amean(f_i_c.y, f_i_yn.y),
            amean(f_i_c.z, f_i_zn.z));

        const Float dt = ndem*devC_dt;

        // The pressure gradient is not needed in Chorin's projection method
        // (ns.beta=0), so only has to be looked up in pressure-dependant
        // projection methods
        Float3 pressure_term = MAKE_FLOAT3(0.0, 0.0, 0.0);
        if (beta > 0.0) {
            __syncthreads();
            const Float p    = dev_ns_p[cellidx];
            const Float p_xn = dev_ns_p[idx(x-1,y,z)];
            const Float p_yn = dev_ns_p[idx(x,y-1,z)];
            const Float p_zn = dev_ns_p[idx(x,y,z-1)];
            const Float3 grad_p = MAKE_FLOAT3(
                (p - p_xn)/dx,
                (p - p_yn)/dy,
                (p - p_zn)/dz);
#ifdef SET_1
            pressure_term = -beta*dt/(rho_f*phi)*grad_p;
#endif
#ifdef SET_2
            pressure_term = -beta*dt/rho_f*grad_p;
#endif
        }

        const Float3 div_phi_vi_v = MAKE_FLOAT3(
            amean(div_phi_vi_v_xn.x, div_phi_vi_v_c.x),
            amean(div_phi_vi_v_yn.x, div_phi_vi_v_c.y),
            amean(div_phi_vi_v_zn.x, div_phi_vi_v_c.z));

        // Determine the terms of the predicted velocity change
        const Float3 interaction_term = -dt/(rho_f*phi)*f_i;
        const Float3 gravity_term = MAKE_FLOAT3(
            devC_params.g[0], devC_params.g[1], devC_params.g[2])*dt;
        const Float3 advection_term = -1.0*div_phi_vi_v*dt/phi;
        const Float3 porosity_term = -1.0*v*dphi/phi;
#ifdef SET_1
        const Float3 diffusion_term = dt/(rho_f*phi)*div_tau;
#endif
#ifdef SET_2
        const Float3 diffusion_term = dt/rho_f*div_tau;
#endif

        // Predict new velocity and add scaling parameters
        Float3 v_p = v + c_v*(
                pressure_term
                + interaction_term
                + diffusion_term
                + gravity_term
                + porosity_term
                + advection_term);

        //// Neumann BCs
        if ((z == 0 && bc_bot == 1) || (z > nz-1 && bc_top == 1))
            v_p.z = 0.0;

        if ((z == 0 && bc_bot == 2) || (z > nz-1 && bc_top == 2))
            v_p = MAKE_FLOAT3(0.0, 0.0, 0.0);

        // Set velocities to zero above the dynamic wall
        if (z >= wall0_iz)
            v_p = MAKE_FLOAT3(0.0, 0.0, 0.0);

#ifdef REPORT_V_P_COMPONENTS
        // Report velocity components to stdout for debugging
        printf("\n[%d,%d,%d] REPORT_V_P_COMPONENTS\n"
               "\tv_p      = %+e %+e %+e\n"
               "\tv_old    = %+e %+e %+e\n"
               "\tpres     = %+e %+e %+e\n"
               "\tinteract = %+e %+e %+e\n"
               "\tdiff     = %+e %+e %+e\n"
               "\tgrav     = %+e %+e %+e\n"
               "\tporos    = %+e %+e %+e\n"
               "\tadv      = %+e %+e %+e\n",
               x, y, z,
               v_p.x, v_p.y, v_p.z,
               v.x, v.y, v.z,
               c_v*pressure_term.x,
               c_v*pressure_term.y,
               c_v*pressure_term.z,
               c_v*interaction_term.x,
               c_v*interaction_term.y,
               c_v*interaction_term.z,
               c_v*diffusion_term.x,
               c_v*diffusion_term.y,
               c_v*diffusion_term.z,
               c_v*gravity_term.x,
               c_v*gravity_term.y,
               c_v*gravity_term.z,
               c_v*porosity_term.x,
               c_v*porosity_term.y,
               c_v*porosity_term.z,
               c_v*advection_term.x,
               c_v*advection_term.y,
               c_v*advection_term.z);
#endif

        // Save the predicted velocity
        __syncthreads();
        dev_ns_v_p_x[fidx] = v_p.x;
        dev_ns_v_p_y[fidx] = v_p.y;
        dev_ns_v_p_z[fidx] = v_p.z;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat3("v_p", x, y, z, v_p);
#endif
    }
}

// Find the value of the forcing function. Only grad(epsilon) changes during
// the Jacobi iterations. The remaining, constant terms are only calculated
// during the first iteration.
// At each iteration, the value of the forcing function is found as:
//   f = f1 - f2 dot grad(epsilon)
__global__ void findNSforcing(
    const Float*  __restrict__ dev_ns_epsilon,     // in
    const Float*  __restrict__ dev_ns_phi,         // in
    const Float*  __restrict__ dev_ns_dphi,        // in
    const Float3* __restrict__ dev_ns_v_p,         // in
    const Float*  __restrict__ dev_ns_v_p_x,       // in
    const Float*  __restrict__ dev_ns_v_p_y,       // in
    const Float*  __restrict__ dev_ns_v_p_z,       // in
    const unsigned int nijac,                      // in
    const unsigned int ndem,                       // in
    const Float c_v,                               // in
    const Float rho_f,                             // in
    Float*  __restrict__ dev_ns_f1,                // out
    Float3* __restrict__ dev_ns_f2,                // out
    Float*  __restrict__ dev_ns_f)                 // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);


    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        // Constant forcing function terms
        Float f1;
        Float3 f2;

        // Check if this is the first Jacobi iteration. If it is, find f1 and f2
        if (nijac == 0) {

            // Read needed values
            __syncthreads();
            const Float3 v_p  = dev_ns_v_p[cellidx];
            const Float  phi  = dev_ns_phi[cellidx];
            const Float  dphi = dev_ns_dphi[cellidx];

            // Calculate derivatives
            const Float  div_v_p
                = divergence(dev_ns_v_p_x, dev_ns_v_p_y, dev_ns_v_p_z,
                             x, y, z, dx, dy, dz);
            const Float3 grad_phi
                = gradient(dev_ns_phi, x, y, z, dx, dy, dz);

            // CFD time step
            const Float dt = devC_dt*ndem;

            // Find forcing function terms
#ifdef SET_1
            const Float t1 = rho_f*phi*div_v_p/(c_v*dt);
            const Float t2 = rho_f*dot(grad_phi, v_p)/(c_v*dt);
            const Float t4 = rho_f*dphi/(dt*dt*c_v);
#endif
#ifdef SET_2
            const Float t1 = rho_f*div_v_p/(c_v*dt);
            const Float t2 = rho_f*dot(grad_phi, v_p)/(c_v*dt*phi);
            const Float t4 = rho_f*dphi/(dt*dt*phi*c_v);
#endif
            f1 = t1 + t2 + t4;
            f2 = grad_phi/phi; // t3/grad(epsilon)

#ifdef REPORT_FORCING_TERMS
            // Report values terms in the forcing function for debugging
            printf("[%d,%d,%d] REPORT_FORCING_TERMS\t"
                    "t1 = %f\tt2 = %f\tt4 = %f\n", x,y,z, t1, t2, t4);
#endif

            // Save values
            __syncthreads();
            dev_ns_f1[cellidx] = f1;
            dev_ns_f2[cellidx] = f2;

        } else {

            // Read previously found values
            __syncthreads();
            f1 = dev_ns_f1[cellidx];
            f2 = dev_ns_f2[cellidx];
        }

        // Find the gradient of epsilon, which changes during Jacobi iterations
        const Float3 grad_epsilon
            = gradient(dev_ns_epsilon, x, y, z, dx, dy, dz);

        // Forcing function value
        const Float f = f1 - dot(grad_epsilon, f2);

#ifdef REPORT_FORCING_TERMS
        const Float t3 = -dot(f2, grad_epsilon);
        if (z >= nz-3)
            printf("[%d,%d,%d] REPORT_FORCING_TERMS\tf= %f\tf1 = %f\tt3 = %f\n",
                   x,y,z, f, f1, t3);
#endif

        // Save forcing function value
        __syncthreads();
        dev_ns_f[cellidx] = f;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("f", x, y, z, f);
#endif
    }
}

// Spatial smoothing, used for the epsilon values. If there are several blocks,
// there will be small errors at the block boundaries, since the update will mix
// non-smoothed and smoothed values.
template<typename T>
__global__ void smoothing(
    T* __restrict__ dev_arr,
    const Float gamma,
    const unsigned int bc_bot,
    const unsigned int bc_top)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Perform the epsilon updates for all non-ghost nodes except the
    // Dirichlet boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;

    if (x < nx && y < ny && z >= z_min && z <= z_max) {

        __syncthreads();
        const T e_xn = dev_arr[idx(x-1,y,z)];
        const T e    = dev_arr[cellidx];
        const T e_xp = dev_arr[idx(x+1,y,z)];
        const T e_yn = dev_arr[idx(x,y-1,z)];
        const T e_yp = dev_arr[idx(x,y+1,z)];
        const T e_zn = dev_arr[idx(x,y,z-1)];
        const T e_zp = dev_arr[idx(x,y,z+1)];

        const T e_avg_neigbors = 1.0/6.0 *
            (e_xn + e_xp + e_yn + e_yp + e_zn + e_zp);

        const T e_smooth = (1.0 - gamma)*e + gamma*e_avg_neigbors;

        __syncthreads();
        dev_arr[cellidx] = e_smooth;

        //printf("%d,%d,%d\te = %f e_smooth = %f\n", x,y,z, e, e_smooth);
        /*printf("%d,%d,%d\te_xn = %f, e_xp = %f, e_yn = %f, e_yp = %f,"
          " e_zn = %f, e_zp = %f\n", x,y,z, e_xn, e_xp,
          e_yn, e_yp, e_zn, e_zp);*/

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("e_smooth", x, y, z, e_smooth);
#endif
    }
}

// Perform a single Jacobi iteration
__global__ void jacobiIterationNS(
    const Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_epsilon_new,
    Float* __restrict__ dev_ns_norm,
    const Float* __restrict__ dev_ns_f,
    const int bc_bot,
    const int bc_top,
    const Float theta,
    const unsigned int wall0_iz,
    const Float dp_dz)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    //if (x < nx && y < ny && z < nz) {

    // internal nodes only
    //if (x > 0 && x < nx-1 && y > 0 && y < ny-1 && z > 0 && z < nz-1) {

    // Lower boundary: Dirichlet. Upper boundary: Dirichlet
    //if (x < nx && y < ny && z > 0 && z < nz-1) {

    // Lower boundary: Neumann. Upper boundary: Dirichlet
    //if (x < nx && y < ny && z < nz-1) {

    // Perform the epsilon updates for all non-ghost nodes except the Dirichlet
    // boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;

    if (x < nx && y < ny && z >= z_min && z <= z_max) {

        // Read the epsilon values from the cell and its 6 neighbors
        __syncthreads();
        const Float e_xn = dev_ns_epsilon[idx(x-1,y,z)];
        const Float e    = dev_ns_epsilon[cellidx];
        const Float e_xp = dev_ns_epsilon[idx(x+1,y,z)];
        const Float e_yn = dev_ns_epsilon[idx(x,y-1,z)];
        const Float e_yp = dev_ns_epsilon[idx(x,y+1,z)];
        const Float e_zn = dev_ns_epsilon[idx(x,y,z-1)];
        const Float e_zp = dev_ns_epsilon[idx(x,y,z+1)];

        // Read the value of the forcing function
        const Float f = dev_ns_f[cellidx];

        // New value of epsilon in 3D update, derived by rearranging the
        // discrete Laplacian
        const Float dxdx = dx*dx;
        const Float dydy = dy*dy;
        const Float dzdz = dz*dz;
        Float e_new
            = (-dxdx*dydy*dzdz*f
               + dydy*dzdz*(e_xn + e_xp)
               + dxdx*dzdz*(e_yn + e_yp)
               + dxdx*dydy*(e_zn + e_zp))
            /(2.0*(dxdx*dydy + dxdx*dzdz + dydy*dzdz));


        // Dirichlet BC at dynamic top wall. wall0_iz will be larger than the
        // grid if the wall isn't dynamic
        if (z == wall0_iz)
            e_new = e;

        // Neumann BC at dynamic top wall
        if (z == wall0_iz + 1)
            e_new = e_zn + dp_dz;

        // New value of epsilon in 1D update
        //const Float e_new = (e_zp + e_zn - dz*dz*f)/2.0;

        // Print values for debugging
        /*printf("[%d,%d,%d]\t e = %f\tf = %f\te_new = %f\n",
          x,y,z, e, f, e_new);*/

        const Float res_norm = (e_new - e)*(e_new - e)/(e_new*e_new + 1.0e-16);
        const Float e_relax = e*(1.0-theta) + e_new*theta;

        __syncthreads();
        dev_ns_epsilon_new[cellidx] = e_relax;
        dev_ns_norm[cellidx] = res_norm;

#ifdef CHECK_FLUID_FINITE
        (void)checkFiniteFloat("e_new", x, y, z, e_new);
        (void)checkFiniteFloat("e_relax", x, y, z, e_relax);
        //(void)checkFiniteFloat("res_norm", x, y, z, res_norm);
        if (checkFiniteFloat("res_norm", x, y, z, res_norm)) {
            printf("[%d,%d,%d]\t e = %f\tf = %f\te_new = %f\tres_norm = %f\n",
                   x,y,z, e, f, e_new, res_norm);
        }
#endif
    }
}

// Copy all values from one array to the other
template<typename T>
__global__ void copyValues(
    const T* __restrict__ dev_read,
    T* __restrict__ dev_write)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Internal nodes only
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // Internal nodes + ghost nodes
        /*if (x <= devC_grid.num[0]+1 &&
          y <= devC_grid.num[1]+1 &&
          z <= devC_grid.num[2]+1) {*/

        const unsigned int cellidx = idx(x,y,z); // without ghost nodes
        //const unsigned int cellidx = idx(x-1,y-1,z-1); // with ghost nodes

        // Read
        __syncthreads();
        const T val = dev_read[cellidx];

        //if (z == devC_grid.num[2]-1)
        //printf("[%d,%d,%d] = %f\n", x, y, z, val);

        // Write
        __syncthreads();
        dev_write[cellidx] = val;
    }
}

// Find and store the normalized residuals
__global__ void findNormalizedResiduals(
    const Float* __restrict__ dev_ns_epsilon_old,
    const Float* __restrict__ dev_ns_epsilon,
    Float* __restrict__ dev_ns_norm,
    const unsigned int bc_bot,
    const unsigned int bc_top)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Perform the epsilon updates for all non-ghost nodes except the
    // Dirichlet boundaries at z=0 and z=nz-1.
    // Adjust z range if a boundary has the Dirichlet boundary condition.
    int z_min = 0;
    int z_max = nz-1;
    if (bc_bot == 0)
        z_min = 1;
    if (bc_top == 0)
        z_max = nz-2;

    if (x < nx && y < ny && z >= z_min && z <= z_max) {

        __syncthreads();
        const Float e = dev_ns_epsilon_old[cellidx];
        const Float e_new = dev_ns_epsilon[cellidx];

        // Find the normalized residual value. A small value is added to the
        // denominator to avoid a divide by zero.
        //const Float res_norm = (e_new - e)*(e_new - e)/(e_new*e_new + 1.0e-16);
        const Float res_norm = (e_new - e)/(e + 1.0e-16);

        __syncthreads();
        dev_ns_norm[cellidx] = res_norm;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("res_norm", x, y, z, res_norm);
#endif
    }
}


// Computes the new velocity and pressure using the corrector
__global__ void updateNSpressure(
    const Float* __restrict__ dev_ns_epsilon,  // in
    const Float  __restrict__ beta,            // in
    Float* __restrict__ dev_ns_p)              // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // 1D thread index
    const unsigned int cellidx = idx(x,y,z);

    // Check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        // Read values
        __syncthreads();
        const Float  p_old   = dev_ns_p[cellidx];
        const Float  epsilon = dev_ns_epsilon[cellidx];

        // New pressure
        Float p = beta*p_old + epsilon;

        // Write new values
        __syncthreads();
        dev_ns_p[cellidx] = p;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat("p", x, y, z, p);
#endif
    }
}

__global__ void updateNSvelocity(
    const Float* __restrict__ dev_ns_v_p_x,    // in
    const Float* __restrict__ dev_ns_v_p_y,    // in
    const Float* __restrict__ dev_ns_v_p_z,    // in
    const Float* __restrict__ dev_ns_phi,      // in
    const Float* __restrict__ dev_ns_epsilon,  // in
    const Float  beta,            // in
    const int    bc_bot,          // in
    const int    bc_top,          // in
    const unsigned int ndem,      // in
    const Float  c_v,             // in
    const Float  rho_f,           // in
    const unsigned int wall0_iz,  // in
    const unsigned int iter,      // in
    Float* __restrict__ dev_ns_v_x,      // out
    Float* __restrict__ dev_ns_v_y,      // out
    Float* __restrict__ dev_ns_v_z)      // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // 1D thread index
    const unsigned int cellidx = vidx(x,y,z);

    // Check that we are not outside the fluid grid
    //if (x <= nx && y <= ny && z <= nz) {
    if (x < nx && y < ny && z < nz) {

        // Read values
        __syncthreads();
        const Float v_p_x = dev_ns_v_p_x[cellidx];
        const Float v_p_y = dev_ns_v_p_y[cellidx];
        const Float v_p_z = dev_ns_v_p_z[cellidx];
        const Float3 v_p = MAKE_FLOAT3(v_p_x, v_p_y, v_p_z);

        const Float epsilon_xn = dev_ns_epsilon[idx(x-1,y,z)];
        const Float epsilon_c  = dev_ns_epsilon[idx(x,y,z)];
        const Float epsilon_yn = dev_ns_epsilon[idx(x,y-1,z)];
        const Float epsilon_zn = dev_ns_epsilon[idx(x,y,z-1)];

        const Float phi_xn = dev_ns_phi[idx(x-1,y,z)];
        const Float phi_c  = dev_ns_phi[idx(x,y,z)];
        const Float phi_yn = dev_ns_phi[idx(x,y-1,z)];
        const Float phi_zn = dev_ns_phi[idx(x,y,z-1)];

        const Float3 phi = MAKE_FLOAT3(
            amean(phi_c, phi_xn),
            amean(phi_c, phi_yn),
            amean(phi_c, phi_zn));

        // Find corrector gradient
        const Float3 grad_epsilon = MAKE_FLOAT3(
            (epsilon_c - epsilon_xn)/dx,
            (epsilon_c - epsilon_yn)/dy,
            (epsilon_c - epsilon_zn)/dz);

        // Find new velocity
        Float3 v = v_p
#ifdef SET_1
            - c_v*ndem*devC_dt/(phi*rho_f)*grad_epsilon;
#endif
#ifdef SET_2
            - c_v*ndem*devC_dt/rho_f*grad_epsilon;
#endif

        if ((z == 0 && bc_bot == 1) || (z > nz-1 && bc_top == 1))
            v.z = 0.0;

        if ((z == 0 && bc_bot == 2) || (z > nz-1 && bc_top == 2))
            v = MAKE_FLOAT3(0.0, 0.0, 0.0);

        // Do not calculate all components at the outer grid edges (nx, ny, nz)
        if (x == nx) {
            v.y = 0.0;
            v.z = 0.0;
        }
        if (y == ny) {
            v.x = 0.0;
            v.z = 0.0;
        }
        if (z == nz) {
            v.x = 0.0;
            v.y = 0.0;
        }

        // Set velocities to zero above the dynamic wall
        if (z >= wall0_iz)
            v = MAKE_FLOAT3(0.0, 0.0, 0.0);

#ifdef REPORT_V_C_COMPONENTS
        printf("[%d,%d,%d] REPORT_V_C_COMPONENTS\n"
                "\tv_p           = % f\t% f\t% f\n"
                "\tv_c           = % f\t% f\t% f\n"
                "\tv             = % f\t% f\t% f\n"
                "\tgrad(epsilon) = % f\t% f\t% f\n"
                "\tphi           = % f\t% f\t% f\n",
                x, y, z,
                v_p.x, v_p.y, v_p.z,
                v.x-v_p.x, v.y-v_p.y, v.z-v_p.z,
                v.x, v.y, v.z,
                grad_epsilon.x, grad_epsilon.y, grad_epsilon.z,
                phi.x, phi.y, phi.z);
#endif

        // Check the advection term using the Courant-Friedrichs-Lewy condition
        __syncthreads();
        if (v.x*ndem*devC_dt/dx
            + v.y*ndem*devC_dt/dy
            + v.z*ndem*devC_dt/dz > 1.0) {
            printf("[%d,%d,%d] Warning: Advection term in fluid may be "
                   "unstable (CFL condition)\n"
                   "\tv = %.2e,%.2e,%.2e\n"
                   "\te_c,e_xn,e_yn,e_zn = %.2e,%.2e,%.2e,%.2e\n",
                   x,y,z, v.x, v.y, v.z,
                   epsilon_c, epsilon_xn, epsilon_yn, epsilon_zn
                   );
        }

        // Write new values
        __syncthreads();
        dev_ns_v_x[cellidx] = v.x;
        dev_ns_v_y[cellidx] = v.y;
        dev_ns_v_z[cellidx] = v.z;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("v", x, y, z, v);
#endif
    }
}

// Find the average particle diameter and velocity for each CFD cell.
// UNUSED: The values are estimated in the porosity estimation function instead
__global__ void findAvgParticleVelocityDiameter(
    const unsigned int* __restrict__ dev_cellStart, // in
    const unsigned int* __restrict__ dev_cellEnd,   // in
    const Float4* __restrict__ dev_vel_sorted,      // in
    const Float4* __restrict__ dev_x_sorted,        // in
    Float3* dev_ns_vp_avg,       // out
    Float*  dev_ns_d_avg)        // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // check that we are not outside the fluid grid
    if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {

        Float4 v;
        Float d;
        unsigned int startIdx, endIdx, i;
        unsigned int n = 0;

        // average particle velocity
        Float3 v_avg = MAKE_FLOAT3(0.0, 0.0, 0.0);  

        // average particle diameter
        Float d_avg = 0.0;

        const unsigned int cellID = x + y * devC_grid.num[0]
            + (devC_grid.num[0] * devC_grid.num[1]) * z; 

        // Lowest particle index in cell
        startIdx = dev_cellStart[cellID];

        // Make sure cell is not empty
        if (startIdx != 0xffffffff) {

            // Highest particle index in cell
            endIdx = dev_cellEnd[cellID];

            // Iterate over cell particles
            for (i=startIdx; i<endIdx; ++i) {

                // Read particle velocity
                __syncthreads();
                v = dev_vel_sorted[i];
                d = 2.0*dev_x_sorted[i].w;
                n++;
                v_avg += MAKE_FLOAT3(v.x, v.y, v.z);
                d_avg += d;
            }

            v_avg /= n;
            d_avg /= n;
        }

        // save average radius and velocity
        const unsigned int cellidx = idx(x,y,z);
        __syncthreads();
        dev_ns_vp_avg[cellidx] = v_avg;
        dev_ns_d_avg[cellidx]  = d_avg;

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("v_avg", x, y, z, v_avg);
        checkFiniteFloat("d_avg", x, y, z, d_avg);
#endif
    }
}

// Find the drag coefficient as dictated by the Reynold's number
// Shamy and Zeghal (2005).
__device__ Float dragCoefficient(Float re)
{
    Float cd;
    if (re >= 1000.0)
        cd = 0.44;
    else
        cd = 24.0/re*(1.0 + 0.15*pow(re, 0.687));
    return cd;
}

// Find particle-fluid interaction force as outlined by Zhou et al. 2010, and
// originally by Gidaspow 1992.
__global__ void findInteractionForce(
    const Float4* __restrict__ dev_x,           // in
    const Float4* __restrict__ dev_vel,         // in
    const Float*  __restrict__ dev_ns_phi,      // in
    const Float*  __restrict__ dev_ns_p,        // in
    const Float*  __restrict__ dev_ns_v_x,      // in
    const Float*  __restrict__ dev_ns_v_y,      // in
    const Float*  __restrict__ dev_ns_v_z,      // in
    const Float*  __restrict__ dev_ns_div_tau_x,// in
    const Float*  __restrict__ dev_ns_div_tau_y,// in
    const Float*  __restrict__ dev_ns_div_tau_z,// in
    const Float mu,                             // in
    const Float rho_f,                          // in
    //const Float c_v,                       // in
    Float3* __restrict__ dev_ns_f_pf,     // out
    Float4* __restrict__ dev_force,       // out
    Float4* __restrict__ dev_ns_f_d,      // out
    Float4* __restrict__ dev_ns_f_p,      // out
    Float4* __restrict__ dev_ns_f_v,      // out
    Float4* __restrict__ dev_ns_f_sum)    // out
{
    unsigned int i = threadIdx.x + blockIdx.x*blockDim.x; // Particle index

    if (i < devC_np) {

        // Particle information
        __syncthreads();
        const Float4 x   = dev_x[i];
        const Float4 vel = dev_vel[i];
        const Float3 v_p = MAKE_FLOAT3(vel.x, vel.y, vel.z);
        const Float  d = x.w;

        // Fluid cell
        const unsigned int i_x =
            floor((x.x - devC_grid.origo[0])/(devC_grid.L[0]/devC_grid.num[0]));
        const unsigned int i_y =
            floor((x.y - devC_grid.origo[1])/(devC_grid.L[1]/devC_grid.num[1]));
        const unsigned int i_z =
            floor((x.z - devC_grid.origo[2])/(devC_grid.L[2]/devC_grid.num[2]));
        const unsigned int cellidx = idx(i_x, i_y, i_z);

        // Cell sizes
        const Float dx = devC_grid.L[0]/devC_grid.num[0];
        const Float dy = devC_grid.L[1]/devC_grid.num[1];
        const Float dz = devC_grid.L[2]/devC_grid.num[2];

        // Fluid information
        __syncthreads();
        const Float phi = dev_ns_phi[cellidx];

        const Float v_x   = dev_ns_v_x[vidx(i_x,i_y,i_z)];
        const Float v_x_p = dev_ns_v_x[vidx(i_x+1,i_y,i_z)];
        const Float v_y   = dev_ns_v_y[vidx(i_x,i_y,i_z)];
        const Float v_y_p = dev_ns_v_y[vidx(i_x,i_y+1,i_z)];
        const Float v_z   = dev_ns_v_z[vidx(i_x,i_y,i_z)];
        const Float v_z_p = dev_ns_v_z[vidx(i_x,i_y,i_z+1)];

        const Float div_tau_x   = dev_ns_div_tau_x[vidx(i_x,i_y,i_z)];
        const Float div_tau_x_p = dev_ns_div_tau_x[vidx(i_x+1,i_y,i_z)];
        const Float div_tau_y   = dev_ns_div_tau_y[vidx(i_x,i_y,i_z)];
        const Float div_tau_y_p = dev_ns_div_tau_y[vidx(i_x,i_y+1,i_z)];
        const Float div_tau_z   = dev_ns_div_tau_z[vidx(i_x,i_y,i_z)];
        const Float div_tau_z_p = dev_ns_div_tau_z[vidx(i_x,i_y,i_z+1)];

        const Float3 v_f = MAKE_FLOAT3(
            amean(v_x, v_x_p),
            amean(v_y, v_y_p),
            amean(v_z, v_z_p));

        const Float3 div_tau = MAKE_FLOAT3(
            amean(div_tau_x, div_tau_x_p),
            amean(div_tau_y, div_tau_y_p),
            amean(div_tau_z, div_tau_z_p));

        const Float3 v_rel = v_f - v_p;
        const Float  v_rel_length = length(v_rel);

        const Float V_p = dx*dy*dz - phi*dx*dy*dz;
        const Float Re  = rho_f*d*phi*v_rel_length/mu;
        Float Cd  = pow(0.63 + 4.8/pow(Re, 0.5), 2.0);
        Float chi = 3.7 - 0.65*exp(-pow(1.5 - log10(Re), 2.0)/2.0);

        if (v_rel_length < 1.0e-6) { // avoid Re=0 -> Cd=inf, chi=-nan
            Cd = 0.0;
            chi = 0.0;
        }

        // Drag force
        const Float3 f_d = 0.125*Cd*rho_f*M_PI*d*d*phi*phi
            *v_rel_length*v_rel*pow(phi, -chi);

        // Pressure gradient force
        const Float3 f_p =
            -1.0*gradient(dev_ns_p, i_x, i_y, i_z, dx, dy, dz)*V_p;

        // Viscous force
        const Float3 f_v = -1.0*div_tau*V_p;

        // Interaction force on particle (force) and fluid (f_pf)
        __syncthreads();
        const Float3 f_pf = f_d + f_p + f_v;

#ifdef CHECK_FLUID_FINITE
        /*
          printf("\nfindInteractionForce %d [%d,%d,%d]\n"
          "\tV_p = %f Re=%f Cd=%f chi=%f\n"
          "\tf_d = %+e %+e %+e\n"
          "\tf_p = %+e %+e %+e\n"
          "\tf_v = %+e %+e %+e\n",
          i, i_x, i_y, i_z, V_p, Re, Cd, chi,
          f_d.x, f_d.y, f_d.z,
          f_p.x, f_p.y, f_p.z,
          f_v.x, f_v.y, f_v.z);// */
        checkFiniteFloat3("f_d", i_x, i_y, i_z, f_d);
        checkFiniteFloat3("f_p", i_x, i_y, i_z, f_p);
        checkFiniteFloat3("f_v", i_x, i_y, i_z, f_v);
        checkFiniteFloat3("f_pf", i_x, i_y, i_z, f_pf);
#endif

        __syncthreads();
#ifdef SET_1
        dev_ns_f_pf[i] = f_pf;
#endif

#ifdef SET_2
        dev_ns_f_pf[i] = f_d;
#endif

        __syncthreads();
        dev_force[i] += MAKE_FLOAT4(f_pf.x, f_pf.y, f_pf.z, 0.0);
        dev_ns_f_d[i] = MAKE_FLOAT4(f_d.x, f_d.y, f_d.z, 0.0);
        dev_ns_f_p[i] = MAKE_FLOAT4(f_p.x, f_p.y, f_p.z, 0.0);
        dev_ns_f_v[i] = MAKE_FLOAT4(f_v.x, f_v.y, f_v.z, 0.0);
        dev_ns_f_sum[i] = MAKE_FLOAT4(
                f_d.x + f_p.x + f_v.x,
                f_d.y + f_p.y + f_v.y,
                f_d.z + f_p.z + f_v.z,
                0.0);
    }
}

// Apply the fluid-particle interaction force to the fluid cell based on the
// interaction forces from each particle in it
__global__ void applyInteractionForceToFluid(
    const unsigned int* __restrict__ dev_gridParticleIndex,    // in
    const unsigned int* __restrict__ dev_cellStart,            // in
    const unsigned int* __restrict__ dev_cellEnd,              // in
    const Float3* __restrict__ dev_ns_f_pf,                    // in
    Float3* __restrict__ dev_ns_F_pf)                    // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Check that we are not outside the fluid grid
    if (x < nx && y < ny && z < nz) {

        const unsigned int cellidx = idx(x,y,z);

        // Calculate linear cell ID
        const unsigned int cellID = x + y * devC_grid.num[0]
            + (devC_grid.num[0] * devC_grid.num[1]) * z; 

        const unsigned int startidx = dev_cellStart[cellID];
        unsigned int endidx, i, origidx;

        Float3 fi;

        if (startidx != 0xffffffff) {

            __syncthreads();
            endidx = dev_cellEnd[cellID];

            for (i=startidx; i<endidx; ++i) {

                __syncthreads();
                origidx = dev_gridParticleIndex[i];
                fi += dev_ns_f_pf[origidx];
            }
        }

        const Float3 F_pf = fi/(dx*dy*dz);

#ifdef CHECK_FLUID_FINITE
        checkFiniteFloat3("F_pf", x, y, z, F_pf);
#endif
        //printf("F_pf [%d,%d,%d] = %f,%f,%f\n", x,y,z, F_pf.x, F_pf.y, F_pf.z);
        __syncthreads();
        dev_ns_F_pf[cellidx] = F_pf;
    }
}


// Launch per cell face node.
// Cell center ghost nodes must be set prior to call.
__global__ void interpolateCenterToFace(
    const Float3* __restrict__ dev_in,
    Float* __restrict__ dev_out_x,
    Float* __restrict__ dev_out_y,
    Float* __restrict__ dev_out_z)
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Check that we are not outside the fluid grid
    if (x <= devC_grid.num[0]
        && y <= devC_grid.num[1]
        && z <= devC_grid.num[2]) {

        const unsigned int faceidx = vidx(x,y,z);

        __syncthreads();
        const Float3 xn = dev_in[idx(x-1,y,z)];
        const Float3 yn = dev_in[idx(x,y-1,z)];
        const Float3 zn = dev_in[idx(x,y,z-1)];
        const Float3 center = dev_in[idx(x,y,z)];

        const Float x_val = amean(center.x, xn.x);
        const Float y_val = amean(center.y, yn.y);
        const Float z_val = amean(center.z, zn.z);

        __syncthreads();
        //printf("c2f [%d,%d,%d] = %f,%f,%f\n", x,y,z, x_val, y_val, z_val);
        dev_out_x[faceidx] = x_val;
        dev_out_y[faceidx] = y_val;
        dev_out_z[faceidx] = z_val;
    }
}

// Launch per cell center node
__global__ void interpolateFaceToCenter(
    Float*  dev_in_x,
    Float*  dev_in_y,
    Float*  dev_in_z,
    Float3* dev_out)
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
    //if (x < devC_grid.num[0] && y < devC_grid.num[1] && z < devC_grid.num[2]) {
    if (x < nx && y < ny && z < nz) {

        __syncthreads();
        const Float x_n = dev_in_x[vidx(x,y,z)];
        const Float x_p = dev_in_x[vidx(x+1,y,z)];
        const Float y_n = dev_in_y[vidx(x,y,z)];
        const Float y_p = dev_in_y[vidx(x,y+1,z)];
        const Float z_n = dev_in_z[vidx(x,y,z)];
        const Float z_p = dev_in_z[vidx(x,y,z+1)];

        const Float3 val = MAKE_FLOAT3(
            amean(x_n, x_p),
            amean(y_n, y_p),
            amean(z_n, z_p));

        __syncthreads();
        //printf("f2c [%d,%d,%d] = %f, %f, %f\n", x,y,z, val.x, val.y, val.z);
        dev_out[idx(x,y,z)] = val;
    }
}

// Launch per cell face node. Set velocity ghost nodes beforehand.
// Find div(tau) at all cell faces.
// Warning: The grid-corner values will be invalid, along with the non-normal
// components of the ghost nodes
__global__ void findFaceDivTau(
    const Float* __restrict__ dev_ns_v_x,   // in
    const Float* __restrict__ dev_ns_v_y,   // in
    const Float* __restrict__ dev_ns_v_z,   // in
    const Float mu,                         // in
    Float* __restrict__ dev_ns_div_tau_x,   // out
    Float* __restrict__ dev_ns_div_tau_y,   // out
    Float* __restrict__ dev_ns_div_tau_z)   // out
{
    // 3D thread index
    const unsigned int x = blockDim.x * blockIdx.x + threadIdx.x;
    const unsigned int y = blockDim.y * blockIdx.y + threadIdx.y;
    const unsigned int z = blockDim.z * blockIdx.z + threadIdx.z;

    // Grid dimensions
    const unsigned int nx = devC_grid.num[0];
    const unsigned int ny = devC_grid.num[1];
    const unsigned int nz = devC_grid.num[2];

    // Cell sizes
    const Float dx = devC_grid.L[0]/nx;
    const Float dy = devC_grid.L[1]/ny;
    const Float dz = devC_grid.L[2]/nz;

    // Check that we are not outside the fluid grid
    //if (x <= nx && y <= ny && z <= nz) {
    if (x < nx && y < ny && z < nz) {

        const unsigned int faceidx = vidx(x,y,z);

        __syncthreads();
        const Float v_x_xn = dev_ns_v_x[vidx(x-1,y,z)];
        const Float v_x = dev_ns_v_x[faceidx];
        const Float v_x_xp = dev_ns_v_x[vidx(x+1,y,z)];
        const Float v_x_yn = dev_ns_v_x[vidx(x,y-1,z)];
        const Float v_x_yp = dev_ns_v_x[vidx(x,y+1,z)];
        const Float v_x_zn = dev_ns_v_x[vidx(x,y,z-1)];
        const Float v_x_zp = dev_ns_v_x[vidx(x,y,z+1)];

        const Float v_y_xn = dev_ns_v_y[vidx(x-1,y,z)];
        const Float v_y_xp = dev_ns_v_y[vidx(x+1,y,z)];
        const Float v_y_yn = dev_ns_v_y[vidx(x,y-1,z)];
        const Float v_y = dev_ns_v_y[faceidx];
        const Float v_y_yp = dev_ns_v_y[vidx(x,y+1,z)];
        const Float v_y_zn = dev_ns_v_y[vidx(x,y,z-1)];
        const Float v_y_zp = dev_ns_v_y[vidx(x,y,z+1)];

        const Float v_z_xn = dev_ns_v_z[vidx(x-1,y,z)];
        const Float v_z_xp = dev_ns_v_z[vidx(x+1,y,z)];
        const Float v_z_yn = dev_ns_v_z[vidx(x,y-1,z)];
        const Float v_z_yp = dev_ns_v_z[vidx(x,y+1,z)];
        const Float v_z_zn = dev_ns_v_z[vidx(x,y,z-1)];
        const Float v_z = dev_ns_v_z[faceidx];
        const Float v_z_zp = dev_ns_v_z[vidx(x,y,z+1)];

        const Float div_tau_x =
            mu*(
                (v_x_xp - 2.0*v_x + v_x_xn)/(dx*dx) +
                (v_x_yp - 2.0*v_x + v_x_yn)/(dy*dy) +
                (v_x_zp - 2.0*v_x + v_x_zn)/(dz*dz));
        const Float div_tau_y =
            mu*(
                (v_y_xp - 2.0*v_y + v_y_xn)/(dx*dx) +
                (v_y_yp - 2.0*v_y + v_y_yn)/(dy*dy) +
                (v_y_zp - 2.0*v_y + v_y_zn)/(dz*dz));
        const Float div_tau_z =
            mu*(
                (v_z_xp - 2.0*v_z + v_z_xn)/(dx*dx) +
                (v_z_yp - 2.0*v_z + v_z_yn)/(dy*dy) +
                (v_z_zp - 2.0*v_z + v_z_zn)/(dz*dz));

        __syncthreads();
        //printf("div_tau [%d,%d,%d] = %f, %f, %f\n", x,y,z,
        //div_tau_x, div_tau_y, div_tau_z);
        dev_ns_div_tau_x[faceidx] = div_tau_x;
        dev_ns_div_tau_y[faceidx] = div_tau_y;
        dev_ns_div_tau_z[faceidx] = div_tau_z;
    }

}

// Print final heads and free memory
void DEM::endNSdev()
{
    freeNSmemDev();
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
