#include <iostream>
#include "sphere.h"

// MISC. UTILITY FUNCTIONS

// Error handler for CUDA GPU calls. 
// Returns error number, filename and line number containing the error to the
// terminal.  Please refer to CUDA_Toolkit_Reference_Manual.pdf, section
// 4.23.3.3 enum cudaError for error discription. Error enumeration starts from
// 0.
void DEM::diagnostics()
{
    // Retrieve information from device to host and run diagnostic tests
    transferFromGlobalDeviceMemory();
    checkValues();

    // Clean up memory before exiting
    if (fluid == 1 && cfd_solver == 0) {
        freeNSmemDev();
        freeNSmem();
    }
    if (fluid == 1 && cfd_solver == 1) {
        freeDarcyMemDev();
        freeDarcyMem();
    }
    freeGlobalDeviceMemory();
    // CPU memory freed upon object destruction
}

void DEM::checkForCudaErrors(const char* checkpoint_description,
        const int run_diagnostics)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCuda error detected, checkpoint: "
            << checkpoint_description << "\nError string: "
            << cudaGetErrorString(err) << std::endl;

        if (run_diagnostics == 1)
            diagnostics();

        exit(EXIT_FAILURE);
    }
}

void DEM::checkForCudaErrorsIter(const char* checkpoint_description,
        const unsigned int iteration,
        const int run_diagnostics)
{
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        std::cerr << "\nCuda error detected, checkpoint: "
            << checkpoint_description << "\nduring iteration " << iteration
            << "\nError string: " << cudaGetErrorString(err) << std::endl;

        if (run_diagnostics == 1)
            diagnostics();

        exit(EXIT_FAILURE);
    }
}

// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
