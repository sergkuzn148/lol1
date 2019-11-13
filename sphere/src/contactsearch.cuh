#ifndef CONTACTSEARCH_CUH_
#define CONTACTSEARCH_CUH_

// contactsearch.cuh
// Functions for identifying contacts and processing boundaries


// Calculate the distance modifier between a contact pair. The modifier
// accounts for periodic boundaries. If the target cell lies outside
// the grid, it returns -1.
// Function is called from overlapsInCell() and findContactsInCell().
__device__ int findDistMod(int3* targetCell, Float3* distmod)
{
    // Check whether x- and y boundaries are to be treated as periodic
    // 1: x- and y boundaries periodic
    // 2: x boundaries periodic
    if (devC_grid.periodic == 1) {

        // Periodic x-boundary
        if (targetCell->x < 0) {
            //targetCell->x = devC_grid.num[0] - 1;
            targetCell->x += devC_grid.num[0];
            *distmod += MAKE_FLOAT3(devC_grid.L[0], 0.0, 0.0);
        }
        if (targetCell->x >= devC_grid.num[0]) {
            //targetCell->x = 0;
            targetCell->x -= devC_grid.num[0];
            *distmod -= MAKE_FLOAT3(devC_grid.L[0], 0.0, 0.0);
        }

        // Periodic y-boundary
        if (targetCell->y < 0) {
            //targetCell->y = devC_grid.num[1] - 1;
            targetCell->y += devC_grid.num[1];
            *distmod += MAKE_FLOAT3(0.0, devC_grid.L[1], 0.0);
        }
        if (targetCell->y >= devC_grid.num[1]) {
            //targetCell->y = 0;
            targetCell->y -= devC_grid.num[1];
            *distmod -= MAKE_FLOAT3(0.0, devC_grid.L[1], 0.0);
        }


    }
    // Only x-boundaries are periodic
    if (devC_grid.periodic == 2) {

        // Periodic x-boundary
        if (targetCell->x < 0) {
            //targetCell->x = devC_grid.num[0] - 1;
            targetCell->x += devC_grid.num[0];
            *distmod += MAKE_FLOAT3(devC_grid.L[0], 0.0, 0.0);
        }
        if (targetCell->x >= devC_grid.num[0]) {
            //targetCell->x = 0;
            targetCell->x -= devC_grid.num[0];
            *distmod -= MAKE_FLOAT3(devC_grid.L[0], 0.0, 0.0);
        }

        // Hande out-of grid cases on y-axis
        if (targetCell->y < 0 || targetCell->y >= devC_grid.num[1])
            return -1;


    }
    // No periodic boundaries
    if (devC_grid.periodic > 2 || devC_grid.periodic < 1) {

        // Don't modify targetCell or distmod.

        // Hande out-of grid cases on x- and y-axes
        if (targetCell->x < 0 || targetCell->x >= devC_grid.num[0])
            return -1;
        if (targetCell->y < 0 || targetCell->y >= devC_grid.num[1])
            return -1;
    }

    // Handle out-of-grid cases on z-axis
    if (targetCell->z < 0 || targetCell->z >= devC_grid.num[2])
        return -1;
    else
        return 0;
}


// Find overlaps between particle no. 'idx' and particles in cell 'gridpos'.
// Contacts are processed as soon as they are identified.
// Used for contactmodel=1, where contact history is not needed.
// Kernel executed on device, and callable from device only.
// Function is called from interact().
__device__ void findAndProcessContactsInCell(
    int3 targetCell, 
    const unsigned int idx_a, 
    const Float4 x_a,
    const Float radius_a,
    Float3* F,
    Float3* T, 
    Float* es_dot,
    Float* ev_dot,
    Float* p,
    const Float4* __restrict__ dev_x_sorted, 
    const Float4* __restrict__ dev_vel_sorted, 
    const Float4* __restrict__ dev_angvel_sorted,
    const unsigned int* __restrict__ dev_cellStart, 
    const unsigned int* __restrict__ dev_cellEnd,
    const Float4* __restrict__ dev_walls_nx, 
    Float4* __restrict__ dev_walls_mvfd)
//uint4 bonds)
{

    // Get distance modifier for interparticle
    // vector, if it crosses a periodic boundary
    Float3 distmod = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    if (findDistMod(&targetCell, &distmod) != -1) {


        //// Check and process particle-particle collisions

        // Calculate linear cell ID
        unsigned int cellID = targetCell.x + targetCell.y * devC_grid.num[0]
            + (devC_grid.num[0] * devC_grid.num[1]) * targetCell.z; 

        // Lowest particle index in cell
        unsigned int startIdx = dev_cellStart[cellID];

        // Make sure cell is not empty
        if (startIdx != 0xffffffff) {

            // Highest particle index in cell + 1
            unsigned int endIdx = dev_cellEnd[cellID];

            // Iterate over cell particles
            for (unsigned int idx_b = startIdx; idx_b<endIdx; ++idx_b) {
                if (idx_b != idx_a) { // Do not collide particle with itself

                    // Fetch position and velocity of particle B.
                    Float4 x_b      = dev_x_sorted[idx_b];
                    Float  radius_b = x_b.w;
                    Float  kappa 	= devC_params.kappa;

                    // Distance between particle centers (Float4 -> Float3)
                    Float3 x_ab = MAKE_FLOAT3(x_a.x - x_b.x, 
                                              x_a.y - x_b.y, 
                                              x_a.z - x_b.z);

                    // Adjust interparticle vector if periodic boundary/boundaries
                    // are crossed
                    x_ab += distmod;
                    Float x_ab_length = length(x_ab);

                    // Distance between particle perimeters
                    Float delta_ab = x_ab_length - (radius_a + radius_b); 

                    // Check for particle overlap
                    if (delta_ab < 0.0f) {
                        contactLinearViscous(F, T, es_dot, ev_dot, p, 
                                             idx_a, idx_b,
                                             dev_vel_sorted, 
                                             dev_angvel_sorted,
                                             radius_a, radius_b, 
                                             x_ab, x_ab_length,
                                             delta_ab, kappa);
                    } else if (delta_ab < devC_params.db) { 
                        // Check wether particle distance satisfies the
                        // capillary bond distance
                        capillaryCohesion_exp(F, radius_a, radius_b, delta_ab, 
                                              x_ab, x_ab_length, kappa);
                    }

                    // Check wether particles are bonded together
                    /*if (bonds.x == idx_b || bonds.y == idx_b ||
                      bonds.z == idx_b || bonds.w == idx_b) {
                      bondLinear(F, T, es_dot, p, % ev_dot missing
                      idx_a, idx_b,
                      dev_x_sorted, dev_vel_sorted,
                      dev_angvel_sorted,
                      radius_a, radius_b,
                      x_ab, x_ab_length,
                      delta_ab);
                      }*/

                } // Do not collide particle with itself end
            } // Iterate over cell particles end
        } // Check wether cell is empty end
    } // Periodic boundary and distance adjustment end
} // End of overlapsInCell(...)


// Find overlaps between particle no. 'idx' and particles in cell 'gridpos'
// Write the indexes of the overlaps in array contacts[].
// Used for contactmodel=2, where bookkeeping of contact history is necessary.
// Kernel executed on device, and callable from device only.
// Function is called from topology().
__device__ void findContactsInCell(
    int3 targetCell, 
    const unsigned int idx_a, 
    const Float4 x_a,
    const Float radius_a,
    const Float4* __restrict__ dev_x_sorted, 
    const unsigned int* __restrict__ dev_cellStart, 
    const unsigned int* __restrict__ dev_cellEnd,
    const unsigned int* __restrict__ dev_gridParticleIndex,
    int* nc,
    unsigned int* __restrict__ dev_contacts,
    Float4* __restrict__ dev_distmod)
{
    // Get distance modifier for interparticle
    // vector, if it crosses a periodic boundary
    Float3 distmod = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    if (findDistMod(&targetCell, &distmod) != -1) {

        __syncthreads();

        //// Check and process particle-particle collisions

        // Calculate linear cell ID
        unsigned int cellID = targetCell.x + targetCell.y * devC_grid.num[0]
            + (devC_grid.num[0] * devC_grid.num[1]) * targetCell.z; 

        // Lowest particle index in cell
        unsigned int startIdx = dev_cellStart[cellID];

        // Make sure cell is not empty
        if (startIdx != 0xffffffff) {

            __syncthreads();

            // Highest particle index in cell + 1
            unsigned int endIdx = dev_cellEnd[cellID];

            // Read the original index of particle A
            unsigned int idx_a_orig = dev_gridParticleIndex[idx_a];

            // Iterate over cell particles
            for (unsigned int idx_b = startIdx; idx_b<endIdx; ++idx_b) {
                if (idx_b != idx_a) { // Do not collide particle with itself


                    // Fetch position and radius of particle B.
                    Float4 x_b      = dev_x_sorted[idx_b];
                    Float  radius_b = x_b.w;

                    // Read the original index of particle B
                    unsigned int idx_b_orig = dev_gridParticleIndex[idx_b];

                    //__syncthreads();

                    // Distance between particle centers (Float4 -> Float3)
                    Float3 x_ab = MAKE_FLOAT3(x_a.x - x_b.x, 
                                              x_a.y - x_b.y, 
                                              x_a.z - x_b.z);

                    // Adjust interparticle vector if periodic boundary/boundaries
                    // are crossed
                    x_ab += distmod;

                    Float x_ab_length = length(x_ab);

                    // Distance between particle perimeters
                    Float delta_ab = x_ab_length - (radius_a + radius_b); 

                    // Check for particle overlap
                    if (delta_ab < 0.0f) {

                        // If the particle is not yet registered in the contact list,
                        // use the next position in the array
                        int cpos = *nc;
                        unsigned int cidx;

                        // Find out, if particle is already registered in contact list
                        for (int i=0; i < devC_nc; ++i) {
                            __syncthreads();
                            cidx = dev_contacts[(unsigned int)(idx_a_orig*devC_nc+i)];
                            if (cidx == devC_np) // Write to position of now-deleted contact
                                cpos = i;
                            else if (cidx == idx_b_orig) { // Write to position of same contact
                                cpos = i;
                                break;
                            }
                        }

                        __syncthreads();

                        // Write the particle index to the relevant position,
                        // no matter if it already is there or not (concurrency of write)
                        dev_contacts[(unsigned int)(idx_a_orig*devC_nc+cpos)] = idx_b_orig;


                        // Write the interparticle vector and radius of particle B
                        //dev_x_ab_r_b[(unsigned int)(idx_a_orig*devC_nc+cpos)] = make_Float4(x_ab, radius_b);
                        dev_distmod[(unsigned int)(idx_a_orig*devC_nc+cpos)] = MAKE_FLOAT4(distmod.x, distmod.y, distmod.z, radius_b);

                        // Increment contact counter
                        ++*nc;

                        // Check if the number of contacts of particle A
                        // exceeds the max. number of contacts per particle
                        if (*nc > devC_nc)
                            return; // I would like to throw an error, but it doesn't seem possible...

                    }

                    // Write the inter-particle position vector correction and radius of particle B
                    //dev_distmod[(unsigned int)(idx_a_orig*devC_nc+cpos)] = make_Float4(distmod, radius_b);

                    // Check wether particles are bonded together
                    /*if (bonds.x == idx_b || bonds.y == idx_b ||
                      bonds.z == idx_b || bonds.w == idx_b) {
                      bondLinear(F, T, es_dot, p, % ev_dot missing
                      idx_a, idx_b,
                      dev_x_sorted, dev_vel_sorted,
                      dev_angvel_sorted,
                      radius_a, radius_b,
                      x_ab, x_ab_length,
                      delta_ab);
                      }*/

                } // Do not collide particle with itself end
            } // Iterate over cell particles end
        } // Check wether cell is empty end
    } // Periodic boundary and distmod end
} // End of findContactsInCell(...)


// For a single particle:
// Search for neighbors to particle 'idx' inside the 27 closest cells, 
// and save the contact pairs in global memory.
// Function is called from mainGPU loop.
__global__ void topology(
    const unsigned int* __restrict__ dev_cellStart, 
    const unsigned int* __restrict__ dev_cellEnd,
    const unsigned int* __restrict__ dev_gridParticleIndex,
    const Float4* __restrict__ dev_x_sorted,
    unsigned int* __restrict__ dev_contacts,
    Float4* __restrict__ dev_distmod)
{
    // Thread index equals index of particle A
    unsigned int idx_a = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx_a < devC_np) {
        // Fetch particle data in global read
        Float4 x_a      = dev_x_sorted[idx_a];
        Float  radius_a = x_a.w;

        // Count the number of contacts in this time step
        int nc = 0;

        // Grid position of host and neighbor cells in uniform, cubic grid
        int3 gridPos;
        int3 targetPos;

        // Calculate cell address in grid from position of particle
        gridPos.x = floor((x_a.x - devC_grid.origo[0]) / (devC_grid.L[0]/devC_grid.num[0]));
        gridPos.y = floor((x_a.y - devC_grid.origo[1]) / (devC_grid.L[1]/devC_grid.num[1]));
        gridPos.z = floor((x_a.z - devC_grid.origo[2]) / (devC_grid.L[2]/devC_grid.num[2]));

        // Find overlaps between particle no. idx and all particles
        // from its own cell + 26 neighbor cells
        for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
            for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis
                    targetPos = gridPos + make_int3(x_dim, y_dim, z_dim);
                    findContactsInCell(targetPos, idx_a, x_a, radius_a,
                                       dev_x_sorted, 
                                       dev_cellStart, dev_cellEnd,
                                       dev_gridParticleIndex,
                                       &nc, dev_contacts, dev_distmod);
                }
            }
        }
    }
} // End of topology(...)


// For a single particle:
// If contactmodel=1:
//   Search for neighbors to particle 'idx' inside the 27 closest cells, 
//   and compute the resulting force and torque on it.
// If contactmodel=2:
//   Process contacts saved in dev_contacts by topology(), and compute
//   the resulting force and torque on it.
// For all contactmodels:
//   Collide with top- and bottom walls, save resulting force on upper wall.
// Kernel is executed on device, and is callable from host only.
// Function is called from mainGPU loop.
__global__ void interact(
    const unsigned int* __restrict__ dev_gridParticleIndex, // in
    const unsigned int* __restrict__ dev_cellStart,         // in
    const unsigned int* __restrict__ dev_cellEnd,           // in
    const Float4* __restrict__ dev_x,                       // in
    const Float4* __restrict__ dev_x_sorted,                // in
    const Float4* __restrict__ dev_vel_sorted,              // in
    const Float4* __restrict__ dev_angvel_sorted,           // in
    const Float4* __restrict__ dev_vel,                     // in
    const Float4* __restrict__ dev_angvel,                  // in
    Float4* __restrict__ dev_force,                         // out
    Float4* __restrict__ dev_torque,                        // out
    Float*  __restrict__ dev_es_dot,                        // out
    Float*  __restrict__ dev_ev_dot,                        // out
    Float*  __restrict__ dev_es,                            // out
    Float*  __restrict__ dev_ev,                            // out
    Float*  __restrict__ dev_p,                             // out
    const Float4* __restrict__ dev_walls_nx,                // in
    Float4* __restrict__ dev_walls_mvfd,                    // in
    Float* __restrict__ dev_walls_force_pp,                 // out
    unsigned int* __restrict__ dev_contacts,                // out
    const Float4* __restrict__ dev_distmod,                 // in
    Float4* __restrict__ dev_delta_t)                       // out
{
    // Thread index equals index of particle A
    unsigned int idx_a = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx_a < devC_np) {

        // Fetch particle data in global read
        unsigned int idx_a_orig = dev_gridParticleIndex[idx_a];
        Float4 x_a      = dev_x_sorted[idx_a];
        Float  radius_a = x_a.w;

        // Fetch world dimensions in constant memory read
        Float3 origo = MAKE_FLOAT3(devC_grid.origo[0], 
                                   devC_grid.origo[1], 
                                   devC_grid.origo[2]); 
        Float3 L = MAKE_FLOAT3(devC_grid.L[0], 
                               devC_grid.L[1], 
                               devC_grid.L[2]);

        // Fetch wall data in global read
        Float4 w_0_nx, w_1_nx, w_2_nx, w_3_nx, w_4_nx;
        Float4 w_0_mvfd, w_1_mvfd, w_2_mvfd, w_3_mvfd, w_4_mvfd;

        // default wall normals and positions
        w_0_nx = MAKE_FLOAT4( 0.0f, 0.0f,-1.0f, L.z);
        w_1_nx = MAKE_FLOAT4(-1.0f, 0.0f, 0.0f, L.x);
        w_2_nx = MAKE_FLOAT4( 1.0f, 0.0f, 0.0f, 0.0f);
        w_3_nx = MAKE_FLOAT4( 0.0f,-1.0f, 0.0f, L.y);
        w_4_nx = MAKE_FLOAT4( 0.0f, 1.0f, 0.0f, 0.0f);

        // default wall mass, vel, force, and sigma0
        w_0_mvfd = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
        w_1_mvfd = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
        w_2_mvfd = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
        w_3_mvfd = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);
        w_4_mvfd = MAKE_FLOAT4(0.0f, 0.0f, 0.0f, 0.0f);

        // fetch data for dynamic walls
        if (devC_nw >= 1) {
            w_0_nx   = dev_walls_nx[0];
            w_0_mvfd = dev_walls_mvfd[0];
            if (devC_nw >= 2) {
                w_1_nx = dev_walls_nx[1];
                w_1_mvfd = dev_walls_mvfd[1];
            }
            if (devC_nw >= 3) {
                w_2_nx = dev_walls_nx[2];
                w_2_mvfd = dev_walls_mvfd[2];
            }
            if (devC_nw >= 4) {
                w_3_nx = dev_walls_nx[3];
                w_3_mvfd = dev_walls_mvfd[3];
            }
            if (devC_nw >= 5) {
                w_4_nx = dev_walls_nx[4];
                w_4_mvfd = dev_walls_mvfd[4];
            }
        }

        // Index of particle which is bonded to particle A.
        // The index is equal to the particle no (p.np)
        // if particle A is bond-less.
        //uint4 bonds = dev_bonds_sorted[idx_a];

        // Initiate shear friction loss rate at 0.0
        Float es_dot = 0.0; 
        Float ev_dot = 0.0;

        // Initiate pressure on particle at 0.0
        Float p = 0.0;

        // Allocate memory for temporal force/torque vector values
        Float3 F = MAKE_FLOAT3(0.0, 0.0, 0.0);
        Float3 T = MAKE_FLOAT3(0.0, 0.0, 0.0);

        // Apply linear elastic, frictional contact model to registered contacts
        if (devC_params.contactmodel == 2 || devC_params.contactmodel == 3) {
            unsigned int idx_b_orig, mempos;
            Float delta_n, x_ab_length, radius_b;
            Float3 x_ab;
            Float4 x_b, distmod;
            Float4 vel_a     = dev_vel_sorted[idx_a];
            Float4 angvel4_a = dev_angvel_sorted[idx_a];
            Float3 angvel_a  = MAKE_FLOAT3(
                angvel4_a.x, angvel4_a.y, angvel4_a.z);

            // Loop over all possible contacts, and remove contacts
            // that no longer are valid (delta_n > 0.0)
            for (int i = 0; i<devC_nc; ++i) {
                mempos = (unsigned int)(idx_a_orig * devC_nc + i);
                __syncthreads();
                idx_b_orig = dev_contacts[mempos];

                if (idx_b_orig != (unsigned int)devC_np) {

                    // Read inter-particle distance correction vector
                    distmod = dev_distmod[mempos];

                    // Read particle b position and radius
                    x_b = dev_x[idx_b_orig];
                    radius_b = x_b.w;

                    // Inter-particle vector, corrected for periodic boundaries
                    x_ab = MAKE_FLOAT3(x_a.x - x_b.x + distmod.x,
                                       x_a.y - x_b.y + distmod.y,
                                       x_a.z - x_b.z + distmod.z);

                    x_ab_length = length(x_ab);
                    delta_n = x_ab_length - (radius_a + radius_b);

                    // Process collision if the particles are overlapping
                    if (delta_n < 0.0) {
                        if (devC_params.contactmodel == 2) {
                            contactLinear(&F, &T, &es_dot, &ev_dot, &p, 
                                          idx_a_orig,
                                          idx_b_orig,
                                          vel_a,
                                          dev_vel,
                                          angvel_a,
                                          dev_angvel,
                                          radius_a, radius_b, 
                                          x_ab, x_ab_length,
                                          delta_n, dev_delta_t, 
                                          mempos);
                        } else if (devC_params.contactmodel == 3) {
                            contactHertz(&F, &T, &es_dot, &ev_dot, &p, 
                                         idx_a_orig,
                                         idx_b_orig,
                                         vel_a,
                                         dev_vel,
                                         angvel_a,
                                         dev_angvel,
                                         radius_a, radius_b, 
                                         x_ab, x_ab_length,
                                         delta_n, dev_delta_t, 
                                         mempos);
                        }
                    } else {
                        __syncthreads();
                        // Remove this contact (there is no particle with
                        // index=np)
                        dev_contacts[mempos] = devC_np;
                        // Zero sum of shear displacement in this position
                        dev_delta_t[mempos] = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
                    }

                } else { // if dev_contacts[mempos] == devC_np
                    __syncthreads();
                    // Zero sum of shear displacement in this position
                    dev_delta_t[mempos] = MAKE_FLOAT4(0.0, 0.0, 0.0, 0.0);
                } 
            } // Contact loop end


            // Find contacts and process collisions immidiately for
            // contactmodel 1 (visco-frictional).
        } else if (devC_params.contactmodel == 1) {

            int3 gridPos;
            int3 targetPos;

            // Calculate address in grid from position
            gridPos.x = floor((x_a.x - devC_grid.origo[0])
                              / (devC_grid.L[0]/devC_grid.num[0]));
            gridPos.y = floor((x_a.y - devC_grid.origo[1])
                              / (devC_grid.L[1]/devC_grid.num[1]));
            gridPos.z = floor((x_a.z - devC_grid.origo[2])
                              / (devC_grid.L[2]/devC_grid.num[2]));

            // Find overlaps between particle no. idx and all particles
            // from its own cell + 26 neighbor cells.
            // Calculate resulting normal- and shear-force components and
            // torque for the particle on the base of contactLinearViscous()
            for (int z_dim=-1; z_dim<2; ++z_dim) { // z-axis
                for (int y_dim=-1; y_dim<2; ++y_dim) { // y-axis
                    for (int x_dim=-1; x_dim<2; ++x_dim) { // x-axis
                        targetPos = gridPos + make_int3(x_dim, y_dim, z_dim);
                        findAndProcessContactsInCell(targetPos, idx_a,
                                                     x_a, radius_a,
                                                     &F, &T, &es_dot,
                                                     &ev_dot, &p,
                                                     dev_x_sorted,
                                                     dev_vel_sorted,
                                                     dev_angvel_sorted,
                                                     dev_cellStart,
                                                     dev_cellEnd,
                                                     dev_walls_nx,
                                                     dev_walls_mvfd);
                    }
                }
            }

        }

        //// Interact with walls
        Float delta_w; // Overlap distance
        Float3 w_n;    // Wall surface normal
        Float w_0_force = 0.0; // Force on wall 0 from particle A
        Float w_1_force = 0.0; // Force on wall 1 from particle A
        Float w_2_force = 0.0; // Force on wall 2 from particle A
        Float w_3_force = 0.0; // Force on wall 3 from particle A
        Float w_4_force = 0.0; // Force on wall 4 from particle A

        // Upper wall (idx 0)
        delta_w = w_0_nx.w - (x_a.z + radius_a);
        w_n = MAKE_FLOAT3(w_0_nx.x, w_0_nx.y, w_0_nx.z);
        if (delta_w < 0.0f) {
            w_0_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p, idx_a,
                                           radius_a, dev_vel_sorted,
                                           dev_angvel_sorted, w_n,
                                           delta_w, w_0_mvfd.y);
        }

        // Lower wall (force on wall not stored)
        delta_w = x_a.z - radius_a - origo.z;
        w_n = MAKE_FLOAT3(0.0f, 0.0f, 1.0f);
        if (delta_w < 0.0f) {
            (void)contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p, idx_a,
                                     radius_a, dev_vel_sorted,
                                     dev_angvel_sorted, w_n, delta_w,
                                     0.0);
        }


        if (devC_grid.periodic == 0) {          // no periodic walls

            // Right wall (idx 1)
            delta_w = w_1_nx.w - (x_a.x + radius_a);
            w_n = MAKE_FLOAT3(w_1_nx.x, w_1_nx.y, w_1_nx.z);
            if (delta_w < 0.0f) {
                w_1_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_1_mvfd.y);
            }

            // Left wall (idx 2)
            delta_w = x_a.x - radius_a - w_2_nx.w;
            w_n = MAKE_FLOAT3(w_2_nx.x, w_2_nx.y, w_2_nx.z);
            if (delta_w < 0.0f) {
                w_2_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_2_mvfd.y);
            }

            // Back wall (idx 3)
            delta_w = w_3_nx.w - (x_a.y + radius_a);
            w_n = MAKE_FLOAT3(w_3_nx.x, w_3_nx.y, w_3_nx.z);
            if (delta_w < 0.0f) {
                w_3_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_3_mvfd.y);
            }

            // Front wall (idx 4)
            delta_w = x_a.y - radius_a - w_4_nx.w;
            w_n = MAKE_FLOAT3(w_4_nx.x, w_4_nx.y, w_4_nx.z);
            if (delta_w < 0.0f) {
                w_4_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_4_mvfd.y);
            }

        } else if (devC_grid.periodic == 2) {   // right and left walls periodic

            // Back wall (idx 3)
            delta_w = w_3_nx.w - (x_a.y + radius_a);
            w_n = MAKE_FLOAT3(w_3_nx.x, w_3_nx.y, w_3_nx.z);
            if (delta_w < 0.0f) {
                w_3_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_3_mvfd.y);
            }

            // Front wall (idx 4)
            delta_w = x_a.y - radius_a - w_4_nx.w;
            w_n = MAKE_FLOAT3(w_4_nx.x, w_4_nx.y, w_4_nx.z);
            if (delta_w < 0.0f) {
                w_4_force = contactLinear_wall(&F, &T, &es_dot, &ev_dot, &p,
                                               idx_a, radius_a,
                                               dev_vel_sorted,
                                               dev_angvel_sorted, w_n,
                                               delta_w, w_4_mvfd.y);
            }
        }

        // Hold threads for coalesced write
        __syncthreads();

        // Write force to unsorted position
        unsigned int orig_idx = dev_gridParticleIndex[idx_a];
        dev_force[orig_idx]   = MAKE_FLOAT4(F.x, F.y, F.z, 0.0f);
        dev_torque[orig_idx]  = MAKE_FLOAT4(T.x, T.y, T.z, 0.0f);
        dev_es_dot[orig_idx]  = es_dot;
        dev_ev_dot[orig_idx]  = ev_dot;
        dev_es[orig_idx]     += es_dot * devC_dt;
        dev_ev[orig_idx]     += ev_dot * devC_dt;
        dev_p[orig_idx]       = p;
        if (devC_nw > 0)
            dev_walls_force_pp[orig_idx] = w_0_force;
        if (devC_nw > 1)
            dev_walls_force_pp[orig_idx+devC_np] = w_1_force;
        if (devC_nw > 2)
            dev_walls_force_pp[orig_idx+devC_np*2] = w_2_force;
        if (devC_nw > 3)
            dev_walls_force_pp[orig_idx+devC_np*3] = w_3_force;
        if (devC_nw > 4)
            dev_walls_force_pp[orig_idx+devC_np*4] = w_4_force;
    }
} // End of interact(...)


#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
