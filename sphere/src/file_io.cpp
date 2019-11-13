#include <iostream>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#include "typedefs.h"
#include "datatypes.h"
#include "constants.h"
#include "sphere.h"


// Get the address of the first byte of an object's representation
// See Stroustrup (2008) p. 388
    template<class T>
char* as_bytes(T& i)    // treat a T as a sequence of bytes
{
    // get the address of the first byte of memory used
    // to store the object
    void* addr = &i;

    // treat the object as bytes
    return static_cast<char*>(addr);
}

// Read DEM data from binary file
// Note: Static-size arrays can be bulk-read with e.g.
//   ifs.read(as_bytes(grid.L), sizeof(grid.L))
// while dynamic, and vector arrays (e.g. Float4) must
// be read one value at a time.
void DEM::readbin(const char *target)
{
    using std::cout;  // stdout
    using std::cerr;  // stderr
    using std::endl;  // endline. Implicitly flushes buffer
    unsigned int i;

    // Open input file
    // if target is string:
    // std::ifstream ifs(target.c_str(), std::ios_base::binary);
    std::ifstream ifs(target, std::ios_base::binary);
    if (!ifs) {
        cerr << "Could not read input binary file '"
            << target << "'" << endl;
        exit(1);
    }

    Float version;
    ifs.read(as_bytes(version), sizeof(Float));
    if (version != VERSION) {
        std::cerr << "Error: The input file '" << target << "' is written by "
            "sphere version " << version << ", which is incompatible with this "
            "version (" << VERSION << ")." << std::endl;
        exit(1);
    }

    ifs.read(as_bytes(nd), sizeof(nd));
    ifs.read(as_bytes(np), sizeof(np));

    if (nd != ND) {
        cerr << "Dimensionality mismatch between dataset and this SPHERE "
            "program.\nThe dataset is " << nd 
            << "D, this SPHERE binary is " << ND << "D.\n"
            << "This execution is terminating." << endl;
        exit(-1); // Return unsuccessful exit status
    }

    // Check precision choice
    if (sizeof(Float) != sizeof(double) && sizeof(Float) != sizeof(float)) {
        cerr << "Error! Chosen precision not available. Check datatypes.h\n";
        exit(1);
    }

    // Read time parameters
    ifs.read(as_bytes(time.dt), sizeof(time.dt));
    ifs.read(as_bytes(time.current), sizeof(time.current));
    ifs.read(as_bytes(time.total), sizeof(time.total));
    ifs.read(as_bytes(time.file_dt), sizeof(time.file_dt));
    ifs.read(as_bytes(time.step_count), sizeof(time.step_count));

    // For spatial vectors an array of Float4 vectors is chosen for best fit
    // with GPU memory handling. Vector variable structure: ( x, y, z, <empty>).
    // Indexing starts from 0.

    // Allocate host arrays
    if (verbose == 1)
        cout << "  Allocating host memory:                         ";
    // Allocate more host arrays
    k.x      = new Float4[np];
    k.xyzsum = new Float4[np];
    k.vel    = new Float4[np];
    k.force  = new Float4[np];
    k.angpos = new Float4[np];
    k.angvel = new Float4[np];
    k.torque = new Float4[np];
    k.color  = new int[np];

    e.es_dot = new Float[np];
    e.es     = new Float[np];
    e.ev_dot = new Float[np];
    e.ev     = new Float[np];
    e.p      = new Float[np];

    if (verbose == 1)
        cout << "Done\n";

    if (verbose == 1)
        cout << "  Reading remaining data from input binary:       ";

    // Read grid parameters
    ifs.read(as_bytes(grid.origo), sizeof(grid.origo));
    ifs.read(as_bytes(grid.L), sizeof(grid.L));
    ifs.read(as_bytes(grid.num), sizeof(grid.num));
    ifs.read(as_bytes(grid.periodic), sizeof(grid.periodic));
    ifs.read(as_bytes(grid.adaptive), sizeof(grid.adaptive));

    // Read kinematic values
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.x[i].x), sizeof(Float));
        ifs.read(as_bytes(k.x[i].y), sizeof(Float));
        ifs.read(as_bytes(k.x[i].z), sizeof(Float));
        ifs.read(as_bytes(k.x[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.xyzsum[i].x), sizeof(Float));
        ifs.read(as_bytes(k.xyzsum[i].y), sizeof(Float));
        ifs.read(as_bytes(k.xyzsum[i].z), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.vel[i].x), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].y), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].z), sizeof(Float));
        ifs.read(as_bytes(k.vel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.force[i].x), sizeof(Float));
        ifs.read(as_bytes(k.force[i].y), sizeof(Float));
        ifs.read(as_bytes(k.force[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.force[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.angpos[i].x), sizeof(Float));
        ifs.read(as_bytes(k.angpos[i].y), sizeof(Float));
        ifs.read(as_bytes(k.angpos[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.angpos[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.angvel[i].x), sizeof(Float));
        ifs.read(as_bytes(k.angvel[i].y), sizeof(Float));
        ifs.read(as_bytes(k.angvel[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.angvel[i].w), sizeof(Float));
    }
    for (i = 0; i<np; ++i) {
        ifs.read(as_bytes(k.torque[i].x), sizeof(Float));
        ifs.read(as_bytes(k.torque[i].y), sizeof(Float));
        ifs.read(as_bytes(k.torque[i].z), sizeof(Float));
        //ifs.read(as_bytes(k.torque[i].w), sizeof(Float));
    }

    // Read energies
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.es_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.es[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.ev_dot[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.ev[i]), sizeof(Float));
    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(e.p[i]), sizeof(Float));

    // Read constant parameters
    ifs.read(as_bytes(params.g), sizeof(params.g));
    ifs.read(as_bytes(params.k_n), sizeof(params.k_n));
    ifs.read(as_bytes(params.k_t), sizeof(params.k_t));
    ifs.read(as_bytes(params.k_r), sizeof(params.k_r));
    ifs.read(as_bytes(params.E), sizeof(params.E));
    ifs.read(as_bytes(params.gamma_n), sizeof(params.gamma_n));
    ifs.read(as_bytes(params.gamma_t), sizeof(params.gamma_t));
    ifs.read(as_bytes(params.gamma_r), sizeof(params.gamma_r));
    ifs.read(as_bytes(params.mu_s), sizeof(params.mu_s));
    ifs.read(as_bytes(params.mu_d), sizeof(params.mu_d));
    ifs.read(as_bytes(params.mu_r), sizeof(params.mu_r));
    ifs.read(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
    ifs.read(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
    ifs.read(as_bytes(params.mu_ws), sizeof(params.mu_s));
    ifs.read(as_bytes(params.mu_wd), sizeof(params.mu_d));
    ifs.read(as_bytes(params.rho), sizeof(params.rho));
    ifs.read(as_bytes(params.contactmodel), sizeof(params.contactmodel));
    ifs.read(as_bytes(params.kappa), sizeof(params.kappa));
    ifs.read(as_bytes(params.db), sizeof(params.db));
    ifs.read(as_bytes(params.V_b), sizeof(params.V_b));

    // Read wall parameters
    ifs.read(as_bytes(walls.nw), sizeof(walls.nw));
    if (walls.nw > MAXWALLS) {
        cerr << "Error; MAXWALLS (" << MAXWALLS << ") in datatypes.h "
            << "is smaller than the number of walls specified in the "
            << "input file (" << walls.nw << ").\n";
        exit(1);
    }

    // Allocate host memory for walls
    // Wall normal (x,y,z), w: wall position on axis parallel to wall normal
    // Wall mass (x), velocity (y), force (z), and deviatoric stress (w)
    walls.nx    = new Float4[walls.nw];
    walls.mvfd  = new Float4[walls.nw];
    walls.tau_x = new Float[1];

    for (i = 0; i<walls.nw; ++i)
        ifs.read(as_bytes(walls.wmode[i]), sizeof(walls.wmode[i]));
    for (i = 0; i<walls.nw; ++i) {
        ifs.read(as_bytes(walls.nx[i].x), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].y), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].z), sizeof(Float));
        ifs.read(as_bytes(walls.nx[i].w), sizeof(Float));
    }
    for (i = 0; i<walls.nw; ++i) {
        ifs.read(as_bytes(walls.mvfd[i].x), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].y), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].z), sizeof(Float));
        ifs.read(as_bytes(walls.mvfd[i].w), sizeof(Float));
    }
    ifs.read(as_bytes(params.sigma0_A), sizeof(params.sigma0_A));
    ifs.read(as_bytes(params.sigma0_f), sizeof(params.sigma0_f));
    ifs.read(as_bytes(walls.tau_x[0]), sizeof(walls.tau_x[0]));

    // Read bond parameters
    ifs.read(as_bytes(params.lambda_bar), sizeof(params.lambda_bar));
    ifs.read(as_bytes(params.nb0), sizeof(params.nb0));
    ifs.read(as_bytes(params.sigma_b), sizeof(params.sigma_b));
    ifs.read(as_bytes(params.tau_b), sizeof(params.tau_b));
    if (params.nb0 > 0) 
        k.bonds = new uint2[params.nb0];
    k.bonds_delta = new Float4[np];
    k.bonds_omega = new Float4[np];
    for (i = 0; i<params.nb0; ++i) {
        ifs.read(as_bytes(k.bonds[i].x), sizeof(unsigned int));
        ifs.read(as_bytes(k.bonds[i].y), sizeof(unsigned int));
    }
    for (i = 0; i<params.nb0; ++i)   // Normal component
        ifs.read(as_bytes(k.bonds_delta[i].w), sizeof(Float));
    for (i = 0; i<params.nb0; ++i) { // Tangential component
        ifs.read(as_bytes(k.bonds_delta[i].x), sizeof(Float));
        ifs.read(as_bytes(k.bonds_delta[i].y), sizeof(Float));
        ifs.read(as_bytes(k.bonds_delta[i].z), sizeof(Float));
    }
    for (i = 0; i<params.nb0; ++i)   // Normal component
        ifs.read(as_bytes(k.bonds_omega[i].w), sizeof(Float));
    for (i = 0; i<params.nb0; ++i) { // Tangential component
        ifs.read(as_bytes(k.bonds_omega[i].x), sizeof(Float));
        ifs.read(as_bytes(k.bonds_omega[i].y), sizeof(Float));
        ifs.read(as_bytes(k.bonds_omega[i].z), sizeof(Float));
    }

    unsigned int x, y, z;

    if (verbose == 1)
        cout << "Done\n";

    // Simulate fluid
    if (fluid == 1) {

        ifs.read(as_bytes(cfd_solver), sizeof(int));

        if (cfd_solver < 0 || cfd_solver > 1) {
            std::cerr << "Value of cfd_solver not understood ("
                << cfd_solver << ")" << std::endl;
            exit(1);
        }

        if (cfd_solver == 0) {    // Navier Stokes flow

            initNSmem();

            ifs.read(as_bytes(ns.mu), sizeof(Float));

            if (verbose == 1)
                cout << "  - Reading fluid values:\t\t\t  ";

            for (z = 0; z<grid.num[2]; ++z) {
                for (y = 0; y<grid.num[1]; ++y) {
                    for (x = 0; x<grid.num[0]; ++x) {
                        i = idx(x,y,z);
                        ifs.read(as_bytes(ns.v[i].x), sizeof(Float));
                        ifs.read(as_bytes(ns.v[i].y), sizeof(Float));
                        ifs.read(as_bytes(ns.v[i].z), sizeof(Float));
                        ifs.read(as_bytes(ns.p[i]), sizeof(Float));
                        ifs.read(as_bytes(ns.phi[i]), sizeof(Float));
                        ifs.read(as_bytes(ns.dphi[i]), sizeof(Float));
                    }
                }
            }

            ifs.read(as_bytes(ns.rho_f), sizeof(Float));
            ifs.read(as_bytes(ns.p_mod_A), sizeof(Float));
            ifs.read(as_bytes(ns.p_mod_f), sizeof(Float));
            ifs.read(as_bytes(ns.p_mod_phi), sizeof(Float));

            ifs.read(as_bytes(ns.bc_top), sizeof(int));
            ifs.read(as_bytes(ns.bc_bot), sizeof(int));
            ifs.read(as_bytes(ns.free_slip_bot), sizeof(int));
            ifs.read(as_bytes(ns.free_slip_top), sizeof(int));
            ifs.read(as_bytes(ns.bc_bot_flux), sizeof(Float));
            ifs.read(as_bytes(ns.bc_top_flux), sizeof(Float));

            for (z = 0; z<grid.num[2]; ++z)
                for (y = 0; y<grid.num[1]; ++y)
                    for (x = 0; x<grid.num[0]; ++x)
                        ifs.read(as_bytes(ns.p_constant[idx(x,y,z)]),
                                sizeof(int));

            ifs.read(as_bytes(ns.gamma), sizeof(Float));
            ifs.read(as_bytes(ns.theta), sizeof(Float));
            ifs.read(as_bytes(ns.beta), sizeof(Float));
            ifs.read(as_bytes(ns.tolerance), sizeof(Float));
            ifs.read(as_bytes(ns.maxiter), sizeof(unsigned int));
            ifs.read(as_bytes(ns.ndem), sizeof(unsigned int));

            ifs.read(as_bytes(ns.c_phi), sizeof(Float));
            ifs.read(as_bytes(ns.c_v), sizeof(Float));
            ifs.read(as_bytes(ns.dt_dem_fac), sizeof(Float));

            for (i = 0; i<np; ++i) {
                ifs.read(as_bytes(ns.f_d[i].x), sizeof(Float));
                ifs.read(as_bytes(ns.f_d[i].y), sizeof(Float));
                ifs.read(as_bytes(ns.f_d[i].z), sizeof(Float));
            }
            for (i = 0; i<np; ++i) {
                ifs.read(as_bytes(ns.f_p[i].x), sizeof(Float));
                ifs.read(as_bytes(ns.f_p[i].y), sizeof(Float));
                ifs.read(as_bytes(ns.f_p[i].z), sizeof(Float));
            }
            for (i = 0; i<np; ++i) {
                ifs.read(as_bytes(ns.f_v[i].x), sizeof(Float));
                ifs.read(as_bytes(ns.f_v[i].y), sizeof(Float));
                ifs.read(as_bytes(ns.f_v[i].z), sizeof(Float));
            }
            for (i = 0; i<np; ++i) {
                ifs.read(as_bytes(ns.f_sum[i].x), sizeof(Float));
                ifs.read(as_bytes(ns.f_sum[i].y), sizeof(Float));
                ifs.read(as_bytes(ns.f_sum[i].z), sizeof(Float));
            }

            if (verbose == 1)
                cout << "Done" << std::endl;

        } else if (cfd_solver == 1) { // Darcy flow

            initDarcyMem();

            ifs.read(as_bytes(darcy.mu), sizeof(Float));

            if (verbose == 1)
                cout << "  - Reading fluid values:\t\t\t  ";

            for (z = 0; z<darcy.nz; ++z) {
                for (y = 0; y<darcy.ny; ++y) {
                    for (x = 0; x<darcy.nx; ++x) {
                        i = d_idx(x,y,z);
                        ifs.read(as_bytes(darcy.v[i].x), sizeof(Float));
                        ifs.read(as_bytes(darcy.v[i].y), sizeof(Float));
                        ifs.read(as_bytes(darcy.v[i].z), sizeof(Float));
                        ifs.read(as_bytes(darcy.p[i]), sizeof(Float));
                        ifs.read(as_bytes(darcy.phi[i]), sizeof(Float));
                        ifs.read(as_bytes(darcy.dphi[i]), sizeof(Float));
                    }
                }
            }

            ifs.read(as_bytes(darcy.rho_f), sizeof(Float));
            ifs.read(as_bytes(darcy.p_mod_A), sizeof(Float));
            ifs.read(as_bytes(darcy.p_mod_f), sizeof(Float));
            ifs.read(as_bytes(darcy.p_mod_phi), sizeof(Float));

            ifs.read(as_bytes(darcy.bc_xn), sizeof(int));
            ifs.read(as_bytes(darcy.bc_xp), sizeof(int));
            ifs.read(as_bytes(darcy.bc_yn), sizeof(int));
            ifs.read(as_bytes(darcy.bc_yp), sizeof(int));
            ifs.read(as_bytes(darcy.bc_bot), sizeof(int));
            ifs.read(as_bytes(darcy.bc_top), sizeof(int));
            ifs.read(as_bytes(darcy.free_slip_bot), sizeof(int));
            ifs.read(as_bytes(darcy.free_slip_top), sizeof(int));
            ifs.read(as_bytes(darcy.bc_bot_flux), sizeof(Float));
            ifs.read(as_bytes(darcy.bc_top_flux), sizeof(Float));

            for (z = 0; z<darcy.nz; ++z)
                for (y = 0; y<darcy.ny; ++y)
                    for (x = 0; x<darcy.nx; ++x)
                        ifs.read(as_bytes(darcy.p_constant[d_idx(x,y,z)]),
                                sizeof(int));

            ifs.read(as_bytes(darcy.tolerance), sizeof(Float));
            ifs.read(as_bytes(darcy.maxiter), sizeof(unsigned int));
            ifs.read(as_bytes(darcy.ndem), sizeof(unsigned int));
            ifs.read(as_bytes(darcy.c_phi), sizeof(Float));

            for (i = 0; i<np; ++i) {
                ifs.read(as_bytes(darcy.f_p[i].x), sizeof(Float));
                ifs.read(as_bytes(darcy.f_p[i].y), sizeof(Float));
                ifs.read(as_bytes(darcy.f_p[i].z), sizeof(Float));
            }

            ifs.read(as_bytes(darcy.beta_f), sizeof(Float));
            ifs.read(as_bytes(darcy.k_c), sizeof(Float));

            if (verbose == 1)
                cout << "Done" << std::endl;
        }
    }

    for (i = 0; i<np; ++i)
        ifs.read(as_bytes(k.color[i]), sizeof(int));

    // Close file if it is still open
    if (ifs.is_open())
        ifs.close();
}

// Write DEM data to binary file
void DEM::writebin(const char *target)
{
    unsigned int i;

    // Open output file
    std::ofstream ofs(target, std::ios_base::binary);
    if (!ofs) {
        std::cerr << "could create output binary file '"
            << target << "'" << std::endl;
        exit(1);
    }

    // If double precision: Values can be written directly
    if (sizeof(Float) == sizeof(double)) {

        double version = VERSION;
        ofs.write(as_bytes(version), sizeof(Float));

        ofs.write(as_bytes(nd), sizeof(nd));
        ofs.write(as_bytes(np), sizeof(np));

        // Write time parameters
        ofs.write(as_bytes(time.dt), sizeof(time.dt));
        ofs.write(as_bytes(time.current), sizeof(time.current));
        ofs.write(as_bytes(time.total), sizeof(time.total));
        ofs.write(as_bytes(time.file_dt), sizeof(time.file_dt));
        ofs.write(as_bytes(time.step_count), sizeof(time.step_count));

        // Write grid parameters
        ofs.write(as_bytes(grid.origo), sizeof(grid.origo));
        ofs.write(as_bytes(grid.L), sizeof(grid.L));
        ofs.write(as_bytes(grid.num), sizeof(grid.num));
        ofs.write(as_bytes(grid.periodic), sizeof(grid.periodic));
        ofs.write(as_bytes(grid.adaptive), sizeof(grid.adaptive));

        // Write kinematic values
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.x[i].x), sizeof(Float));
            ofs.write(as_bytes(k.x[i].y), sizeof(Float));
            ofs.write(as_bytes(k.x[i].z), sizeof(Float));
            ofs.write(as_bytes(k.x[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.xyzsum[i].x), sizeof(Float));
            ofs.write(as_bytes(k.xyzsum[i].y), sizeof(Float));
            ofs.write(as_bytes(k.xyzsum[i].z), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.vel[i].x), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].y), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].z), sizeof(Float));
            ofs.write(as_bytes(k.vel[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.force[i].x), sizeof(Float));
            ofs.write(as_bytes(k.force[i].y), sizeof(Float));
            ofs.write(as_bytes(k.force[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.force[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.angpos[i].x), sizeof(Float));
            ofs.write(as_bytes(k.angpos[i].y), sizeof(Float));
            ofs.write(as_bytes(k.angpos[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.angpos[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.angvel[i].x), sizeof(Float));
            ofs.write(as_bytes(k.angvel[i].y), sizeof(Float));
            ofs.write(as_bytes(k.angvel[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.angvel[i].w), sizeof(Float));
        }
        for (i = 0; i<np; ++i) {
            ofs.write(as_bytes(k.torque[i].x), sizeof(Float));
            ofs.write(as_bytes(k.torque[i].y), sizeof(Float));
            ofs.write(as_bytes(k.torque[i].z), sizeof(Float));
            //ofs.write(as_bytes(k.torque[i].w), sizeof(Float));
        }

        // Write energies
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.es_dot[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.es[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.ev_dot[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.ev[i]), sizeof(Float));
        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(e.p[i]), sizeof(Float));

        // Write constant parameters
        ofs.write(as_bytes(params.g), sizeof(params.g));
        ofs.write(as_bytes(params.k_n), sizeof(params.k_n));
        ofs.write(as_bytes(params.k_t), sizeof(params.k_t));
        ofs.write(as_bytes(params.k_r), sizeof(params.k_r));
        ofs.write(as_bytes(params.E), sizeof(params.E));
        ofs.write(as_bytes(params.gamma_n), sizeof(params.gamma_n));
        ofs.write(as_bytes(params.gamma_t), sizeof(params.gamma_t));
        ofs.write(as_bytes(params.gamma_r), sizeof(params.gamma_r));
        ofs.write(as_bytes(params.mu_s), sizeof(params.mu_s));
        ofs.write(as_bytes(params.mu_d), sizeof(params.mu_d));
        ofs.write(as_bytes(params.mu_r), sizeof(params.mu_r));
        ofs.write(as_bytes(params.gamma_wn), sizeof(params.gamma_wn));
        ofs.write(as_bytes(params.gamma_wt), sizeof(params.gamma_wt));
        ofs.write(as_bytes(params.mu_ws), sizeof(params.mu_ws));
        ofs.write(as_bytes(params.mu_wd), sizeof(params.mu_wd));
        ofs.write(as_bytes(params.rho), sizeof(params.rho));
        ofs.write(as_bytes(params.contactmodel), sizeof(params.contactmodel));
        ofs.write(as_bytes(params.kappa), sizeof(params.kappa));
        ofs.write(as_bytes(params.db), sizeof(params.db));
        ofs.write(as_bytes(params.V_b), sizeof(params.V_b));

        // Write wall parameters
        ofs.write(as_bytes(walls.nw), sizeof(walls.nw));
        ofs.write(as_bytes(walls.wmode), sizeof(walls.wmode[0])*walls.nw);
        for (i = 0; i<walls.nw; ++i) {
            ofs.write(as_bytes(walls.nx[i].x), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].y), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].z), sizeof(Float));
            ofs.write(as_bytes(walls.nx[i].w), sizeof(Float));
        }
        for (i = 0; i<walls.nw; ++i) {
            ofs.write(as_bytes(walls.mvfd[i].x), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].y), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].z), sizeof(Float));
            ofs.write(as_bytes(walls.mvfd[i].w), sizeof(Float));
        }
        ofs.write(as_bytes(params.sigma0_A), sizeof(params.sigma0_A));
        ofs.write(as_bytes(params.sigma0_f), sizeof(params.sigma0_f));
        ofs.write(as_bytes(walls.tau_x[0]), sizeof(walls.tau_x[0]));

        // Write bond parameters
        ofs.write(as_bytes(params.lambda_bar), sizeof(params.lambda_bar));
        ofs.write(as_bytes(params.nb0), sizeof(params.nb0));
        ofs.write(as_bytes(params.sigma_b), sizeof(params.sigma_b));
        ofs.write(as_bytes(params.tau_b), sizeof(params.tau_b));
        for (i = 0; i<params.nb0; ++i) {
            ofs.write(as_bytes(k.bonds[i].x), sizeof(unsigned int));
            ofs.write(as_bytes(k.bonds[i].y), sizeof(unsigned int));
        }
        for (i = 0; i<params.nb0; ++i)   // Normal component
            ofs.write(as_bytes(k.bonds_delta[i].w), sizeof(Float));
        for (i = 0; i<params.nb0; ++i) { // Tangential component
            ofs.write(as_bytes(k.bonds_delta[i].x), sizeof(Float));
            ofs.write(as_bytes(k.bonds_delta[i].y), sizeof(Float));
            ofs.write(as_bytes(k.bonds_delta[i].z), sizeof(Float));
        }
        for (i = 0; i<params.nb0; ++i)   // Normal component
            ofs.write(as_bytes(k.bonds_omega[i].w), sizeof(Float));
        for (i = 0; i<params.nb0; ++i) { // Tangential component
            ofs.write(as_bytes(k.bonds_omega[i].x), sizeof(Float));
            ofs.write(as_bytes(k.bonds_omega[i].y), sizeof(Float));
            ofs.write(as_bytes(k.bonds_omega[i].z), sizeof(Float));
        }

        if (fluid == 1) {

            ofs.write(as_bytes(cfd_solver), sizeof(int));

            if (cfd_solver == 0) { // Navier Stokes flow

                ofs.write(as_bytes(ns.mu), sizeof(Float));

                int x, y, z;
                for (z=0; z<ns.nz; z++) {
                    for (y=0; y<ns.ny; y++) {
                        for (x=0; x<ns.nx; x++) {
                            i = idx(x,y,z);

                            // Interpolated cell-center velocities
                            ofs.write(as_bytes(ns.v[i].x), sizeof(Float));
                            ofs.write(as_bytes(ns.v[i].y), sizeof(Float));
                            ofs.write(as_bytes(ns.v[i].z), sizeof(Float));

                            // Cell-face velocities
                            //ofs.write(as_bytes(ns.v_x[vidx(x,y,z)]), sizeof(Float));
                            //ofs.write(as_bytes(ns.v_y[vidx(x,y,z)]), sizeof(Float));
                            //ofs.write(as_bytes(ns.v_z[vidx(x,y,z)]), sizeof(Float));

                            ofs.write(as_bytes(ns.p[i]), sizeof(Float));
                            ofs.write(as_bytes(ns.phi[i]), sizeof(Float));
                            ofs.write(as_bytes(ns.dphi[i]), sizeof(Float));
                        }
                    }
                }

                ofs.write(as_bytes(ns.rho_f), sizeof(Float));
                ofs.write(as_bytes(ns.p_mod_A), sizeof(Float));
                ofs.write(as_bytes(ns.p_mod_f), sizeof(Float));
                ofs.write(as_bytes(ns.p_mod_phi), sizeof(Float));

                ofs.write(as_bytes(ns.bc_bot), sizeof(int));
                ofs.write(as_bytes(ns.bc_top), sizeof(int));
                ofs.write(as_bytes(ns.free_slip_bot), sizeof(int));
                ofs.write(as_bytes(ns.free_slip_top), sizeof(int));
                ofs.write(as_bytes(ns.bc_bot_flux), sizeof(Float));
                ofs.write(as_bytes(ns.bc_top_flux), sizeof(Float));

                for (z = 0; z<ns.nz; ++z)
                    for (y = 0; y<ns.ny; ++y)
                        for (x = 0; x<ns.nx; ++x)
                            ofs.write(as_bytes(ns.p_constant[idx(x,y,z)]),
                                    sizeof(int));

                ofs.write(as_bytes(ns.gamma), sizeof(Float));
                ofs.write(as_bytes(ns.theta), sizeof(Float));
                ofs.write(as_bytes(ns.beta), sizeof(Float));
                ofs.write(as_bytes(ns.tolerance), sizeof(Float));
                ofs.write(as_bytes(ns.maxiter), sizeof(unsigned int));
                ofs.write(as_bytes(ns.ndem), sizeof(unsigned int));

                ofs.write(as_bytes(ns.c_phi), sizeof(Float));
                ofs.write(as_bytes(ns.c_v), sizeof(Float));
                ofs.write(as_bytes(ns.dt_dem_fac), sizeof(Float));

                for (i = 0; i<np; ++i) {
                    ofs.write(as_bytes(ns.f_d[i].x), sizeof(Float));
                    ofs.write(as_bytes(ns.f_d[i].y), sizeof(Float));
                    ofs.write(as_bytes(ns.f_d[i].z), sizeof(Float));
                }
                for (i = 0; i<np; ++i) {
                    ofs.write(as_bytes(ns.f_p[i].x), sizeof(Float));
                    ofs.write(as_bytes(ns.f_p[i].y), sizeof(Float));
                    ofs.write(as_bytes(ns.f_p[i].z), sizeof(Float));
                }
                for (i = 0; i<np; ++i) {
                    ofs.write(as_bytes(ns.f_v[i].x), sizeof(Float));
                    ofs.write(as_bytes(ns.f_v[i].y), sizeof(Float));
                    ofs.write(as_bytes(ns.f_v[i].z), sizeof(Float));
                }
                for (i = 0; i<np; ++i) {
                    ofs.write(as_bytes(ns.f_sum[i].x), sizeof(Float));
                    ofs.write(as_bytes(ns.f_sum[i].y), sizeof(Float));
                    ofs.write(as_bytes(ns.f_sum[i].z), sizeof(Float));
                }

            } else if (cfd_solver == 1) {    // Darcy flow

                ofs.write(as_bytes(darcy.mu), sizeof(Float));

                int x, y, z;
                for (z=0; z<darcy.nz; z++) {
                    for (y=0; y<darcy.ny; y++) {
                        for (x=0; x<darcy.nx; x++) {
                            i = d_idx(x,y,z);

                            // Interpolated cell-center velocities
                            ofs.write(as_bytes(darcy.v[i].x), sizeof(Float));
                            ofs.write(as_bytes(darcy.v[i].y), sizeof(Float));
                            ofs.write(as_bytes(darcy.v[i].z), sizeof(Float));

                            // Cell-face velocities
                            //ofs.write(as_bytes(darcy.v_x[vidx(x,y,z)]), sizeof(Float));
                            //ofs.write(as_bytes(darcy.v_y[vidx(x,y,z)]), sizeof(Float));
                            //ofs.write(as_bytes(darcy.v_z[vidx(x,y,z)]), sizeof(Float));

                            ofs.write(as_bytes(darcy.p[i]), sizeof(Float));
                            ofs.write(as_bytes(darcy.phi[i]), sizeof(Float));
                            ofs.write(as_bytes(darcy.dphi[i]), sizeof(Float));
                        }
                    }
                }

                ofs.write(as_bytes(darcy.rho_f), sizeof(Float));
                ofs.write(as_bytes(darcy.p_mod_A), sizeof(Float));
                ofs.write(as_bytes(darcy.p_mod_f), sizeof(Float));
                ofs.write(as_bytes(darcy.p_mod_phi), sizeof(Float));

                ofs.write(as_bytes(darcy.bc_xn), sizeof(int));
                ofs.write(as_bytes(darcy.bc_xp), sizeof(int));
                ofs.write(as_bytes(darcy.bc_yn), sizeof(int));
                ofs.write(as_bytes(darcy.bc_yp), sizeof(int));
                ofs.write(as_bytes(darcy.bc_bot), sizeof(int));
                ofs.write(as_bytes(darcy.bc_top), sizeof(int));
                ofs.write(as_bytes(darcy.free_slip_bot), sizeof(int));
                ofs.write(as_bytes(darcy.free_slip_top), sizeof(int));
                ofs.write(as_bytes(darcy.bc_bot_flux), sizeof(Float));
                ofs.write(as_bytes(darcy.bc_top_flux), sizeof(Float));

                for (z = 0; z<darcy.nz; ++z)
                    for (y = 0; y<darcy.ny; ++y)
                        for (x = 0; x<darcy.nx; ++x)
                            ofs.write(as_bytes(darcy.p_constant[d_idx(x,y,z)]),
                                    sizeof(int));

                ofs.write(as_bytes(darcy.tolerance), sizeof(Float));
                ofs.write(as_bytes(darcy.maxiter), sizeof(unsigned int));
                ofs.write(as_bytes(darcy.ndem), sizeof(unsigned int));
                ofs.write(as_bytes(darcy.c_phi), sizeof(Float));

                for (i = 0; i<np; ++i) {
                    ofs.write(as_bytes(darcy.f_p[i].x), sizeof(Float));
                    ofs.write(as_bytes(darcy.f_p[i].y), sizeof(Float));
                    ofs.write(as_bytes(darcy.f_p[i].z), sizeof(Float));
                }
                ofs.write(as_bytes(darcy.beta_f), sizeof(Float));
                ofs.write(as_bytes(darcy.k_c), sizeof(Float));
            }
        }

        for (i = 0; i<np; ++i)
            ofs.write(as_bytes(k.color[i]), sizeof(int));

        // Close file if it is still open
        if (ofs.is_open())
            ofs.close();

    } else {
        std::cerr << "Can't write output when in single precision mode.\n";
        exit(1);
    }
}

// Write image structure to PPM file
void DEM::writePPM(const char *target)
{
    // Open output file
    std::ofstream ofs(target);
    if (!ofs) {
        std::cerr << "Could not create output PPM file '"
            << target << std::endl;
        exit(1); // Return unsuccessful exit status
    }

    if (verbose == 1)
        std::cout << "  Saving image: " << target << std::endl;

    // Write PPM header
    ofs << "P6 " << width << " " << height << " 255\n";

    // Write pixel array to ppm image file
    for (unsigned int i=0; i<height*width; ++i)
        ofs << img[i].r << img[i].g << img[i].b;

    // Close file if it is still open
    if (ofs.is_open())
        ofs.close();
}

// Write write depth vs. porosity values to file
void DEM::writePorosities(
        const char *target,
        const int z_slices,
        const Float *z_pos,
        const Float *porosity)
{
    // Open output file
    std::ofstream ofs(target);
    if (!ofs) {
        std::cerr << "Could not create output porosity file '"
            << target << std::endl;
        exit(1); // Return unsuccessful exit status
    }

    if (verbose == 1)
        std::cout << "  Saving porosities: " << target << std::endl;

    // Write pixel array to ppm image file
    for (int i=0; i<z_slices; ++i)
        ofs << z_pos[i] << '\t' << porosity[i] << '\n';

    // Close file if it is still open
    if (ofs.is_open())
        ofs.close();
}



// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
