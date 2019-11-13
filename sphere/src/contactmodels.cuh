#ifndef CONTACTMODELS_CUH_
#define CONTACTMODELS_CUH_

// contactmodels.cuh
// Functions governing repulsive forces between contacts


// Linear viscoelastic contact model for particle-wall interactions
// with tangential friction and rolling resistance
__device__ Float contactLinear_wall(
    Float3* F,
    Float3* T,
    Float* es_dot,
    Float* ev_dot,
    Float* p,
    const unsigned int idx_a,
    const Float radius_a,
    const Float4* __restrict__ dev_vel_sorted,
    const Float4* __restrict__ dev_angvel_sorted,
    const Float3 n,
    const Float delta,
    const Float wvel)
{
    // Fetch particle velocities from global memory
    Float4 vel_tmp = dev_vel_sorted[idx_a];
    Float4 angvel_tmp = dev_angvel_sorted[idx_a];

    // Convert velocities to three-component vectors
    Float3 vel_linear = MAKE_FLOAT3(
        vel_tmp.x,
        vel_tmp.y,
        vel_tmp.z);
    Float3 angvel = MAKE_FLOAT3(
        angvel_tmp.x,
        angvel_tmp.y,
        angvel_tmp.z);

    // Store the length of the angular velocity for later use
    Float angvel_length = length(angvel);

    // Contact velocity is the sum of the linear and
    // rotational components
    Float3 vel = vel_linear + (radius_a + delta/2.0) * cross(n, angvel) + wvel;

    // Normal component of the contact velocity
    Float vel_n = dot(vel_linear, n);

    // The tangential velocity is the contact velocity
    // with the normal component subtracted
    const Float3 vel_t = vel - n * (dot(n, vel));
    const Float vel_t_length = length(vel_t);

    // Use scale-invariant contact model (Ergenzinger et al 2010, Obermayr et al 
    // 2013) based on Young's modulus if params.E > 0.0.
    // Use the calculated stiffness for normal and tangential components.
    Float k_n = devC_params.k_n;
    if (devC_params.E > .001) {
        k_n = 3.141592654/2.0 * devC_params.E * radius_a;
    }

    // Normal force component: Elastic - viscous damping
    Float3 f_n = fmax(0.0, -k_n*delta
                      - devC_params.gamma_wn*vel_n) * n;
    const Float f_n_length = length(f_n); // Save length for later use

    // Store the energy lost by viscous damping. See derivation in
    // contactLinear()
    *ev_dot += devC_params.gamma_wn * vel_n * vel_n;

    // Initialize vectors
    Float3 f_t = MAKE_FLOAT3(0.0, 0.0, 0.0);

    // Check that the tangential velocity is high enough to avoid
    // divide by zero (producing a NaN)
    if (vel_t_length > 0.0 && devC_params.gamma_wt > 0.0) {

        // Tangential force by viscous model
        const Float3 f_t_visc = devC_params.gamma_wt * vel_t;
        const Float f_t_visc_length = length(f_t_visc);

        // Determine max. friction
        Float f_t_limit;
        if (vel_t_length > 0.0) { // Dynamic
            f_t_limit = devC_params.mu_wd * f_n_length;
        } else { // Static
            f_t_limit = devC_params.mu_ws * f_n_length;
        }

        // If the shear force component exceeds the friction,
        // the particle slips and energy is dissipated
        if (f_t_visc_length < f_t_limit) {
            f_t = -1.0*f_t_visc;

        } else { // Dynamic friction, friction failure
            f_t = -f_t_limit * vel_t/vel_t_length;

            // Shear energy production rate [W]
            //*es_dot += -dot(vel_t, f_t);
            *es_dot += length(length(f_t) * vel_t * devC_dt) / devC_dt;
        }
    }

    // Total force from wall
    *F += f_n + f_t;

    // Total torque from wall
    *T += cross(-(radius_a + delta*0.5)*n, f_t);

    // Pressure excerted onto particle from this contact
    *p += f_n_length / (4.0f * PI * radius_a*radius_a);

    // Return force excerted onto the wall
    return dot(f_n, n);
}


// Linear vicoelastic contact model for particle-particle interactions
// with tangential friction and rolling resistance
__device__ void contactLinearViscous(
    Float3* F,
    Float3* T, 
    Float* es_dot,
    Float* ev_dot,
    Float* p,
    const unsigned int idx_a,
    const unsigned int idx_b, 
    const Float4* __restrict__ dev_vel_sorted, 
    const Float4* __restrict__ dev_angvel_sorted,
    const Float radius_a,
    const Float radius_b, 
    const Float3 x_ab,
    const Float x_ab_length, 
    const Float delta_ab,
    const Float kappa) 
{

    // Allocate variables and fetch missing time=t values for particle A and B
    Float4 vel_a     = dev_vel_sorted[idx_a];
    Float4 vel_b     = dev_vel_sorted[idx_b];
    Float4 angvel4_a = dev_angvel_sorted[idx_a];
    Float4 angvel4_b = dev_angvel_sorted[idx_b];

    // Convert to Float3's
    Float3 angvel_a = MAKE_FLOAT3(angvel4_a.x, angvel4_a.y, angvel4_a.z);
    Float3 angvel_b = MAKE_FLOAT3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

    // Force between grain pair decomposed into normal- and tangential part
    Float3 f_n, f_t, f_c;

    // Normal vector of contact
    Float3 n_ab = x_ab/x_ab_length;

    // Relative contact interface velocity, w/o rolling
    Float3 vel_ab_linear = MAKE_FLOAT3(vel_a.x - vel_b.x, 
                                       vel_a.y - vel_b.y, 
                                       vel_a.z - vel_b.z);

    // Relative contact interface velocity of particle surfaces at
    // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
    Float3 vel_ab = vel_ab_linear
        + (radius_a + delta_ab/2.0f) * cross(n_ab, angvel_a)
        + (radius_b + delta_ab/2.0f) * cross(n_ab, angvel_b);

    // Relative contact interface rolling velocity
    Float3 angvel_ab = angvel_a - angvel_b;
    Float  angvel_ab_length = length(angvel_ab);

    // Normal component of the relative contact interface velocity
    Float vel_n_ab = dot(vel_ab_linear, n_ab);

    // Tangential component of the relative contact interface velocity
    // Hinrichsen and Wolf 2004, eq. 13.9
    Float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));
    Float  vel_t_ab_length = length(vel_t_ab);

    // Use scale-invariant contact model (Ergenzinger et al 2010, Obermayr et al 
    // 2013) based on Young's modulus if params.E > 0.0.
    // Use the calculated stiffness for normal and tangential components.
    Float k_n = devC_params.k_n;
    if (devC_params.E > .001) {
        k_n = 3.141592654/2.0 * devC_params.E * (radius_a + radius_b)/2.;
    }

    // Normal force component: Elastic - viscous damping
    f_n = fmax(0.0, -k_n * delta_ab
               - devC_params.gamma_n * vel_n_ab) * n_ab;

    // Make sure the viscous damping doesn't exceed the elastic component,
    // i.e. the damping factor doesn't exceed the critical damping, 2*sqrt(m*k_n)
    if (dot(f_n, n_ab) < 0.0f)
        f_n = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

    Float f_n_length = length(f_n);

    // Add max. capillary force
    f_c = -kappa * sqrtf(radius_a * radius_b) * n_ab;

    // Initialize force vectors to zero
    f_t = MAKE_FLOAT3(0.0, 0.0, 0.0);

    // Shear force component: Nonlinear relation
    // Coulomb's law of friction limits the tangential force to less or equal
    // to the normal force
    if (vel_t_ab_length > 0.0) {

        // Tangential force by viscous model
        Float f_t_visc  = devC_params.gamma_t * vel_t_ab_length;

        // Determine max. friction
        Float f_t_limit;
        if (vel_t_ab_length > 0.001f) { // Dynamic
            f_t_limit = devC_params.mu_d * length(f_n-f_c);
        } else { // Static
            f_t_limit = devC_params.mu_s * length(f_n-f_c);
        }

        // If the shear force component exceeds the friction,
        // the particle slips and energy is dissipated
        if (f_t_visc < f_t_limit) { // Static
            f_t = -f_t_visc * vel_t_ab/vel_t_ab_length;

        } else { // Dynamic, friction failure
            f_t = -f_t_limit * vel_t_ab/vel_t_ab_length;

            // Shear friction production rate [W]
            //*es_dot += -dot(vel_t_ab, f_t);
        }
    }

    // Add force components from this collision to total force for particle
    *F += f_n + f_t + f_c; 
    *T += -(radius_a + delta_ab/2.0f) * cross(n_ab, f_t);

    // Pressure excerted onto the particle from this contact
    *p += f_n_length / (4.0f * PI * radius_a*radius_a);

} // End of contactLinearViscous()


// Linear elastic contact model for particle-particle interactions
__device__ void contactLinear(
    Float3* F,
    Float3* T, 
    Float* es_dot,
    Float* ev_dot,
    Float* p,
    const unsigned int idx_a_orig,
    const unsigned int idx_b_orig, 
    const Float4  vel_a, 
    const Float4* __restrict__ dev_vel,
    const Float3  angvel_a,
    const Float4* __restrict__ dev_angvel,
    const Float radius_a,
    const Float radius_b, 
    const Float3 x,
    const Float x_length, 
    const Float delta,
    Float4* __restrict__ dev_delta_t,
    const unsigned int mempos) 
{

    // Allocate variables and fetch missing time=t values for particle A and B
    Float4 vel_b     = dev_vel[idx_b_orig];
    Float4 angvel4_b = dev_angvel[idx_b_orig];

    // Fetch previous sum of shear displacement for the contact pair
    Float4 delta_t0_4 = dev_delta_t[mempos];

    Float3 delta_t0_uncor = MAKE_FLOAT3(
        delta_t0_4.x,
        delta_t0_4.y,
        delta_t0_4.z);

    // Convert to Float3
    Float3 angvel_b = MAKE_FLOAT3(
        angvel4_b.x,
        angvel4_b.y,
        angvel4_b.z);

    // Force between grain pair decomposed into normal- and tangential part
    Float3 f_n, f_t, f_c;

    // Normal vector of contact
    Float3 n = x / x_length;

    // Relative contact interface velocity, w/o rolling
    Float3 vel_linear = MAKE_FLOAT3(
        vel_a.x - vel_b.x, 
        vel_a.y - vel_b.y, 
        vel_a.z - vel_b.z);

    // Relative contact interface velocity of particle surfaces at
    // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10,
    // or Luding 2008, eq. 10)
    Float3 vel = vel_linear
        + (radius_a + delta/2.0) * cross(n, angvel_a)
        + (radius_b + delta/2.0) * cross(n, angvel_b);

    // Relative contact interface rolling velocity
    Float3 angvel = angvel_a - angvel_b;
    Float  angvel_length = length(angvel);

    // Normal component of the relative contact interface velocity
    Float vel_n = -dot(vel_linear, n);

    // Tangential component of the relative contact interface velocity
    // Hinrichsen and Wolf 2004, eq. 13.9
    Float3 vel_t = vel - n * dot(n, vel);
    Float  vel_t_length = length(vel_t);

    // Correct tangential displacement vector, which is
    // necessary if the tangential plane rotated
    Float3 delta_t0 = delta_t0_uncor - (n * dot(n, delta_t0_uncor));
    Float  delta_t0_length = length(delta_t0);

    // New tangential displacement vector
    Float3 delta_t;

    // Use scale-invariant contact model (Ergenzinger et al 2010, Obermayr et al 
    // 2013) based on Young's modulus if params.E > 0.0.
    // Use the calculated stiffness for normal and tangential components.
    Float k_n = devC_params.k_n;
    Float k_t = devC_params.k_t;
    if (devC_params.E > .001) {
        k_n = 3.141592654/2.0 * devC_params.E * (radius_a + radius_b)/2.;
        k_t = k_n;
    }

    // Normal force component: Elastic - viscous damping
    f_n = fmax(0.0, -k_n*delta + devC_params.gamma_n * vel_n) * n;
    Float f_n_length = length(f_n);

    // Store energy dissipated in normal viscous component
    // watt = gamma_n * vel_n * dx_n / dt
    // watt = gamma_n * vel_n * vel_n * dt / dt
    // watt = gamma_n * vel_n * vel_n
    // watt = N*m/s = N*s/m * m/s * m/s * s / s
    *ev_dot += 0.5 * devC_params.gamma_n * vel_n * vel_n;

    // Add max. capillary force
    f_c = -devC_params.kappa * sqrtf(radius_a * radius_b) * n;

    // Initialize force vectors to zero
    f_t   = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

    // Apply a tangential force if the previous tangential displacement
    // is non-zero, or the current sliding velocity is non-zero.
    if (delta_t0_length > 0.0 || vel_t_length > 0.0) {

        // Add tangential displacement to total tangential displacement
        delta_t = delta_t0 + vel_t * devC_dt;

        // Tangential force: Visco-Elastic, before limitation criterion
        Float3 f_t_elast = -k_t * delta_t;
        Float3 f_t_visc  = -devC_params.gamma_t * vel_t;
        f_t = f_t_elast + f_t_visc;
        Float f_t_length = length(f_t);

        // Static frictional limit
        Float f_t_limit = devC_params.mu_s * length(f_n-f_c);

        // Store energy dissipated in tangential viscous component
        *ev_dot += 0.5 * devC_params.gamma_t * vel_t_length * vel_t_length;

        // If failure criterion is not met, contact is viscous-linear elastic.
        // If failure criterion is met, contact force is limited, 
        // resulting in a slip and energy dissipation
        if (f_t_length > f_t_limit) { // Static friciton exceeded: Dynamic case

            // tangential vector
            Float3 t = f_t/length(f_t);

            // Frictional force is reduced to equal the dynamic limit
            f_t = f_t_limit * t;

            // In the sliding friction case, the tangential spring is adjusted
            // to a length consistent with Coulombs (dynamic) condition (Luding
            // 2008)
            delta_t = -1.0/k_t
                * (devC_params.mu_d * length(f_n-f_c) * t
                   + devC_params.gamma_t * vel_t);

            // Shear friction heat production rate: 
            // The energy lost from the tangential spring is dissipated as heat
            *es_dot += 0.5*length(length(f_t) * vel_t * devC_dt) / devC_dt; // Seen in ESyS-Particle

        }
    }


    // Add force components from this collision to total force for particle
    *F += f_n + f_t + f_c;

    // Add torque components from this collision to total torque for particle
    // Comment out the line below to disable rotation
    *T += cross(-(radius_a + delta*0.5) * n, f_t);

    // Pressure excerted onto the particle from this contact
    *p += f_n_length / (4.0f * PI * radius_a*radius_a);

    // Store sum of tangential displacements
    dev_delta_t[mempos] = MAKE_FLOAT4(
        delta_t.x,
        delta_t.y,
        delta_t.z,
        0.0f);

} // End of contactLinear()


// Non-linear contact model for particle-particle interactions
// Based on Hertzian and Mindlin contact theories (e.g. Hertz, 1882, Mindlin and 
// Deresiewicz, 1953, Johnson, 1985). See Yohannes et al 2012 for example.
__device__ void contactHertz(
    Float3* F,
    Float3* T, 
    Float* es_dot,
    Float* ev_dot,
    Float* p,
    const unsigned int idx_a_orig,
    const unsigned int idx_b_orig, 
    const Float4  vel_a, 
    const Float4* __restrict__ dev_vel,
    const Float3  angvel_a,
    const Float4* __restrict__ dev_angvel,
    const Float radius_a,
    const Float radius_b, 
    const Float3 x_ab,
    const Float x_ab_length, 
    const Float delta_ab,
    Float4* __restrict__ dev_delta_t,
    const unsigned int mempos) 
{

    // Allocate variables and fetch missing time=t values for particle A and B
    Float4 vel_b     = dev_vel[idx_b_orig];
    Float4 angvel4_b = dev_angvel[idx_b_orig];

    // Fetch previous sum of shear displacement for the contact pair
    Float4 delta_t0_4 = dev_delta_t[mempos];

    Float3 delta_t0_uncor = MAKE_FLOAT3(delta_t0_4.x,
                                        delta_t0_4.y,
                                        delta_t0_4.z);

    // Convert to Float3
    Float3 angvel_b = MAKE_FLOAT3(angvel4_b.x, angvel4_b.y, angvel4_b.z);

    // Force between grain pair decomposed into normal- and tangential part
    Float3 f_n, f_t, f_c, T_res;

    // Normal vector of contact
    Float3 n_ab = x_ab/x_ab_length;

    // Relative contact interface velocity, w/o rolling
    Float3 vel_ab_linear = MAKE_FLOAT3(vel_a.x - vel_b.x, 
                                       vel_a.y - vel_b.y, 
                                       vel_a.z - vel_b.z);

    // Relative contact interface velocity of particle surfaces at
    // the contact, with rolling (Hinrichsen and Wolf 2004, eq. 13.10)
    Float3 vel_ab = vel_ab_linear
        + (radius_a + delta_ab/2.0f) * cross(n_ab, angvel_a)
        + (radius_b + delta_ab/2.0f) * cross(n_ab, angvel_b);

    // Relative contact interface rolling velocity
    Float3 angvel_ab = angvel_a - angvel_b;
    Float  angvel_ab_length = length(angvel_ab);

    // Normal component of the relative contact interface velocity
    Float vel_n_ab = dot(vel_ab_linear, n_ab);

    // Tangential component of the relative contact interface velocity
    // Hinrichsen and Wolf 2004, eq. 13.9
    Float3 vel_t_ab = vel_ab - (n_ab * dot(vel_ab, n_ab));
    Float  vel_t_ab_length = length(vel_t_ab);

    // Correct tangential displacement vector, which is
    // necessary if the tangential plane rotated
    Float3 delta_t0 = delta_t0_uncor - (n_ab * dot(delta_t0_uncor, n_ab));
    Float  delta_t0_length = length(delta_t0);

    // New tangential displacement vector
    Float3 delta_t;

    // Use scale-invariant contact model (Ergenzinger et al 2010, Obermayr et al 
    // 2013) based on Young's modulus if params.E > 0.0.
    // Use the calculated stiffness for normal and tangential components.
    Float k_n = devC_params.k_n;
    Float k_t = devC_params.k_t;
    if (devC_params.E > .001) {
        k_n = 3.141592654/2.0 * devC_params.E * (radius_a + radius_b)/2.;
        k_t = k_n;
    }

    // Normal force component
    f_n = (-k_n * powf(delta_ab, 3.0f/2.0f)  -devC_params.gamma_n * 
           powf(delta_ab, 1.0f/4.0f) * vel_n_ab)
        * n_ab;

    // Store energy dissipated in normal viscous component
    // watt = gamma_n * vel_n * dx_n / dt
    // watt = gamma_n * vel_n * vel_n * dt / dt
    // watt = gamma_n * vel_n * vel_n
    // watt = N*m/s = N*s/m * m/s * m/s * s / s
    *ev_dot += devC_params.gamma_n * vel_n_ab * vel_n_ab;

    // Make sure the viscous damping doesn't exceed the elastic component,
    // i.e. the damping factor doesn't exceed the critical damping, 2*sqrt(m*k_n)
    if (dot(f_n, n_ab) < 0.0f)
        f_n = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

    Float f_n_length = length(f_n);

    // Add max. capillary force
    f_c = -devC_params.kappa * sqrtf(radius_a * radius_b) * n_ab;

    // Initialize force vectors to zero
    f_t   = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);
    T_res = MAKE_FLOAT3(0.0f, 0.0f, 0.0f);

    // Apply a tangential force if the previous tangential displacement
    // is non-zero, or the current sliding velocity is non-zero.
    if (delta_t0_length > 0.f || vel_t_ab_length > 0.f) {

        // Shear force: Visco-Elastic, limited by Coulomb friction
        Float3 f_t_elast = -k_t * powf(delta_ab, 1.0f/2.0f) * delta_t0;
        Float3 f_t_visc  = -devC_params.gamma_t * powf(delta_ab, 1.0f/4.0f) * vel_t_ab;
        Float f_t_limit;

        if (vel_t_ab_length > 0.001f) { // Dynamic friciton
            f_t_limit = devC_params.mu_d * length(f_n-f_c);
        } else { // Static friction
            f_t_limit = devC_params.mu_s * length(f_n-f_c);
        }

        // Tangential force before friction limit correction
        f_t = f_t_elast + f_t_visc;
        Float f_t_length = length(f_t);

        // If failure criterion is not met, contact is viscous-linear elastic.
        // If failure criterion is met, contact force is limited, 
        // resulting in a slip and energy dissipation
        if (f_t_length > f_t_limit) { // Dynamic case

            // Frictional force is reduced to equal the limit
            f_t *= f_t_limit/f_t_length;

            // A slip event zeros the displacement vector
            //delta_t = make_Float3(0.0f, 0.0f, 0.0f);

            // In a slip event, the tangential spring is adjusted to a 
            // length which is consistent with Coulomb's equation
            // (Hinrichsen and Wolf, 2004)
            delta_t = (f_t + devC_params.gamma_t * powf(delta_ab, 1.0f/4.0f) * vel_t_ab)
                / (-k_t * powf(delta_ab, 1.0f/2.0f));

            // Shear friction heat production rate: 
            // The energy lost from the tangential spring is dissipated as heat
            *es_dot += length(delta_t0 - delta_t) * k_t / devC_dt; // Seen in EsyS-Particle

        } else { // Static case

            // No correction of f_t is required

            // Add tangential displacement to total tangential displacement
            delta_t = delta_t0 + vel_t_ab * devC_dt;
        }
    }

    if (angvel_ab_length > 0.f) {
        // Apply rolling resistance (Zhou et al. 1999)
        //T_res = -angvel_ab/angvel_ab_length * devC_params.mu_r * R_bar * length(f_n);

        // New rolling resistance model
        T_res = -1.0f * fmin(devC_params.gamma_r * radius_a * angvel_ab_length,
                             devC_params.mu_r * radius_a * f_n_length)
            * angvel_ab/angvel_ab_length;
    }

    // Add force components from this collision to total force for particle
    *F += f_n + f_t + f_c; 
    *T += -(radius_a + delta_ab/2.0f) * cross(n_ab, f_t) + T_res;

    // Pressure excerted onto the particle from this contact
    *p += f_n_length / (4.0f * PI * radius_a*radius_a);

    // Store sum of tangential displacements
    dev_delta_t[mempos] = MAKE_FLOAT4(delta_t.x, delta_t.y, delta_t.z, 0.0f);

} // End of contactHertz()


#endif
// vim: tabstop=8 expandtab shiftwidth=4 softtabstop=4
