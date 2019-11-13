Fluid simulation and particle-fluid interaction
===============================================
A new and experimental addition to *sphere* is the ability to simulate a mixture
of particles and a Newtonian fluid. The fluid is simulated using an Eulerian
continuum approach, using a custom CUDA solver for GPU computation. This
approach allows for fast simulations due to the limited need for GPU-CPU
communications, as well as a flexible code base.

The following sections will describe the theoretical background, as well as the
solution procedure and the numerical implementation.

Derivation of the Navier Stokes equations with porosity
-------------------------------------------------------
Following the outline presented by `Limache and Idelsohn (2006)`_, the
continuity equation for an incompressible fluid material is given by:

.. math::
    \nabla \cdot \boldsymbol{v} = 0

and the momentum equation:

.. math::
    \rho \frac{\partial \boldsymbol{v}}{\partial t}
    + \rho (\boldsymbol{v} \cdot \nabla \boldsymbol{v})
    = \nabla \cdot \boldsymbol{\sigma}
    - \boldsymbol{f}^i
    + \rho \boldsymbol{g}

Here, :math:`\boldsymbol{v}` is the fluid velocity, :math:`\rho` is the
fluid density, :math:`\boldsymbol{\sigma}` is the `Cauchy stress tensor`_,
:math:`\boldsymbol{f}^i` is the particle-fluid interaction vector and
:math:`\boldsymbol{g}` is the gravitational acceleration. For incompressible
Newtonian fluids, the Cauchy stress is given by:

.. math::
    \boldsymbol{\sigma} = -p \boldsymbol{I} + \boldsymbol{\tau}

:math:`p` is the fluid pressure, :math:`\boldsymbol{I}` is the identity
tensor, and :math:`\boldsymbol{\tau}` is the deviatoric stress tensor, given
by:

.. math::
    \boldsymbol{\tau} =
    \mu_f \nabla \boldsymbol{v}
    + \mu_f (\nabla \boldsymbol{v})^T

By using the following vector identities:

.. math::
    \nabla \cdot (p \boldsymbol{I}) = \nabla p

    \nabla \cdot (\nabla \boldsymbol{v}) = \nabla^2 \boldsymbol{v}

    \nabla \cdot (\nabla \boldsymbol{v})^T
    = \nabla (\nabla \cdot \boldsymbol{v})

the deviatoric component of the Cauchy stress tensor simplifies to the
following, assuming that spatial variations in the viscosity can be neglected:

.. math::
    = -\nabla p
    + \mu_f \nabla^2 \boldsymbol{v}

Since we are dealing with fluid flow in a porous medium, additional terms are
introduced to the equations for conservation of mass and momentum. In the
following, the equations are derived for the first spatial component. The
solution for the other components is trivial.

The porosity value (in the saturated porous medium the volumetric fraction of
the fluid phase) denoted :math:`\phi` is incorporated in the continuity and
momentum equations. The continuity equation becomes:

.. math::
    \frac{\partial \phi}{\partial t}
    + \nabla \cdot (\phi \boldsymbol{v}) = 0

For the :math:`x` component, the Lagrangian formulation of the momentum equation
with a body force :math:`\boldsymbol{f}` becomes:

.. math::
    \frac{D (\phi v_x)}{D t}
    = \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\sigma}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g

In the Eulerian formulation, an advection term is added, and the Cauchy stress
tensor is represented as isotropic and deviatoric components individually:

.. math::
    \frac{\partial (\phi v_x)}{\partial t}
    + \boldsymbol{v} \cdot \nabla (\phi v_x)
    = \frac{1}{\rho} \left[ \nabla \cdot (-\phi p \boldsymbol{I})
    + \phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x

Using vector identities to rewrite the advection term, and expanding the fluid
stress tensor term:

.. math::
    \frac{\partial (\phi v_x)}{\partial t}
    + \nabla \cdot (\phi v_x \boldsymbol{v})
    - \phi v_x (\nabla \cdot \boldsymbol{v})
    = \frac{1}{\rho} \left[ -\nabla \phi p \right]_x
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x

Spatial variations in the porosity are neglected,

.. math::
    \nabla \phi := 0

and the pressure is attributed to the fluid phase alone (model B in Zhu et al.
2007 and Zhou et al. 2010). The divergence of fluid velocities is defined to be
zero:

.. math::
    \nabla \cdot \boldsymbol{v} := 0

With these assumptions, the momentum equation simplifies to:

.. math::
    \frac{\partial (\phi v_x)}{\partial t}
    + \nabla \cdot (\phi v_x \boldsymbol{v})
    = -\frac{1}{\rho} \frac{\partial p}{\partial x}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x

The remaining part of the advection term is for the :math:`x` component
found as:

.. math::
    \nabla \cdot (\phi v_x \boldsymbol{v}) =
    \left[
        \frac{\partial}{\partial x},
        \frac{\partial}{\partial y},
        \frac{\partial}{\partial z}
    \right]
    \left[
        \begin{array}{c}
            \phi v_x v_x\\
            \phi v_x v_y\\
            \phi v_x v_z\\
        \end{array}
    \right]
    =
    \frac{\partial (\phi v_x v_x)}{\partial x} +
    \frac{\partial (\phi v_x v_y)}{\partial y} +
    \frac{\partial (\phi v_x v_z)}{\partial z}

The deviatoric stress tensor is in this case symmetrical, i.e. :math:`\tau_{ij}
= \tau_{ji}`, and is found by:

.. math::
    \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    = \frac{1}{\rho}
    \left[
        \left[
            \frac{\partial}{\partial x},
            \frac{\partial}{\partial y},
            \frac{\partial}{\partial z}
        \right]
        \phi
        \left[
            \begin{matrix}
                \tau_{xx} & \tau_{xy} & \tau_{xz}\\
                \tau_{yx} & \tau_{yy} & \tau_{yz}\\
                \tau_{zx} & \tau_{zy} & \tau_{zz}\\
            \end{matrix}
        \right]
    \right]_x

    = \frac{1}{\rho}
    \left[
        \begin{array}{c}
            \frac{\partial (\phi \tau_{xx})}{\partial x}
            + \frac{\partial (\phi \tau_{xy})}{\partial y}
            + \frac{\partial (\phi \tau_{xz})}{\partial z}\\
            \frac{\partial (\phi \tau_{yx})}{\partial x}
            + \frac{\partial (\phi \tau_{yy})}{\partial y}
            + \frac{\partial (\phi \tau_{yz})}{\partial z}\\
            \frac{\partial (\phi \tau_{zx})}{\partial x}
            + \frac{\partial (\phi \tau_{zy})}{\partial y}
            + \frac{\partial (\phi \tau_{zz})}{\partial z}\\
        \end{array}
    \right]_x
    = \frac{1}{\rho}
    \left(
        \frac{\partial (\phi \tau_{xx})}{\partial x}
        + \frac{\partial (\phi \tau_{xy})}{\partial y}
        + \frac{\partial (\phi \tau_{xz})}{\partial z}
    \right)

In a linear viscous fluid, the stress and strain rate
(:math:`\dot{\boldsymbol{\epsilon}}`) is linearly dependent, scaled by the
viscosity parameter :math:`\mu_f`:

.. math::
    \tau_{ij} = 2 \mu_f \dot{\epsilon}_{ij}
    = \mu_f \left(
    \frac{\partial v_i}{\partial x_j} + \frac{\partial v_j}{\partial x_i}
    \right)

With this relationship, the deviatoric stress tensor components can be
calculated as:

.. math::
    \tau_{xx} = 2 \mu_f \frac{\partial v_x}{\partial x} \qquad
    \tau_{yy} = 2 \mu_f \frac{\partial v_y}{\partial y} \qquad
    \tau_{zz} = 2 \mu_f \frac{\partial v_z}{\partial z}

    \tau_{xy} = \mu_f \left(
    \frac{\partial v_x}{\partial y} + \frac{\partial v_y}{\partial x} \right)

    \tau_{xz} = \mu_f \left(
    \frac{\partial v_x}{\partial z} + \frac{\partial v_z}{\partial x} \right)

    \tau_{yz} = \mu_f \left(
    \frac{\partial v_y}{\partial z} + \frac{\partial v_z}{\partial y} \right)

where :math:`\mu_f` is the dynamic viscosity. The above formulation of the
fluid rheology assumes identical bulk and shear viscosities. The derivation of
the equations for the other spatial components is trivial.

Porosity estimation
-------------------
The solid volume in each fluid cell is determined by the ratio of the
a cell-centered spherical cell volume (:math:`V_c`) and the sum of intersecting
particle volumes (:math:`V_s`). The spherical cell volume has a center at
:math:`\boldsymbol{x}_i`, and a radius of :math:`R_i`, which is equal to half
the fluid cell width. The nearby particles are characterized by position
:math:`\boldsymbol{x}_j` and radius :math:`r_j`. The center distance is defined
as:

.. math::
    d_{ij} = ||\boldsymbol{x}_i - \boldsymbol{x}_j||

The common volume of the two intersecting spheres is zero if the volumes aren't
intersecting, lens shaped if they are intersecting, and spherical if the
particle is fully contained by the spherical cell volume:

.. math::
    V^s_{i} = \sum_j
    \begin{cases}
        0 & \textit{if } R_i + r_j \leq d_{ij} \\
        \frac{1}{12d_{ij}} \left[ \pi (R_i + r_j - d_{ij})^2
        (d_{ij}^2 + 2d_{ij}r_j - 3r_j^2 + 2d_{ij} R_i + 6r_j R_i - 3R_i^2)
        \right] & \textit{if } R_i - r_j < d_{ij} < R_i + r_j \\
        \frac{4}{3} \pi r^3_j & \textit{if } d_{ij} \leq R_i - r_j
    \end{cases}

Using this method, the cell porosity values are continuous through time as
particles enter and exit the cell volume. The rate of porosity change
(:math:`d\phi/dt`) is estimated by the backwards Euler method
by considering the previous and current porosity.

Particle-fluid interaction
--------------------------
The momentum exchange of the granular and fluid phases follows the procedure
outlined by Gidaspow 1992 and Shamy and Zhegal 2005. The fluid and particle
interaction is based on the concept of drag, where the magnitude is based on
semi-empirical relationships. The drag force scales linearly with the relative
difference in velocity between the fluid and particle phase. On the base of
Newton's third law, the resulting drag force is applied with opposite signs to
the particle and fluid.

For fluid cells with porosities (:math:`\phi`) less or equal to 0.8, the drag
force is based on the Ergun (1952) equation:

.. math::
    \bar{\boldsymbol{f}}_d = \left(
    150 \frac{\mu_f (1-\phi)^2}{\phi\bar{d}^2}
    + 1.75 \frac{(1-\phi)\rho_f
      ||\boldsymbol{v}_f - \bar{\boldsymbol{v}}_p||}{\bar{d}}
    \right)
    (\boldsymbol{v}_f - \bar{\boldsymbol{v}}_p)

here, :math:`\bar{d}` denotes the average particle diameter in the cell,
:math:`\boldsymbol{v}_f` is the fluid flow velocity, and
:math:`\bar{\boldsymbol{v}}_p` is the average particle velocity in the cell. All
particles in contact with the previously mentioned cell-centered sphere for
porosity estimation contribute to the average particle velocity and diameter in
the fluid cell.

If the porosity is greater than 0.8, the cell-averaged drag force
(:math:`\bar{\boldsymbol{f}}_d` is found from the Wen and Yu (1966) equation,
which considers the fluid flow situation:

.. math::
    \bar{\boldsymbol{f}}_d = \left(
    \frac{3}{4}
    \frac{C_d (1-\phi) \phi^{-2.65} \mu_f \rho_f
    ||\boldsymbol{v}_f - \bar{\boldsymbol{v}}_p||}{\bar{d}}
    \right)
    (\boldsymbol{v}_f - \bar{\boldsymbol{v}}_p)

The drag coefficient :math:`C_d` is evaluated depending on the magnitude of the
Reynolds number :math:`Re`:

.. math::
    C_d =
    \begin{cases}
    \frac{24}{Re} (1+0.15 (Re)^{0.687} & \textit{if } Re < 1,000 \\
    0.44 & \textit{if } Re \geq 1,000
    \end{cases}

where the Reynold's number is found by:

.. math::
    Re = \frac{\phi\rho_f\bar{d}}{\mu_f}
    ||\boldsymbol{v}_f - \bar{\boldsymbol{v}}_p||

The interaction force is applied to the fluid with negative sign as a
contribution to the body force :math:`\boldsymbol{f}`. The fluid interaction
force applied particles in the fluid cell is:

.. math::
    \boldsymbol{f}_i = \frac{\bar{\boldsymbol{f}}_d V_p}{1-\phi}

where :math:`V_p` denotes the particle volume. Optionally, the above
interaction force could be expanded to include the force induced by the fluid
pressure gradient:

.. math::
    \boldsymbol{f}_i = \left(
    -\nabla p +
    \frac{\bar{\boldsymbol{f}}_d}{1-\phi}
    \right) V_p


Fluid dynamics solution procedure by operator splitting
-------------------------------------------------------
The partial differential terms in the previously described equations are found
using finite central differences. Modifying the operator splitting methodology
presented by Langtangen et al.  (2002), the predicted velocity
:math:`\boldsymbol{v}^*` after a finite time step
:math:`\Delta t` is found by explicit integration of the momentum equation.

.. math::
    \frac{\Delta (\phi v_x)}{\Delta t}
    + \nabla \cdot (\phi v_x \boldsymbol{v})
    = - \frac{1}{\rho} \frac{\Delta p}{\Delta x}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x

    \Downarrow

    \phi \frac{\Delta v_x}{\Delta t}
    + v_x \frac{\Delta \phi}{\Delta t}
    + \nabla \cdot (\phi v_x \boldsymbol{v})
    = - \frac{1}{\rho} \frac{\Delta p}{\Delta x}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x

We want to isolate :math:`\Delta v_x` in the above equation in order to project
the new velocity.

.. math::
    \phi \frac{\Delta v_x}{\Delta t}
    = - \frac{1}{\rho} \frac{\Delta p}{\Delta x}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
    - \frac{1}{\rho} f^i_x
    + \phi g_x
    - v_x \frac{\Delta \phi}{\Delta t}
    - \nabla \cdot (\phi v_x \boldsymbol{v})

    \Delta v_x
    = - \frac{1}{\rho} \frac{\Delta p}{\Delta x} \frac{\Delta t}{\phi}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi \boldsymbol{\tau}) \right]_x
      \frac{\Delta t}{\phi}
    - \frac{\Delta t}{\rho\phi} f^i_x
    + \Delta t g_x
    - v_x \frac{\Delta \phi}{\phi}
    - \nabla \cdot (\phi v_x \boldsymbol{v}) \frac{\Delta t}{\phi}

The term :math:`\beta` is introduced as an adjustable, dimensionless parameter
in the range :math:`[0;1]`, and determines the importance of the old pressure
values in the solution procedure (Langtangen et al. 2002).  A value of 0
corresponds to `Chorin's projection method`_ originally described
in `Chorin (1968)`_.

.. math::
    v_x^* = v_x^t + \Delta v_x

    v_x^* = v_x^t
    - \frac{\beta}{\rho} \frac{\Delta p^t}{\Delta x} \frac{\Delta t}{\phi^t}
    + \frac{1}{\rho} \left[ \nabla \cdot (\phi^t \boldsymbol{\tau}^t) \right]_x
      \frac{\Delta t}{\phi}
    - \frac{\Delta t}{\rho\phi} f^i_x
    + \Delta t g_x
    - v^t_x \frac{\Delta \phi}{\phi^t}
    - \nabla \cdot (\phi^t v_x^t \boldsymbol{v}^t) \frac{\Delta t}{\phi^t}

Here, :math:`\Delta x` denotes the cell spacing. The velocity found
(:math:`v_x^*`) is only a prediction of the fluid velocity at time
:math:`t+\Delta t`, since the estimate isn't constrained by the continuity
equation:

.. math::
    \frac{\Delta \phi^t}{\Delta t} + \nabla \cdot (\phi^t
    \boldsymbol{v}^{t+\Delta t}) = 0

The divergence of a scalar and vector can be `split`_:

.. math::
    \phi^t \nabla \cdot \boldsymbol{v}^{t+\Delta t} +
    \boldsymbol{v}^{t+\Delta t} \cdot \nabla \phi^t
    + \frac{\Delta \phi^t}{\Delta t} = 0

The predicted velocity is corrected using the new pressure (Langtangen et al.
2002):

.. math::
    \boldsymbol{v}^{t+\Delta t} = \boldsymbol{v}^*
    %- \frac{\Delta t}{\rho} \nabla \epsilon
    - \frac{\Delta t}{\rho \phi^t} \nabla \epsilon
    \quad \text{where} \quad
    \epsilon = p^{t+\Delta t} - \beta p^t

The above formulation of the future velocity is put into the continuity
equation:

.. math::
    \Rightarrow
    \phi^t \nabla \cdot
    \left( \boldsymbol{v}^* - \frac{\Delta t}{\rho \phi^t} \nabla \epsilon \right)
    +
    \left( \boldsymbol{v}^* - \frac{\Delta t}{\rho \phi^t} \nabla \epsilon \right)
    \cdot \nabla \phi^t + \frac{\Delta \phi^t}{\Delta t} = 0

.. math::
    \Rightarrow
    \phi^t \nabla \cdot
    \boldsymbol{v}^* - \frac{\Delta t}{\rho \phi^t} \phi^t \nabla^2 \epsilon
    + \nabla \phi^t \cdot \boldsymbol{v}^*
    - \nabla \phi^t \cdot \nabla \epsilon \frac{\Delta t}{\rho \phi^t}
    + \frac{\Delta \phi^t}{\Delta t} = 0

.. math::
    \Rightarrow
    \frac{\Delta t}{\rho} \nabla^2 \epsilon
    = \phi^t \nabla \cdot \boldsymbol{v}^*
    + \nabla \phi^t \cdot \boldsymbol{v}^*
    - \nabla \phi^t \cdot \nabla \epsilon \frac{\Delta t}{\rho \phi^t}
    + \frac{\Delta \phi^t}{\Delta t}

The pressure difference in time becomes a `Poisson equation`_ with added terms:

.. math::
    \Rightarrow
    \nabla^2 \epsilon
    = \frac{\nabla \cdot \boldsymbol{v}^* \phi^t \rho}{\Delta t}
    + \frac{\nabla \phi^t \cdot \boldsymbol{v}^* \rho}{\Delta t}
    - \frac{\nabla \phi^t \cdot \nabla \epsilon}{\phi^t}
    + \frac{\Delta \phi^t \rho}{\Delta t^2}

The right hand side of the above equation is termed the *forcing function*
:math:`f`, which is decomposed into two terms, :math:`f_1` and :math:`f_2`:

.. math::
    f_1 
    = \frac{\nabla \cdot \boldsymbol{v}^* \phi^t \rho}{\Delta t}
    + \frac{\nabla \phi^t \cdot \boldsymbol{v}^* \rho}{\Delta t}
    + \frac{\Delta \phi^t \rho}{\Delta t^2}

    f_2 =
    \frac{\nabla \phi^t \cdot \nabla \epsilon}{\phi^t}


During the `Jacobi iterative solution procedure`_ :math:`f_1` remains constant,
while :math:`f_2` changes value. For this reason, :math:`f_1` is found only
during the first iteration, while :math:`f_2` is updated every time. The value
of the forcing function is found as:

.. math::
    f = f_1 - f_2

Using second-order finite difference approximations of the Laplace operator
second-order partial derivatives, the differential equations become a system of
equations that is solved using `iteratively`_ using Jacobi updates. The total
number of unknowns is :math:`(n_x - 1)(n_y - 1)(n_z - 1)`.

The discrete Laplacian (approximation of the Laplace operator) can be obtained
by a finite-difference seven-point stencil in a three-dimensional, cubic
grid with cell spacing :math:`\Delta x, \Delta y, \Delta z`, considering the six
face neighbors:

.. math::
    \nabla^2 \epsilon_{i_x,i_y,i_z}  \approx 
    \frac{\epsilon_{i_x-1,i_y,i_z} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x+1,i_y,i_z}}{\Delta x^2}
    + \frac{\epsilon_{i_x,i_y-1,i_z} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x,i_y+1,i_z}}{\Delta y^2}

    + \frac{\epsilon_{i_x,i_y,i_z-1} - 2 \epsilon_{i_x,i_y,i_z}
    + \epsilon_{i_x,i_y,i_z+1}}{\Delta z^2}
    \approx f_{i_x,i_y,i_z}

Within a Jacobi iteration, the value of the unknowns (:math:`\epsilon^n`) is
used to find an updated solution estimate (:math:`\epsilon^{n+1}`).
The solution for the updated value takes the form:

.. math::
    \epsilon^{n+1}_{i_x,i_y,i_z}
    = \frac{-\Delta x^2 \Delta y^2 \Delta z^2 f_{i_x,i_y,i_z}
    + \Delta y^2 \Delta z^2 (\epsilon^n_{i_x-1,i_y,i_z} +
      \epsilon^n_{i_x+1,i_y,i_z})
    + \Delta x^2 \Delta z^2 (\epsilon^n_{i_x,i_y-1,i_z} +
      \epsilon^n_{i_x,i_y+1,i_z})
    + \Delta x^2 \Delta y^2 (\epsilon^n_{i_x,i_y,i_z-1} +
      \epsilon^n_{i_x,i_y,i_z+1})}
      {2 (\Delta x^2 \Delta y^2
      + \Delta x^2 \Delta z^2
      + \Delta y^2 \Delta z^2) }

The difference between the current and updated value is termed the *normalized
residual*:

.. math::
    r_{i_x,i_y,i_z} = \frac{(\epsilon^{n+1}_{i_x,i_y,i_z}
    - \epsilon^n_{i_x,i_y,i_z})^2}{(\epsilon^{n+1}_{i_x,i_y,i_z})^2}

Note that the :math:`\epsilon` values cannot be 0 due to the above normalization
of the residual.

The updated values are at the end of the iteration stored as the current values,
and the maximal value of the normalized residual is found. If this value is
larger than a tolerance criteria, the procedure is repeated. The iterative
procedure is ended if the number of iterations exceeds a defined limit. 

After the values of :math:`\epsilon` are found, they are used to find the new
pressures and velocities:

.. math::
    \bar{p}^{t+\Delta t} = \beta \bar{p}^t + \epsilon

.. math::
    \bar{\boldsymbol{v}}^{t+\Delta t} =
    \bar{\boldsymbol{v}}^* - \frac{\Delta t}{\rho\phi} \nabla \epsilon


Boundary conditions
-------------------
The lateral boundaries are periodic. This cannot be changed in the current
version of ``sphere``. This means that the fluid properties at the paired,
parallel lateral (:math:`x` and :math:`y`) boundaries are identical. A flow
leaving through one side reappears on the opposite side.

The top and bottom boundary conditions of the fluid grid can be either:
prescribed pressure (Dirichlet), or prescribed velocity (Neumann). The
(horizontal) velocities parallel to the boundaries are free to attain other
values (free slip). The Dirichlet boundary condition is enforced by keeping the
value of :math:`\epsilon` constant at the boundaries, e.g.:

.. math::
   \epsilon^{n+1}_{i_x,i_y,i_z = 1 \vee n_z}
   =
   \epsilon^{n}_{i_x,i_y,i_z = 1 \vee n_z}

The Neumann boundary condition of no flow across the boundary is enforced by
setting the gradient of :math:`\epsilon` perpendicular to the boundary to zero,
e.g.:

.. math::
   \nabla_z \epsilon^{n+1}_{i_x,i_y,i_z = 1 \vee n_z} = 0


Numerical implementation
------------------------
Ghost nodes

---




.. _Limache and Idelsohn (2006): http://www.cimec.org.ar/ojs/index.php/mc/article/view/486/464
.. _Cauchy stress tensor: https://en.wikipedia.org/wiki/Cauchy_stress_tensor
.. _`Chorin's projection method`: https://en.wikipedia.org/wiki/Projection_method_(fluid_dynamics)#Chorin.27s_projection_method
.. _`Chorin (1968)`: http://www.ams.org/journals/mcom/1968-22-104/S0025-5718-1968-0242392-2/S0025-5718-1968-0242392-2.pdf
.. _split: http://www.wolframalpha.com/input/?i=div(p+v)
.. _Poisson equation: https://en.wikipedia.org/wiki/Poisson's_equation
.. _`Jacobi iterative solution procedure`: http://www.rsmas.miami.edu/personal/miskandarani/Courses/MSC321/Projects/prjpoisson.pdf
.. _iteratively: https://en.wikipedia.org/wiki/Relaxation_(iterative_method)

