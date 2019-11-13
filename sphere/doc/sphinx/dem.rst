Discrete element method
=======================
Granular material is a very common form of matter, both in nature and industry.
It can be defined as material consisting of interacting, discrete particles.
Common granular materials include gravels, sands and soils, ice bergs,
asteroids, powders, seeds, and other foods. Over 75% of the raw materials that
pass through industry are granular. This wide occurrence has driven the desire
to understand the fundamental mechanics of the material.

Contrary to other common materials such as gases, liquids and solids, a general
mathematical formulation of it's behavior hasn't yet been found. Granular
material can, however, display states that somewhat resemble gases, fluids and
solids.

..  The discrete element method (or distinct element method) was initially
    formulated by Cundall and Strack (1979). It simulates the physical behavior and
    interaction of discrete, unbreakable particles, with their own mass and inertia,
    under the influence of e.g. gravity and boundary conditions such as moving
    walls. By discretizing time into small time steps, explicit integration of
    Newton's second law of motion is used to predict the new position and kinematic
    values for each particle from the previous sums of forces. This Lagrangian
    approach is ideal for simulating discontinuous materials, such as granular
    matter.
    The complexity of the computations is kept low by representing the particles as
    spheres, which keeps contact-searching algorithms simple.

The `Discrete Element Method
<https://en.wikipedia.org/wiki/Discrete_element_method>`_ (DEM) is a numerical
method that can be used to
simulate the interaction of particles. Originally derived from
`Molecular Dynamics <https://en.wikipedia.org/wiki/Molecular_dynamics>`_,
it simulates particles as separate entities, and calculates their positions,
velocities, and accelerations through time. See Cundall and Strack (1979) and
`this blog post
<http://anders-dc.github.io/2013/10/16/the-discrete-element-method/>`_ for
general introduction to the DEM. The following sections will highlight the
DEM implementation in ``sphere``. Some of the details are also described in
Damsgaard et al. 2013. In the used notation, a bold symbol denotes a
three-dimensional vector, and a dot denotes that the entity is a temporal
derivative.

Contact search
--------------
Homogeneous cubic grid.

.. math::
   \delta_n^{ij} = ||\boldsymbol{x}^i - \boldsymbol{x}^j|| - (r^i + r^j)

where :math:`r` is the particle radius, and :math:`\boldsymbol{x}` denotes the
positional vector of a particle, and :math:`i` and :math:`j` denote the indexes
of two particles. Negative values of :math:`\delta_n` denote that the particles
are overlapping.


Contact interaction
-------------------
Now that the inter-particle contacts have been identified and characterized by
their overlap, the resulting forces from the interaction can be resolved. The
interaction is decomposed into normal and tangential components, relative to the
contact interface orientation. The normal vector to the contact interface is
found by:

.. math::
   \boldsymbol{n}^{ij} = 
   \frac{\boldsymbol{x}^i - \boldsymbol{x}^j}
   {||\boldsymbol{x}^i - \boldsymbol{x}^j||}

The contact velocity :math:`\dot{\boldsymbol{\delta}}` is found by:

.. math::
   \dot{\boldsymbol{\delta}}^{ij} =
   (\boldsymbol{x}^i - \boldsymbol{x}^j)
   + (r^i + \frac{\delta_n^{ij}}{2})
     (\boldsymbol{n}^{ij} \times \boldsymbol{\omega}^{i})
   + (r^j + \frac{\delta_n^{ij}}{2})
     (\boldsymbol{n}^{ij} \times \boldsymbol{\omega}^{j})

The contact velocity is decomposed into normal and tangential components,
relative to the contact interface. The normal component is:

.. math::
   \dot{\delta}^{ij}_n =
   -(\dot{\boldsymbol{\delta}}^{ij} \cdot \boldsymbol{n}^{ij})

and the tangential velocity component is found as:

.. math::
   \dot{\boldsymbol{\delta}}^{ij}_t =
   \dot{\boldsymbol{\delta}}^{ij}
   - \boldsymbol{n}^{ij}
     (\boldsymbol{n}^{ij} \cdot \dot{\boldsymbol{\delta}}^{ij})

where :math:`\boldsymbol{\omega}` is the rotational velocity vector of a
particle. The total tangential displacement on the contact plane is found
incrementally:

.. math::
   \boldsymbol{\delta}_{t,\text{uncorrected}}^{ij} =
   \int_0^{t_c} 
   \dot{\boldsymbol{\delta}}^{ij}_t \Delta t

where :math:`t_c` is the duration of the contact and :math:`\Delta t` is the
computational time step length. The tangential contact interface displacement is
set to zero when a contact pair no longer overlaps. At each time step, the value
of :math:`\boldsymbol{\delta}_t` is corrected for rotation of the contact
interface:

.. math::
   \boldsymbol{\delta}_t^{ij} = \boldsymbol{\delta}_{t,\text{uncorrected}}^{ij}
   - (\boldsymbol{n}
     (\boldsymbol{n} \cdot \boldsymbol{\delta}_{t,\text{uncorrected}}^{ij})

With all the geometrical and kinetic components determined, the resulting forces
of the particle interaction can be determined using a contact model. ``sphere``
features only one contact model in the normal direction to the contact; the
linear-elastic-viscous (*Hookean* with viscous damping, or *Kelvin-Voigt*)
contact model. The resulting force in the normal direction of the contact
interface on particle :math:`i` is:

.. math::
   \boldsymbol{f}_n^{ij} = \left(
   -k_n \delta_n^{ij} -\gamma_n \dot{\delta_n}^{ij}
   \right) \boldsymbol{n}^{ij}

The parameter :math:`k_n` is the defined `spring coefficient
<https://en.wikipedia.org/wiki/Hooke's_law>`_ in the normal direction of the
contact interface, and :math:`\gamma_n` is the defined contact interface
viscosity, also in the normal direction. The loss of energy in this interaction
due to the viscous component is for particle :math:`i` calculated as:

.. math::
    \dot{e}^i_v = \gamma_n (\dot{\delta}^{ij}_n)^2

The tangential force is determined by either a viscous-frictional contact model,
or a elastic-viscous-frictional contact model. The former contact model is very
computationally efficient, but somewhat inaccurate relative to the mechanics of
real materials.  The latter contact model is therefore the default, even though
it results in longer computational times. The tangential force in the
visco-frictional contact model:

.. math::
   \boldsymbol{f}_t^{ij} = -\gamma_t \dot{\boldsymbol{\delta}_t}^{ij}

:math:`\gamma_n` is the defined contact interface viscosity in the tangential
direction. The tangential displacement along the contact interface
(:math:`\boldsymbol{\delta}_t`) is not calculated and stored for this contact
model. The tangential force in the more realistic elastic-viscous-frictional
contact model:

.. math::
   \boldsymbol{f}_t^{ij} =
   -k_t \boldsymbol{\delta}_t^{ij} -\gamma_t \dot{\boldsymbol{\delta}_t}^{ij}

The parameter :math:`k_n` is the defined spring coefficient in the tangential
direction of the contact interface. Note that the tangential force is only
found if the tangential displacement (:math:`\delta_t`) or the tangential
velocity (:math:`\dot{\delta}_t`) is non-zero, in order to avoid division by
zero. Otherwise it is defined as being :math:`[0,0,0]`.

For both types of contact model, the tangential force is limited by the Coulomb
criterion of static and dynamic friction:

.. math::
   ||\boldsymbol{f}^{ij}_t|| \leq
   \begin{cases}
   \mu_s ||\boldsymbol{f}^{ij}_n|| &
       \text{if} \quad ||\boldsymbol{f}_t^{ij}|| = 0 \\
   \mu_d ||\boldsymbol{f}^{ij}_n|| &
       \text{if} \quad ||\boldsymbol{f}_t^{ij}|| > 0
   \end{cases}

If the elastic-viscous-frictional contact model is used and the Coulomb limit is
reached, the tangential displacement along the contact interface is limited to
this value:

.. math::
   \boldsymbol{\delta}_t^{ij} =
   \frac{1}{k_t} \left(
   \mu_d ||\boldsymbol{f}_n^{ij}||
   \frac{\boldsymbol{f}^{ij}_t}{||\boldsymbol{f}^{ij}_t||}
   + \gamma_t \dot{\boldsymbol{\delta}}_t^{ij} \right)

If the tangential force reaches the Coulomb limit, the energy lost due to
frictional dissipation is calculated as:

.. math::
   \dot{e}^i_s = \frac{||\boldsymbol{f}^{ij}_t
   \dot{\boldsymbol{\delta}}_t^{ij} \Delta t||}{\Delta t}

The loss of energy by viscous dissipation in the tangential direction is not
found.


Temporal integration
--------------------
In the DEM, the time is discretized into small steps (:math:`\Delta t`). For each time
step, the entire network of contacts is resolved, and the resulting forces and
torques for each particle are found. With these values at hand, the new
linear and rotational accelerations can be found using
`Newton's second law <https://en.wikipedia.org/wiki/Newton%27s_laws_of_motion>`_
of the motion of solid bodies. If a particle with mass :math:`m` at a point in time
experiences a sum of forces denoted :math:`\boldsymbol{F}`, the resultant acceleration
(:math:`\boldsymbol{a}`) can be found by rearranging Newton's second law:

.. math::
   \boldsymbol{F} = m \boldsymbol{a} \Rightarrow \boldsymbol{a} = \frac{\boldsymbol{F}}{m}

The new velocity and position is found by integrating the above equation
with regards to time. The simplest integration scheme in this regard is the 
`Euler method <https://en.wikipedia.org/wiki/Euler_method>`_:

.. math::
   \boldsymbol{v} = \boldsymbol{v}_{old} + \boldsymbol{a} \Delta t

.. math::
   \boldsymbol{p} = \boldsymbol{p}_{old} + \boldsymbol{v} \Delta t

