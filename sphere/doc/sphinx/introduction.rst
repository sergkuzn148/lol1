Introduction and Installation
=============================

The ``sphere``-software is used for three-dimensional discrete element method 
(DEM) particle simulations. The source code is written in C++, CUDA C and
Python, and is compiled by the user. The main computations are performed on the
graphics processing unit (GPU) using NVIDIA's general purpose parallel computing
architecture, CUDA. Simulation setup and data analysis is performed with the
included Python API.

The ultimate aim of the ``sphere`` software is to simulate soft-bedded subglacial
conditions, while retaining the flexibility to perform simulations of granular
material in other environments.

The purpose of this documentation is to provide the user with a walk-through of
the installation, work-flow, data-analysis and visualization methods of
``sphere``. In addition, the ``sphere`` internals are exposed to provide a way of
understanding of the discrete element method numerical routines taking place.

.. note:: Command examples in this document starting with the symbol ``$`` are
   meant to be executed in the shell of the operational system, and ``>>>``
   means execution in Python. `IPython <http://ipython.org>`_ is an excellent,
   interactive Python shell.

All numerical values in this document, the source code, and the configuration
files are typeset with strict respect to the SI unit system.


Requirements
------------

The build requirements are:

  * A Nvidia CUDA-supported version of Linux or Mac OS X (see the `CUDA toolkit
    release notes <http://docs.nvidia.com/cuda/cuda-toolkit-release-notes/index.html>`_ for more information)
  * `GNU Make <https://www.gnu.org/software/make/>`_
  * `CMake <http://www.cmake.org>`_, version 2.8 or newer
  * The `GNU Compiler Collection <http://gcc.gnu.org/>`_ (GCC)
  * The `Nvidia CUDA toolkit <https://developer.nvidia.com/cuda-downloads>`_,
    version 8.0 or newer

In Debian GNU/Linux, these dependencies can be installed by running::

 $ sudo apt-get install build-essential cmake nvidia-cuda-toolkit clang-3.8

Unfortunately, the Nvidia Toolkit is shipped under a non-free license. In order
to install it in Debian GNU/Linux, add ``non-free`` archives to your
``/etc/apt/sources.list``.

The runtime requirements are:

  * A `CUDA-enabled GPU <http://www.nvidia.com/object/cuda_gpus.html>`_ with
    compute capability 2.0 or greater.
  * A Nvidia CUDA-enabled GPU and device driver

Optional tools, required for simulation setup and data processing:

  * `Python <http://www.python.org/>`_
  * `Numpy <http://numpy.scipy.org>`_
  * `Matplotlib <http://matplotlib.org>`_
  * `Python bindings for VTK <http://www.vtk.org>`_
  * `Imagemagick <http://www.imagemagick.org/script/index.php>`_
  * `ffmpeg <http://ffmpeg.org/>`_. Soon to be replaced by avconv!

In Debian GNU/Linux, these dependencies can be installed by running::

 $ sudo apt-get install python python-numpy python-matplotlib python-vtk \
     imagemagick libav-tools

``sphere`` is distributed with a HTML and PDF build of the documentation. The
following tools are required for building the documentation:

  * `Sphinx <http://sphinx-doc.org>`_

    * `sphinxcontrib-programoutput <http://packages.python.org/sphinxcontrib-programoutput/>`_

  * `Doxygen <http://www.stack.nl/~dimitri/doxygen/>`_
  * `Breathe <http://michaeljones.github.com/breathe/>`_
  * `dvipng <http://www.nongnu.org/dvipng/>`_
  * `TeX Live <http://www.tug.org/texlive/>`_, including ``pdflatex``

In Debian GNU/Linux, these dependencies can be installed by running::

 $ sudo apt-get install python-sphinx python-pip doxygen dvipng \
     python-sphinxcontrib-programoutput texlive-full
 $ sudo pip install breathe

`Git <http://git-scm.com>`_ is used as the distributed version control system
platform, and the source code is maintained at `Github
<https://github.com/anders-dc/sphere/>`_. ``sphere`` is licensed under the `GNU
Public License, v.3 <https://www.gnu.org/licenses/gpl.html>`_.

.. note:: All Debian GNU/Linux runtime, optional, and documentation dependencies
   mentioned above can be installed by executing the following command from the
   ``doc/`` folder::

     $ make install-debian-pkgs


Obtaining sphere
----------------

The best way to keep up to date with subsequent updates, bugfixes and 
development, is to use the Git version control system. To obtain a local 
copy, execute::

 $ git clone git@github.com:anders-dc/sphere.git


Building ``sphere``
-------------------

``sphere`` is built using ``cmake``, the platform-specific C/C++ compilers,
and ``nvcc`` from the Nvidia CUDA toolkit.

If you instead plan to execute it on a Fermi GPU, change ``set(GPU_GENERATION
1)`` to ``set(GPU_GENERATION 0`` in ``CMakeLists.txt``.

In some cases the CMake FindCUDA module will have troubles locating the
CUDA samples directory, and will complain about ``helper_math.h`` not being 
found.

In that case, modify the ``CUDA_SDK_ROOT_DIR`` variable in
``src/CMakeLists.txt`` to the path where you installed the CUDA samples, and run
``cmake . && make`` again. Alternatively, copy ``helper_math.h`` from the CUDA
sample subdirectory ``common/inc/helper_math.h`` into the sphere ``src/``
directory, and run ``cmake`` and ``make`` again. Due to license restrictions,
sphere cannot be distributed with this file.

If you plan to run ``sphere`` on a Kepler GPU, execute the following commands
from the root directory::

 $ cmake . && make

NOTE: If your system does not have a GCC compiler compatible with the installed
CUDA version (e.g. GCC-5 for CUDA 8), you will see errors at the linker stage.  
In that case, try using ``clang-3.8`` as the C and C++ compiler instead::

 $ rm -rf CMakeCache.txt CMakeFiles/
 $ export CC=$(which clang-3.8) && export CXX=$(which clang++-3.8) && cmake . && make

After a successfull installation, the ``sphere`` executable will be located
in the root folder. To make sure that all components are working correctly,
execute::

 $ make test

Disclaimer: On some systems the Navier-Stokes related tests will fail.  If you 
do encounter these problems, but do not plan on using the Navier Stokes solver 
for fluid dynamics, carry on.

If successful the Makefiles will create the required data folders, object
files, as well as the ``sphere`` executable in the root folder. Issue the
following commands to check the executable::

 $ ./sphere --version

The output should look similar to this:

.. program-output:: ../../sphere --version

The documentation can be read in the `reStructuredText
<http://docutils.sourceforge.net/docs/ref/rst/restructuredtext.html>`_-format in
the ``doc/sphinx/`` folder, or in the HTML or PDF formats in the folders
``doc/html`` and ``doc/pdf``.

Optionally, the documentation can be built using the following commands::

 $ cd doc/sphinx
 $ make html
 $ make latexpdf

To see all available output formats, execute::

 $ make help


Updating sphere
---------------

To update your local version, type the following commands in the ``sphere`` root 
directory::

 $ git pull && cmake . && make


Work flow
---------

After compiling the ``sphere`` binary, the procedure of a creating and handling
a simulation is typically arranged in the following order:

  * Setup of particle assemblage, physical properties and conditions using the
    Python API (``python/sphere.py``).
  * Execution of ``sphere`` software, which simulates the particle behavior as a
    function of time, as a result of the conditions initially specified in the
    input file.
  * Inspection, analysis, interpretation and visualization of ``sphere`` output
    in Python, and/or scene rendering using the built-in ray tracer.
