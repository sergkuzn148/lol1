Python API
==========
The Python module ``sphere`` is intended as the main interface to the ``sphere``
application. It is recommended to use this module for simulation setup,
simulation execution, and analysis of the simulation output data.

In order to use the API, the file ``sphere.py`` must be placed in the same
directory as the Python files. 

Sample usage
------------
Below is a simple, annotated example of how to setup, execute, and post-process
a ``sphere`` simulation.  The example is also found in the ``python/`` folder as
``collision.py``.

.. literalinclude:: ../../python/collision.py
   :language: python
   :linenos:

The full documentation of the ``sphere`` Python API can be found below.


The ``sphere`` module
---------------------
.. automodule:: sphere
   :members:

