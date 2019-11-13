find_package(PythonInterp REQUIRED)

add_test(io_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/io_tests.py)

add_test(wall_contact_model_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/contactmodel_wall.py)

add_test(contact_model_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/contactmodel.py)

add_test(wall_contact_model_young_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/contactmodel_wall_young.py)

add_test(contact_model_young_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/contactmodel_young.py)

add_test(io_tests_fluid ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/io_tests_fluid.py)

add_test(porosity_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/porosity_tests.py)

add_test(memory_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/memcheck_tests.py)

add_test(bond_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/bond_tests.py)

add_test(cfd_tests ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests.py)

add_test(cfd_tests_neumann ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_neumann.py)

add_test(cfd_tests_neumann-c_v=0.1 ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_neumann-c_v=0.1.py)

add_test(fluid_particle_interaction ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/fluid_particle_interaction.py)

add_test(cfd_tests_darcy ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_darcy.py)

add_test(cfd_tests_neumann_darcy ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_neumann_darcy.py)

add_test(fluid_particle_interaction_darcy ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/fluid_particle_interaction_darcy.py)

add_test(cfd_tests_darcy_particles ${PYTHON_EXECUTABLE} 
    ${CMAKE_CURRENT_BINARY_DIR}/cfd_tests_darcy_particles.py)
