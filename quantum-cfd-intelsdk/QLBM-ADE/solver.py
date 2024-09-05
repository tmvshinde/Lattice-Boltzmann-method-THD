"""
Time stepping for 1D
Time stepping for 2D
"""

import subprocess
from dataclasses import dataclass

from preprocessing import (
    norm,
    normalize_vector,
    decomposing,
    conversion,
    interpolate_initial_1D,
    interpolate_initial_2D,
    interpolate_initial_2d,
    initial_state_circle,
)
from postprocessing import (
    extracting_states,
    plotting_1D,
    plotting_2D,
    analytic1d,
    analytic2d,
    setup_logging,
    image_to_gif,
)

import numpy as np
import logging


@dataclass
class Parameters1D:
    c_init: float = 0.0
    c_max: float = 1.0
    x_max: int = 3
    velocity: float = 0.2
    steps: int = 1
    lattices: int = 16


@dataclass
class Parameters2D:
    c_init: float = 0.0
    c_max: float = 1.0
    x_max: int = 1.0
    y_max: int = 1.0
    x_vel: float = 0.2
    y_vel: float = 0.2
    steps: int = 1


def Time_stepping_1d(num_qubits_work, num_states, parameters, use_1D):
    for i in range(parameters.steps):
        # Start with decomposing the state- where we get the .qasm file
        print("\nITERATION", i, "\n")

        if i == 0:
            # for iter 0 take the vector from the initialization step decompose and move forward
            initial_vector = interpolate_initial_1D(
                num_states, parameters.x_max, parameters.c_init, parameters.c_max
            )
            logging.info("init vector : " + str(initial_vector))
            factor = norm(initial_vector)
            logging.info("Euclidean Norm : " + str(factor))
            normalize_initial_vector = normalize_vector(
                initial_vector, factor, parameters.lattices
            )
            logging.info("normalized vector : " + str(normalize_initial_vector))
            decomposing(normalize_initial_vector, i, parameters.lattices)
        else:
            # for iter 1 take vector from the extract states function do if else loop
            factor = norm(physical_concentration)
            logging.info("Euclidean Norm : " + str(factor))
            renormalized_vector = normalize_vector(
                physical_concentration, factor, parameters.lattices
            )
            logging.info("normalized vector : " + str(renormalized_vector))
            decomposing(renormalized_vector, i, parameters.lattices)

        # converting the .qasm to .cpp with the help of the intel python interface

        subprocess.run(
            [
                "/glob/development-tools/intel-quantum-sdk/intel-quantum-compiler",
                "-B",
                "encoding_x02.qasm",
            ]
        )

        # converting OpenQASM to C++

        conversion(use_1D)

        # Calculation in Intel Quantum SDK
        #

        subprocess.run(
            [
                "/glob/development-tools/intel-quantum-sdk/intel-quantum-compiler",
                "1d-ade.cpp",
            ]
        )
        # subprocess.run(["mpirun", "-n", "2", "./1d-ade"])
        subprocess.run(["./1d-ade"])

        probfiles = ["qlbm-states-timestep.txt"]
        physical_concentration = extracting_states(
            probfiles, factor, parameters.lattices, use_1D
        )
        logging.info("The concentration : \n" + str(physical_concentration))
        step = np.linspace(0, 40, 41, endpoint=True)

        for item in step:
            if i == item:
                analytic_sol = analytic1d(parameters, num_states, i)
                logging.info("The analytic concentration : \n" + str(analytic_sol))
                plotting_1D(
                    physical_concentration, analytic_sol, parameters.lattices, i
                )

    print("Finished!!!!")

    return physical_concentration


def Time_stepping_2d(
    num_qubits_work, num_qubits_lattice_side, num_lattices, parameters, use_1D
):
    for i in range(parameters.steps):
        # Start with decomposing the state- where we get the .qasm file
        # print("\nITERATION", i, "\n")
        logging.info(" ITERATION : " + str(i))

        if i == 0:
            # for iter 0 take the vector from the initialization step decompose and move forward
            # initial_vector = initial_state_circle(
            #     num_qubits_lattice_side, radius=5, c_max=parameters.c_max
            # )
            initial_vector = interpolate_initial_2D(
                num_qubits_lattice_side, num_lattices, parameters.c_max
            )
            factor = norm(initial_vector)

            logging.info("Euclidean Norm : " + str(factor))

            normalize_initial_vector = normalize_vector(
                initial_vector, factor, num_lattices
            )
            decomposing(normalize_initial_vector, i, num_lattices)
        else:
            # for iter 1 take vector from the extract states function do if else loop
            factor = norm(physical_concentration)

            logging.info("Euclidean Norm : " + str(factor))

            renormalized_vector = normalize_vector(
                physical_concentration, factor, num_lattices
            )
            decomposing(renormalized_vector, i, num_lattices)

        # converting the .qasm to .cpp with the help of the intel python interface

        subprocess.run(
            [
                "/glob/development-tools/intel-quantum-sdk/intel-quantum-compiler",
                "-B",
                "encoding_x02.qasm",
            ]
        )

        # converting OpenQASM to C++

        conversion(use_1D)

        # Calculation in Intel Quantum SDK
        #

        subprocess.run(
            [
                "/glob/development-tools/intel-quantum-sdk/intel-quantum-compiler",
                "2d-ade.cpp",
            ]
        )
        # subprocess.run(["mpirun", "-n", "2", "./1d-ade"])
        subprocess.run(
            ["./2d-ade", str(i), str(parameters.x_vel), str(parameters.y_vel)]
        )

        probfiles = ["qlbm-2d-states.txt"]
        physical_concentration = extracting_states(
            probfiles, factor, num_lattices, use_1D
        )

        logging.info("The concentration : \n" + str(physical_concentration))

        analytic_sol = analytic2d(parameters, num_lattices, i)
        plotting_2D(physical_concentration, num_lattices, i, analytic_sol)

    print("Finished!!!!")


def run_1D_QLBM(use_1D):
    parameters = Parameters1D()
    parameters.c_init = 0.0
    parameters.c_max = 1.0
    parameters.x_max = 6.0
    parameters.x_vel = 0.2
    parameters.steps = 1
    parameters.lattices = 16

    logging.info(
        "Parameters set to : "
        + "x_velocity : "
        + str(parameters.x_vel)
        + ", drop placement co-ordinate : ("
        + str(parameters.x_max)
        + "), drop concentration : "
        + str(parameters.c_max)
    )

    num_qubits_lattice = np.log2(parameters.lattices)
    num_qubits_work = num_qubits_lattice + 1
    num_states = 2 * parameters.lattices

    Concentration = Time_stepping_1d(num_qubits_work, num_states, parameters, use_1D)

    return Concentration


def run_2D_QLBM(use_1D):
    num_lattice_pts = 16  # number of lattices in 1 direction
    parameters = Parameters2D()
    parameters.c_init = 0.0
    parameters.c_max = 1.0
    parameters.x_max = 3.0
    parameters.y_max = 3.0
    parameters.x_vel = 0.2
    parameters.y_vel = 0.2
    parameters.steps = 1

    logging.info(
        "Parameters set to : "
        + "x_velocity : "
        + str(parameters.x_vel)
        + ", y_vel : "
        + str(parameters.y_vel)
        + ", drop placement co-ordinate : ("
        + str(parameters.x_max)
        + ","
        + str(parameters.y_max)
        + "), drop concentration : "
        + str(parameters.c_max)
    )

    num_lattices = num_lattice_pts * num_lattice_pts  # total number of lattices
    num_qubits_lattice_side = np.log2(num_lattice_pts)
    num_qubits_work = 2 * num_qubits_lattice_side + 3 + 1

    Time_stepping_2d(
        num_qubits_work, num_qubits_lattice_side, num_lattices, parameters, use_1D
    )

    image_to_gif(parameters.steps)


def main():
    # specify true if want to do 1D and false if 2D
    setup_logging(logfile="report.log")
    logger = logging.getLogger(__name__)

    use_1D = False

    # when changing the lattice size, change the number of qubits work in cpp main()
    if use_1D == True:
        logger.info("Solving 1D problem")
        run_1D_QLBM(use_1D)
    else:
        logger.info("Solving 2D problem")
        run_2D_QLBM(use_1D)


if __name__ == "__main__":
    main()
